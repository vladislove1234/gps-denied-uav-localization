"""Where along the ground the onboard matcher is likely to get a fix.

`SelfSimilarity` asks the real matcher: a footprint-sized patch of the class map is scored against
its own surroundings. A unique, sharp peak means a fix is likely; a ridge (one straight road),
repeated look-alike structure or a flat surface (open field) means it is not. The map is compared
with itself, so this ranks terrain rather than predicting real fix rates: no season change, stale
map or drone-side segmentation error.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol

import cv2
import numpy as np
import torch
from scipy.ndimage import map_coordinates
from tqdm.auto import tqdm

from uavloc.data.satellite import GeoImage
from uavloc.geo import GeoBounds, LatLon, footprint_m
from uavloc.matching.correlator import (MIN_WINDOW_STD, CorrelatorConfig, PositionCorrelator,
                                        patch_size_for_footprint_m, posterior, weighted_coverage, z_scored)
from uavloc.matching.learned import batched_ncc, class_presence
from uavloc.planning.classmap import PROBABILITY_LEVELS, ProgressHook

NORTH_UP_DEG = 0.0
S1_CHANNEL, COVERAGE_CHANNEL = 0, 1
WINDOW_MARGIN = 1.5   # window half-size / (longest side / 2); > sqrt(2), so a patch at any heading fits inside
BATCH_POINTS = 64


class LocalizabilityModel(Protocol):
    def blind_map(self, class_map: GeoImage, area: GeoBounds, altitude_m: float, spacing_m: float,
                  progress: ProgressHook | None = None) -> GeoImage:
        """`(rows, cols, 1)` float32 over `area`: 1 = a fix is unlikely, 0 = likely."""
        ...


def sample_grid(area: GeoBounds, spacing_m: float) -> GeoImage:
    """An empty `(rows, cols, 1)` raster over `area` with cells of about `spacing_m`."""
    width_m, height_m = area.size_m
    rows, cols = max(round(height_m / spacing_m), 1), max(round(width_m / spacing_m), 1)
    return GeoImage(np.zeros((rows, cols, 1), np.float32), area.top_left, area.bottom_right)


class ClassMapFeatures:
    """`FeatureExtractor` over a uint8 class-probability map; only the satellite side exists."""

    def __init__(self, class_map: GeoImage, match_gsd_m: float) -> None:
        self.class_map = class_map
        self.match_gsd_m = match_gsd_m

    def drone(self, image: object, width_m: float, height_m: float) -> torch.Tensor:
        raise NotImplementedError("self-similarity uses map patches on both sides")

    def satellite_patch(self, center: LatLon, yaw_deg: float, width_m: float, height_m: float) -> torch.Tensor:
        window = self.class_map.window(center, WINDOW_MARGIN * max(width_m, height_m) / 2)
        patch = window.crop_rotated(center, yaw_deg, width_m, height_m, out_mpp=self.match_gsd_m, allow_outside=True)
        probabilities = patch.astype(np.float32).transpose(2, 0, 1) / PROBABILITY_LEVELS
        return torch.from_numpy(np.ascontiguousarray(probabilities))[None]


def batched_weighted_ncc(drone: torch.Tensor, satellite: torch.Tensor, class_weights: tuple[float, ...]) -> list[np.ndarray]:
    """`weighted_ncc_heatmap` for a batch `(B, C, h, w)` x `(B, C, H, W)` in one FFT pass, one z-scored map per sample.

    A class takes part in a sample only where it is present in that sample's drone map, as in `class_ncc_maps`.
    """
    weighted = [c for c, weight in enumerate(class_weights) if weight > 0]
    ncc = batched_ncc(drone[:, weighted], satellite[:, weighted], MIN_WINDOW_STD)
    weights = torch.tensor([class_weights[c] for c in weighted]) * class_presence(drone)[:, weighted].float()
    weight_sums = weights.sum(1)
    out = []
    for b in range(drone.shape[0]):
        if weight_sums[b] == 0:
            out.append(np.zeros(ncc.shape[2:], np.float32))
        else:
            out.append(z_scored((weights[b, :, None, None] * ncc[b]).sum(0) / weight_sums[b]))
    return out


class SelfSimilarity:
    def __init__(self, matcher: CorrelatorConfig, match_gsd_m: float, s_max_m: float, c_min: float) -> None:
        if matcher.matching != "weighted-ncc" or matcher.window != "rect":
            raise ValueError("self-similarity scores the onboard weighted NCC over a rect window")
        self.matcher = matcher
        self.match_gsd_m = match_gsd_m
        self.s_max_m = s_max_m
        self.c_min = c_min

    def scores(self, class_map: GeoImage, area: GeoBounds, altitude_m: float, spacing_m: float,
               progress: ProgressHook | None = None) -> GeoImage:
        """`(rows, cols, 2)` over `area`: posterior long-axis std (m) and weighted-class coverage per grid point."""
        grid = sample_grid(area, spacing_m)
        rows, cols = grid.pixels.shape[:2]
        out = np.zeros((rows, cols, 2), np.float32)
        features = ClassMapFeatures(class_map, self.match_gsd_m)
        correlator = PositionCorrelator(features, self.matcher)
        footprint = footprint_m(altitude_m, self.matcher.fov_w_deg, self.matcher.aspect_w_over_h)
        patch_size = patch_size_for_footprint_m(self.matcher, footprint)
        points = [(r, c) for r in range(rows) for c in range(cols)]
        for start in tqdm(range(0, len(points), BATCH_POINTS), desc="localizability"):
            chunk = points[start:start + BATCH_POINTS]
            prepared = []
            for r, c in chunk:
                center = grid.to_latlon(c + 0.5, r + 0.5)
                prepared.append(correlator.prepare(features.satellite_patch(center, NORTH_UP_DEG, *footprint),
                                                   center, NORTH_UP_DEG, patch_size))
            drones = torch.cat([drone for drone, _, _, _ in prepared])
            heatmaps = batched_weighted_ncc(drones, torch.cat([sat for _, sat, _, _ in prepared]),
                                            self.matcher.class_weights)
            for (r, c), (drone, _, empty, _), heatmap in zip(chunk, prepared, heatmaps):
                coverage = weighted_coverage(drone, self.matcher.class_weights)
                s1_m, _, _, _ = posterior(replace(empty, heatmap=heatmap, coverage=coverage), self.matcher).axes
                out[r, c] = s1_m, coverage
            if progress:
                progress("localizability", start + len(chunk), len(points))
        return grid.with_pixels(out)

    def blind_from_scores(self, scores: GeoImage) -> GeoImage:
        s1_m, coverage = scores.pixels[..., S1_CHANNEL], scores.pixels[..., COVERAGE_CHANNEL]
        blind = (s1_m > self.s_max_m) | (coverage < self.c_min) | ~np.isfinite(s1_m)
        return scores.with_pixels(blind.astype(np.float32)[..., None])

    def blind_map(self, class_map: GeoImage, area: GeoBounds, altitude_m: float, spacing_m: float,
                  progress: ProgressHook | None = None) -> GeoImage:
        return self.blind_from_scores(self.scores(class_map, area, altitude_m, spacing_m, progress))


class ClassMix:
    """Fallback: blind where weighted classes cover too little of the footprint. Fast, but only a proxy."""

    def __init__(self, class_weights: tuple[float, ...], fov_w_deg: float, aspect_w_over_h: float,
                 min_coverage: float) -> None:
        self.class_weights = class_weights
        self.fov_w_deg = fov_w_deg
        self.aspect_w_over_h = aspect_w_over_h
        self.min_coverage = min_coverage

    def blind_map(self, class_map: GeoImage, area: GeoBounds, altitude_m: float, spacing_m: float,
                  progress: ProgressHook | None = None) -> GeoImage:
        weighted = np.isin(class_map.pixels.argmax(-1), np.flatnonzero(np.asarray(self.class_weights) > 0))
        width_m, height_m = footprint_m(altitude_m, self.fov_w_deg, self.aspect_w_over_h)
        mpp_x, mpp_y = class_map.mpp_xy
        kernel = (max(round(width_m / mpp_x), 1), max(round(height_m / mpp_y), 1))
        coverage = cv2.blur(weighted.astype(np.float32), kernel)
        grid = sample_grid(area, spacing_m)
        rows, cols = grid.pixels.shape[:2]
        centres = [class_map.to_px(grid.to_latlon(c + 0.5, r + 0.5)) for r in range(rows) for c in range(cols)]
        xs, ys = np.asarray(centres).T
        sampled = map_coordinates(coverage, [ys - 0.5, xs - 0.5], order=1, mode="nearest").reshape(rows, cols)
        return grid.with_pixels((sampled < self.min_coverage).astype(np.float32)[..., None])
