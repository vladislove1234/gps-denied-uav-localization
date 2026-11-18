"""A search window sized from a belief instead of from the frame (027).

The correlator's rect window reaches only ~±9 m across the frame, whatever the
uncertainty. Here the window is the heading-up box that covers a set of belief
points (particles, hypothesis ellipses), never smaller than the rect window and
capped. Scoring uses the FFT NCC, which equals `weighted_ncc_heatmap` to ~1e-7
and stays fast on a 200 m window where direct conv2d takes seconds.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch

from uavloc.geo import LatLon, east_north_m, offset_latlon, rotate_to_heading_up_array, rotate_to_north_up_array
from uavloc.matching.correlator import (MIN_PRESENT_SHARE, MIN_WINDOW_STD, PositionCorrelator, SearchSurface,
                                        drone_class_shares, weighted_coverage, z_scored)
from uavloc.matching.learned import batched_ncc


@dataclass(frozen=True)
class SearchWindow:
    """How far the drone centre may move from `center`, `(right_m, down_m)` in the heading-up patch."""

    center: LatLon
    yaw_deg: float
    half_m: tuple[float, float]
    capped: bool = False


def to_heading_up(origin: LatLon, center: LatLon, points_en_m: np.ndarray, yaw_deg: float) -> np.ndarray:
    """`(N, 2)` east/north offsets from `origin` → `(N, 2)` (right, down) offsets from `center`."""
    center_east_m, center_north_m = east_north_m(origin, center)
    right, down = rotate_to_heading_up_array(points_en_m[:, 0] - center_east_m,
                                             -(points_en_m[:, 1] - center_north_m), yaw_deg)
    return np.stack([right, down], axis=1)


def window_around(origin: LatLon, points_en_m: np.ndarray, yaw_deg: float, floor_half_m: tuple[float, float],
                  cap_half_m: float, margin_m: float, anchor_en_m: np.ndarray) -> SearchWindow:
    """The heading-up box around `points_en_m` (east/north from `origin`), at least `floor_half_m` per axis.

    A box wider than `cap_half_m` is cut to the cap around `anchor_en_m` (the estimate).
    """
    right, down = rotate_to_heading_up_array(points_en_m[:, 0], -points_en_m[:, 1], yaw_deg)
    low, high = np.array([right.min(), down.min()]), np.array([right.max(), down.max()])
    center_rd = (low + high) / 2
    half = np.maximum((high - low) / 2 + margin_m, floor_half_m)
    capped = bool((half > cap_half_m).any())
    if capped:
        anchor_r, anchor_d = rotate_to_heading_up_array(anchor_en_m[0], -anchor_en_m[1], yaw_deg)
        center_rd = np.where(half > cap_half_m, [anchor_r, anchor_d], center_rd)
        half = np.minimum(half, cap_half_m)
    east_m, south_m = rotate_to_north_up_array(center_rd[0], center_rd[1], yaw_deg)
    return SearchWindow(offset_latlon(origin, float(east_m), float(-south_m)), yaw_deg, (float(half[0]), float(half[1])),
                        capped)


def patch_size_m(correlator: PositionCorrelator, footprint_m: tuple[float, float],
                 half_m: tuple[float, float]) -> tuple[float, float]:
    """Satellite patch `(width_m, height_m)` whose valid offsets reach `half_m` after both border crops."""
    cfg = correlator.config
    kept_sat, kept_drone = 1 - 2 * cfg.sat_feature_crop_pct, 1 - 2 * cfg.drone_feature_crop_pct
    return ((footprint_m[0] * kept_drone + 2 * half_m[0]) / kept_sat,
            (footprint_m[1] * kept_drone + 2 * half_m[1]) / kept_sat)


def fft_weighted_ncc_heatmap(drone: torch.Tensor, satellite: torch.Tensor, class_weights: tuple[float, ...]) -> np.ndarray:
    """`weighted_ncc_heatmap` computed with FFT NCC; all zeros when no weighted class is present."""
    shares = drone_class_shares(drone)
    present = [c for c, weight in enumerate(class_weights) if weight > 0 and shares[c] >= MIN_PRESENT_SHARE]
    out_shape = (satellite.shape[2] - drone.shape[2] + 1, satellite.shape[3] - drone.shape[3] + 1)
    if not present:
        return np.zeros(out_shape, np.float32)
    ncc = batched_ncc(drone[:, present], satellite[:, present], MIN_WINDOW_STD)[0]
    weights = torch.tensor([class_weights[c] for c in present], dtype=ncc.dtype)
    return z_scored((weights[:, None, None] * ncc).sum(0) / weights.sum())


def score_window(correlator: PositionCorrelator, drone_full: torch.Tensor, footprint_m: tuple[float, float],
                 window: SearchWindow) -> SearchSurface:
    """The z-scored surface over `window`: weighted NCC, or the correlator's learned scorer when it has one.

    Raises ValueError if the patch leaves the map raster.
    """
    cfg = correlator.config
    drone, sat, empty, _ = correlator.prepare(drone_full, window.center, window.yaw_deg,
                                              patch_size_m(correlator, footprint_m, window.half_m))
    if cfg.matching == "learned":
        heatmap = correlator.scorer(drone, sat, None)
    else:
        heatmap = fft_weighted_ncc_heatmap(drone, sat, cfg.class_weights)
    return replace(empty, heatmap=heatmap, coverage=weighted_coverage(drone, cfg.class_weights))


def cells_of(surface: SearchSurface, offsets_rd_m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest `(rows, cols)` of `(N, 2)` (right, down) offsets, and which of them fall inside the heatmap."""
    rows, cols = surface.cell_of_offset((offsets_rd_m[:, 0], offsets_rd_m[:, 1]))
    rows, cols = np.rint(rows).astype(int), np.rint(cols).astype(int)
    height, width = surface.heatmap.shape
    return rows, cols, (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)


def z_at(surface: SearchSurface, offsets_rd_m: np.ndarray) -> np.ndarray:
    """z at each offset; 0 (no evidence) outside the heatmap or on NaN cells."""
    rows, cols, inside = cells_of(surface, offsets_rd_m)
    z = np.zeros(len(offsets_rd_m))
    z[inside] = surface.heatmap[rows[inside], cols[inside]]
    return np.nan_to_num(z, nan=0.0)


def is_flat(surface: SearchSurface) -> bool:
    """Nothing to read: every finite cell equal, or no finite cell at all."""
    finite = surface.heatmap[np.isfinite(surface.heatmap)]
    return bool(finite.size == 0 or finite.max() == finite.min())


def peak_at_edge(surface: SearchSurface) -> bool:
    """The best cell lies on the heatmap border: the true maximum may be outside the window."""
    if not np.isfinite(surface.heatmap).any():
        return False
    row, col = np.unravel_index(np.nanargmax(surface.heatmap), surface.heatmap.shape)
    return bool(row in (0, surface.heatmap.shape[0] - 1) or col in (0, surface.heatmap.shape[1] - 1))
