"""Search the heading and footprint scale around the logged pose, then translation as before (experiment 003).

The schedule is fixed by the 003 proposal: coarse heading offsets at scale 1, then every scale at the best
coarse heading, then fine offsets around that heading at the chosen scale. The winner is the hypothesis with
the highest raw score over everything evaluated. Its surface is z-scored and decided exactly like a
translation-only surface, so `peak_z`, the gate and the posterior keep their meaning.

The satellite patch is always `search_scale` x the logged footprint, turned to the hypothesised heading; only
the drone side is resampled to the scaled footprint. Correlation runs through the FFT on CPU
(`learned.xcorr_valid`, and `acontrario.raw_scores` for weighted NCC): ~100x faster than a direct conv with these
kernels, and free of the CPU `conv2d` NaN flake seen on this M1 once MPS is in use.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, replace

import numpy as np
import torch
import torch.nn.functional as F

from uavloc.data.drone_frames import DroneFrame
from uavloc.geo import LatLon
from uavloc.matching.acontrario import raw_scores
from uavloc.matching.correlator import FeatureExtractor, PositionCorrelator, SearchSurface, weighted_coverage, z_scored
from uavloc.matching.learned import xcorr_valid

LOGGED_POSE = (0.0, 1.0)   # (heading offset, scale): if its patch leaves the raster, the whole draw does
KEY_DECIMALS = 6           # offsets are summed (coarse + fine), so keys are rounded before comparing
OUT_OF_RASTER = "leaves the raster"   # `GeoImage.crop_rotated`'s ValueError; any other ValueError is a bug


@dataclass(frozen=True)
class Hypothesis:
    """One scored pose: its raw score surface (comparable across poses) and the geometry to decide on it."""

    heading_offset_deg: float
    scale: float
    raw: torch.Tensor            # (H', W') before z-scoring
    mask: np.ndarray | None      # round window, as in `PositionCorrelator.prepare`
    surface: SearchSurface       # geometry at this pose; heatmap not filled in yet
    coverage: float

    @property
    def peak(self) -> float:
        values = self.raw if self.mask is None else self.raw[torch.from_numpy(self.mask)]
        peak = float(values.max()) if values.numel() else float("nan")
        return peak if math.isfinite(peak) else -math.inf

    def rank(self) -> tuple[float, float, float]:
        """Highest raw peak wins; an exact tie goes to the pose nearer the logged one (keeps the prior when flat)."""
        return self.peak, -abs(self.heading_offset_deg), -abs(self.scale - 1.0)


def mean_cosine_raw(drone: torch.Tensor, satellite: torch.Tensor) -> torch.Tensor:
    """Sliding sum of per-cell unit-vector dot products over the number of drone cells: a smaller kernel does not
    lose by being smaller.

    In float64: a VGG11 surface spans ~0.1-2 on a sum of ~800 cells, so float32 rounding (~0.006) moves its argmax.
    """
    kernel = F.normalize(drone.cpu().double(), dim=1)
    total = xcorr_valid(F.normalize(satellite.cpu().double(), dim=1), kernel).sum(1)[0]
    return total / (kernel.shape[2] * kernel.shape[3])


class SatellitePatchCache:
    """Wraps a feature extractor for one draw: the satellite patch depends on the heading, not the scale, so each
    heading's patch (a VGG11 forward) is computed once. Speed only; the tensors are the same."""

    def __init__(self, inner: FeatureExtractor) -> None:
        self.inner = inner
        self._patches: dict[tuple, torch.Tensor] = {}

    def __getattr__(self, name: str):
        return getattr(self.inner, name)

    def drone(self, *args, **kwargs) -> torch.Tensor:
        return self.inner.drone(*args, **kwargs)

    def satellite_patch(self, center: LatLon, yaw_deg: float, width_m: float, height_m: float) -> torch.Tensor:
        key = (center, yaw_deg, width_m, height_m)
        if key not in self._patches:
            self._patches[key] = self.inner.satellite_patch(center, yaw_deg, width_m, height_m)
        return self._patches[key]


def score_pose(correlator: PositionCorrelator, frame: DroneFrame, prior: LatLon,
               heading_offset_deg: float, scale: float) -> Hypothesis:
    """Raises ValueError when the satellite patch at this heading leaves the raster."""
    cfg = correlator.config
    drone, sat, empty, mask = correlator.prepare(correlator.drone_map(frame, scale), prior,
                                                 frame.yaw_deg + heading_offset_deg, correlator.patch_size_m(frame))
    if cfg.matching == "weighted-ncc":
        # the weighted mean of per-class NCC before z-scoring, already scale-free
        raw = torch.from_numpy(raw_scores(drone, sat, cfg.class_weights)[0])
        coverage = weighted_coverage(drone, cfg.class_weights)
    else:
        raw, coverage = mean_cosine_raw(drone, sat), float("nan")
    surface = replace(empty, heading_offset_deg=heading_offset_deg, scale=scale)
    return Hypothesis(heading_offset_deg, scale, raw, mask, surface, coverage)


def best(hypotheses: list[Hypothesis]) -> Hypothesis:
    return max(hypotheses, key=Hypothesis.rank)


def evaluate_schedule(correlator: PositionCorrelator, frame: DroneFrame,
                      prior: LatLon) -> dict[tuple[float, float], Hypothesis | None]:
    """Every pose the schedule asks for, each scored once, in evaluation order.

    "greedy" (registered): coarse offsets at scale 1, every scale at the best coarse offset, fine offsets at the
    chosen scale. "joint" (2026-10-20 amendments): coarse offsets x every scale, then fine offsets around each of the
    `fine_pass_top_k` best coarse (offset, scale) pairs, at that pair's scale.

    A pose whose satellite patch leaves the raster maps to None; the logged pose leaving it raises ValueError,
    which the benchmark records as out-of-raster, as for translation-only.
    """
    cfg = correlator.config
    scored: dict[tuple[float, float], Hypothesis | None] = {}
    correlator = copy.copy(correlator)   # shares the drone-map cache; only the satellite side is wrapped
    correlator.features = SatellitePatchCache(correlator.features)

    def stage(poses: list[tuple[float, float]]) -> list[Hypothesis]:
        """Score the poses not scored yet; return all of them that stayed on the raster, best first."""
        for offset, scale in poses:
            key = (round(offset, KEY_DECIMALS), round(scale, KEY_DECIMALS))
            if key in scored:
                continue
            try:
                scored[key] = score_pose(correlator, frame, prior, key[0], key[1])
            except ValueError as error:
                if key == LOGGED_POSE or OUT_OF_RASTER not in str(error):
                    raise
                scored[key] = None
        keys = {(round(o, KEY_DECIMALS), round(s, KEY_DECIMALS)) for o, s in poses}
        return sorted((scored[k] for k in keys if scored[k] is not None), key=Hypothesis.rank, reverse=True)

    stage([LOGGED_POSE])
    if cfg.pose_schedule == "joint":
        coarse = stage([(offset, scale) for scale in cfg.scales for offset in cfg.heading_offsets_deg])
        for anchor in coarse[:cfg.fine_pass_top_k]:
            stage([(anchor.heading_offset_deg + fine, anchor.scale) for fine in cfg.heading_fine_offsets_deg])
        return scored
    coarse = stage([(offset, 1.0) for offset in cfg.heading_offsets_deg])[0]
    chosen = stage([(coarse.heading_offset_deg, scale) for scale in cfg.scales])[0]
    stage([(coarse.heading_offset_deg + fine, chosen.scale) for fine in cfg.heading_fine_offsets_deg])
    return scored


def search_pose(correlator: PositionCorrelator, frame: DroneFrame, prior: LatLon) -> SearchSurface:
    """The winning pose's z-scored surface; its `yaw_deg` is the logged heading plus the winning offset."""
    scored = evaluate_schedule(correlator, frame, prior)
    hypotheses = [h for h in scored.values() if h is not None]
    winner = best(hypotheses)
    return replace(winner.surface, heatmap=z_scored(winner.raw, winner.mask), coverage=winner.coverage,
                   hypotheses=len(hypotheses), skipped_hypotheses=len(scored) - len(hypotheses))
