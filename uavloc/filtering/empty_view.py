"""Empty-view likelihood (032): a blind frame says the map under the camera is empty too.

A blind frame (no weighted-class pixels) is unlikely where the map has roads or buildings
under the footprint. `P(blind | map coverage)` is a logistic fit on frames with a known true
footprint; particles are weighted by it on the heatmap's own grid.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from uavloc.matching.correlator import PositionCorrelator, SearchSurface, weighted_coverage
from uavloc.matching.window import SearchWindow, patch_size_m
from uavloc.simulation.correctors import RenderedFrame

MIN_BLIND_PROB = 0.01        # one segmentation error must not zero a particle
UNREGULARISED_C = 1e6        # sklearn's inverse regularisation: effectively a plain maximum-likelihood fit


@dataclass(frozen=True)
class EmptyViewModel:
    """`P(blind | c) = max(sigmoid(intercept + slope * c), floor)`, with `c` the map's weighted coverage in [0, 1]."""

    intercept: float
    slope: float
    floor: float = MIN_BLIND_PROB

    def log_prob_blind(self, map_coverage: np.ndarray) -> np.ndarray:
        return np.log(np.maximum(expit(self.intercept + self.slope * map_coverage), self.floor))


def fit_empty_view(map_coverage: np.ndarray, blind: np.ndarray) -> EmptyViewModel:
    model = LogisticRegression(C=UNREGULARISED_C).fit(np.asarray(map_coverage, float)[:, None], np.asarray(blind, int))
    return EmptyViewModel(float(model.intercept_[0]), float(model.coef_[0, 0]))


def map_coverage_grid(drone: torch.Tensor, satellite: torch.Tensor, class_weights: tuple[float, ...]) -> np.ndarray:
    """Share of weighted-class satellite pixels under the drone footprint at every offset: the heatmap's grid.

    `drone` `(1, C, h, w)`, `satellite` `(1, C, H, W)` → `(H - h + 1, W - w + 1)`.
    """
    weighted = np.asarray(class_weights) > 0
    mask = weighted[satellite[0].float().argmax(0).cpu().numpy()].astype(np.float64)
    h, w = drone.shape[2:]
    integral = np.pad(mask.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    sums = integral[h:, w:] - integral[:-h, w:] - integral[h:, :-w] + integral[:-h, :-w]
    return sums / (h * w)


def coverage_surface(correlator: PositionCorrelator, frame: RenderedFrame, window: SearchWindow) -> SearchSurface:
    """The map-coverage grid over `window`, as a surface (`heatmap` = coverage), on `score_window`'s grid.

    Raises ValueError if the patch leaves the map raster.
    """
    try:
        drone, sat, empty, _ = correlator.prepare(correlator.drone_map(frame), window.center, window.yaw_deg,
                                                  patch_size_m(correlator, correlator.footprint_m(frame), window.half_m))
    finally:
        correlator.clear_cache()
    grid = map_coverage_grid(drone, sat, correlator.config.class_weights)
    return replace(empty, heatmap=grid, coverage=weighted_coverage(drone, correlator.config.class_weights))
