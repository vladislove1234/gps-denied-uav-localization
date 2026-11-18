"""Stateful position estimators for a simulated flight (027).

Unlike a `Corrector`, which turns one view into a fix, an estimator keeps a belief:
`predict` moves it with every INS increment, `update` folds in one view, and
`estimate` reads the position and covariance out. The single-frame correlator is
the `ReplaceEstimator`: an applied fix overwrites the estimate.
"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from uavloc.geo import LatLon, offset_latlon
from uavloc.matching.correlator import PositionCorrelator
from uavloc.matching.window import SearchWindow
from uavloc.simulation.correctors import APPLIED_DECISIONS, MS_PER_S, RenderedFrame

UNKNOWN_COV_M2 = np.full((2, 2), np.nan)


@dataclass(frozen=True)
class FixRecord:
    """One update. `decision`: update | blind | out-of-raster | none, or the correlator's accept | across | reject."""

    decision: str
    elapsed_ms: float
    window: SearchWindow | None = None
    extent_m: tuple[float, float, float, float] | None = None   # heatmap (left, right, bottom, top) around window.center
    peak_z: float = math.nan
    coverage: float = math.nan
    peak_at_edge: bool = False
    effective_size: float = math.nan     # particles: effective sample size after the update
    live_hypotheses: int = 0             # hypotheses: how many survive the update
    dominance: float = math.nan          # hypotheses: weight share of the heaviest


class Estimator(Protocol):
    name: str

    def start(self, position: LatLon, sigma0_m: float) -> None: ...

    def predict(self, move_en_m: tuple[float, float], dt_s: float) -> None: ...

    def update(self, frame: RenderedFrame) -> FixRecord: ...

    def estimate(self) -> tuple[LatLon, np.ndarray]: ...


class SimulatedFrames:
    """Wraps a rendered view as the `DroneFrame` a correlator expects; `position` is the prior, not the truth."""

    def __init__(self) -> None:
        self._ids = itertools.count()

    def frame(self, view: Image.Image, prior: LatLon, yaw_deg: float, altitude_m: float) -> RenderedFrame:
        return RenderedFrame(f"sim-{next(self._ids)}", Path(), prior, altitude_m, yaw_deg, math.nan, (), view)


class InsOnlyEstimator:
    """No vision: the estimate is the INS track."""

    name = "ins-only"

    def start(self, position: LatLon, sigma0_m: float) -> None:
        self.position = position

    def predict(self, move_en_m: tuple[float, float], dt_s: float) -> None:
        self.position = offset_latlon(self.position, *move_en_m)

    def update(self, frame: RenderedFrame) -> FixRecord:
        return FixRecord("none", 0.0)

    def estimate(self) -> tuple[LatLon, np.ndarray]:
        return self.position, UNKNOWN_COV_M2


class ReplaceEstimator(InsOnlyEstimator):
    """Today's system: `PositionCorrelator.correct` around the estimate; an applied fix replaces it."""

    def __init__(self, correlator: PositionCorrelator, name: str = "replace") -> None:
        self.correlator = correlator
        self.name = name

    def update(self, frame: RenderedFrame) -> FixRecord:
        start = time.perf_counter()
        try:
            correction = self.correlator.correct(frame, self.position)
        except ValueError:   # the search patch leaves the map raster
            self.correlator.clear_cache()
            return FixRecord("out-of-raster", (time.perf_counter() - start) * MS_PER_S)
        elapsed_ms = (time.perf_counter() - start) * MS_PER_S
        self.correlator.clear_cache()
        window = SearchWindow(self.position, frame.yaw_deg, (math.nan, math.nan))
        if correction.decision in APPLIED_DECISIONS:
            self.position = correction.position
        heatmap = correction.heatmap
        edge = False
        if np.isfinite(heatmap).any():   # an all-NaN heatmap (bug-121 conv2d NaN) still moves, as today's system does
            row, col = np.unravel_index(np.nanargmax(heatmap), heatmap.shape)
            edge = bool(row in (0, heatmap.shape[0] - 1) or col in (0, heatmap.shape[1] - 1))
        return FixRecord(correction.decision, elapsed_ms, window, correction.heatmap_extent_m, correction.peak_z,
                         correction.coverage, edge)
