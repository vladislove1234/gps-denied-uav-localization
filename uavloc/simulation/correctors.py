"""Position correctors a simulated flight can call. Each turns one drone view and a prior into a fix.

A new corrector (particle filter, Kalman fusion, another matcher) only has to implement
`Corrector`; the flight loop, plots and video take it as is.
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

from uavloc.data.drone_frames import DroneFrame
from uavloc.features.segmentation import SegmentationFeatures
from uavloc.geo import LatLon
from uavloc.matching.correlator import PositionCorrelator

APPLIED_DECISIONS = ("accept", "across", "update")   # "update": a filter estimator used the fix
MS_PER_S = 1000.0


@dataclass(frozen=True)
class Fix:
    """One correction attempt. `decision`: accept | across | reject | out-of-raster | none (no corrector)."""

    position: LatLon
    decision: str
    elapsed_ms: float
    peak_z: float = math.nan
    uncertainty_m: tuple[float, float] | None = None   # posterior std along its long and short axis
    class_map: np.ndarray | None = None                # (H, W) class ids of the drone view, heading-up
    heatmap: np.ndarray | None = None                  # the search surface, heading-up

    @property
    def applied(self) -> bool:
        return self.decision in APPLIED_DECISIONS


class Corrector(Protocol):
    name: str

    def correct(self, view: Image.Image, prior: LatLon, yaw_deg: float, altitude_m: float) -> Fix: ...


class InsOnly:
    """No vision: the estimate is the INS track."""

    name = "ins-only"

    def correct(self, view: Image.Image, prior: LatLon, yaw_deg: float, altitude_m: float) -> Fix:
        return Fix(prior, "none", 0.0)


@dataclass(frozen=True)
class RenderedFrame(DroneFrame):
    """A drone frame held in memory rather than on disk. `position` is the prior: the truth is unknown onboard."""

    rendered: Image.Image | None = None

    def image(self) -> Image.Image:
        return self.rendered


class CorrelatorCorrector:
    """The single-frame `PositionCorrelator` (any features, window, prior and gate) as a corrector."""

    def __init__(self, correlator: PositionCorrelator, name: str = "correlator") -> None:
        self.correlator = correlator
        self.name = name
        self._frame_ids = itertools.count()

    def correct(self, view: Image.Image, prior: LatLon, yaw_deg: float, altitude_m: float) -> Fix:
        frame = RenderedFrame(f"sim-{next(self._frame_ids)}", Path(), prior, altitude_m, yaw_deg, math.nan, (), view)
        start = time.perf_counter()
        try:
            correction = self.correlator.correct(frame, prior)
        except ValueError:   # the search patch leaves the map raster
            self.correlator.clear_cache()
            return Fix(prior, "out-of-raster", (time.perf_counter() - start) * MS_PER_S)
        elapsed_ms = (time.perf_counter() - start) * MS_PER_S
        class_map = self._class_map(frame)
        self.correlator.clear_cache()
        return Fix(correction.position, correction.decision, elapsed_ms, correction.peak_z,
                   correction.uncertainty_m, class_map, correction.heatmap)

    def _class_map(self, frame: RenderedFrame) -> np.ndarray | None:
        """Argmax class ids of the cached drone map; None for features that are not class maps."""
        features = self.correlator.features
        if not isinstance(features, SegmentationFeatures):
            return None
        channel_argmax = self.correlator.drone_map(frame)[0].argmax(0).cpu().numpy()
        return np.asarray(features.channels)[channel_argmax]
