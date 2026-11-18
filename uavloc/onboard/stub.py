"""A stand-in localizer for SITL: the simulator's true position plus noise, and no fix where the map is blind."""

from __future__ import annotations

import numpy as np

from uavloc.geo import offset_latlon
from uavloc.onboard.contracts import Fix, FlightState, Frame, ReferenceMap, TruthSource

SITL_NOISE_M = 6.0                 # the test ladder's step 0a (docs/ardupilot-integration.md section 6)
REASON_FIX = "stub"
REASON_BLIND = "blind"
REASON_OUTSIDE = "outside-reference"
REASON_NO_TRUTH = "no-truth"
FIX_QUALITY = 1.0
NO_FIX_QUALITY = 0.0


class StubLocalizer:
    """`Localizer` that needs no image. `noise_m` is the std per axis; it is also the reported uncertainty.

    `latency_s` makes the fix describe the moment that long before the flight state, as a real localizer's would,
    so the sink's carry-forward has something to correct. `ignore_blind` answers over blind stretches too, to show
    what a flight does when the map never loses the fix.
    """

    def __init__(self, truth: TruthSource, noise_m: float = SITL_NOISE_M, seed: int = 0, latency_s: float = 0.0,
                 ignore_blind: bool = False) -> None:
        self.truth = truth
        self.noise_m = noise_m
        self.latency_s = latency_s
        self.ignore_blind = ignore_blind
        self._rng = np.random.default_rng(seed)

    def locate(self, frame: Frame | None, state: FlightState, reference: ReferenceMap) -> Fix:
        capture_s = frame.time_s if frame is not None else state.time_s - self.latency_s
        true_position = self.truth.position_at(capture_s)
        if true_position is None:
            return Fix(capture_s, None, self.noise_m, NO_FIX_QUALITY, REASON_NO_TRUTH)
        if not reference.contains(true_position):
            return Fix(capture_s, None, self.noise_m, NO_FIX_QUALITY, REASON_OUTSIDE)
        if reference.is_blind(true_position) and not self.ignore_blind:
            return Fix(capture_s, None, self.noise_m, NO_FIX_QUALITY, REASON_BLIND)
        east_m, north_m = self._rng.normal(0.0, self.noise_m, 2)
        return Fix(capture_s, offset_latlon(true_position, east_m, north_m), self.noise_m, FIX_QUALITY, REASON_FIX)
