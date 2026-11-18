"""The onboard loop: read state, locate, send, log, once per period."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from typing import Callable

from uavloc.onboard.contracts import (CameraSource, Fix, FlightLogger, FlightStateSource, HeightSource, Localizer,
                                      PositionSink)
from uavloc.onboard.package import MissionPackage

DEFAULT_PERIOD_S = 0.5
NO_POSITION_UNCERTAINTY_M = 0.0     # meaningless without a position
NO_FIX_QUALITY = 0.0
REASON_NO_STATE = "no-state"

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunnerParts:
    camera: CameraSource
    state_source: FlightStateSource
    height: HeightSource
    localizer: Localizer
    sink: PositionSink
    logger: FlightLogger


class OnboardRunner:
    """One `step()` per `period_s`. A localizer that raises costs that step a fix, not the flight."""

    def __init__(self, package: MissionPackage, parts: RunnerParts, period_s: float = DEFAULT_PERIOD_S,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep) -> None:
        if period_s <= 0:
            raise ValueError(f"period_s must be positive, got {period_s}")
        self.package = package
        self.parts = parts
        self.period_s = period_s
        self._clock = clock
        self._sleep = sleep
        self.steps = 0
        self.steps_without_state = 0

    def step(self) -> Fix | None:
        """One pass; returns the fix (possibly a no-fix), or None when there is no flight state yet."""
        parts = self.parts
        frame = parts.camera.next_frame()
        state = parts.state_source.state_at(frame.time_s) if frame is not None else parts.state_source.latest()
        self.steps += 1
        if state is None:
            self.steps_without_state += 1
            return None
        state = replace(state, altitude_m=parts.height.height_m(state))
        try:
            fix = parts.localizer.locate(frame, state, self.package.reference)
        except Exception as error:
            log.exception("localizer failed")
            fix = Fix(state.time_s, None, NO_POSITION_UNCERTAINTY_M, NO_FIX_QUALITY, f"error: {type(error).__name__}")
        parts.sink.send(fix)
        parts.logger.log_state(state)
        parts.logger.log_fix(fix)
        if frame is not None:
            parts.logger.log_frame(frame)
        return fix

    def run(self, max_steps: int | None = None, should_stop: Callable[[], bool] = lambda: False) -> int:
        """Loop until `max_steps`, or until `should_stop()`; always closes the logger. Returns the steps taken."""
        deadline = self._clock()
        try:
            while not should_stop() and (max_steps is None or self.steps < max_steps):
                self.step()
                deadline += self.period_s
                self._sleep(max(deadline - self._clock(), 0.0))
        finally:
            self.parts.logger.close()
        return self.steps
