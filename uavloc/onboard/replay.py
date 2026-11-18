"""Fly a planned route in simulated time through the real runner, to see where the stub would lose its fix."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from uavloc.geo import LatLon, bearing_deg, east_north_m, haversine_m, offset_latlon
from uavloc.onboard.cameras import NullCamera
from uavloc.onboard.contracts import Fix, FlightState
from uavloc.onboard.mavlink_sources import ArdupilotHeight
from uavloc.onboard.package import MissionPackage
from uavloc.onboard.runner import OnboardRunner, RunnerParts
from uavloc.onboard.stub import StubLocalizer


class PolylineFlight:
    """A vehicle flying a polyline at constant speed on a simulated clock.

    It is the flight state source, the simulator truth and the clock of the runner at once. It holds the last
    point once the route is done.
    """

    def __init__(self, waypoints: tuple[LatLon, ...], speed_mps: float, altitude_m: float) -> None:
        self.waypoints = waypoints
        self.speed_mps = speed_mps
        self.altitude_m = altitude_m
        self.time_s = 0.0
        legs = [east_north_m(a, b) for a, b in zip(waypoints, waypoints[1:])]
        self._leg_m = np.array([math.hypot(*leg) for leg in legs])
        self._ends_m = np.cumsum(self._leg_m)
        self._bearing_deg = [bearing_deg(*leg) for leg in legs]
        self.length_m = float(self._ends_m[-1])

    @property
    def duration_s(self) -> float:
        return self.length_m / self.speed_mps

    def advance(self, seconds: float) -> None:
        self.time_s += seconds

    def along_m(self, time_s: float) -> float:
        return min(max(time_s, 0.0) * self.speed_mps, self.length_m)

    def _leg_at(self, along_m: float) -> int:
        return min(int(np.searchsorted(self._ends_m, along_m, side="right")), len(self._leg_m) - 1)

    def position_at(self, time_s: float) -> LatLon:
        along_m = self.along_m(time_s)
        leg = self._leg_at(along_m)
        start_m = self._ends_m[leg] - self._leg_m[leg]
        east_m, north_m = east_north_m(self.waypoints[leg], self.waypoints[leg + 1])
        fraction = (along_m - start_m) / self._leg_m[leg]
        return offset_latlon(self.waypoints[leg], east_m * fraction, north_m * fraction)

    def state_at(self, time_s: float) -> FlightState:
        heading = self._bearing_deg[self._leg_at(self.along_m(time_s))]
        flying = self.along_m(time_s) < self.length_m
        speed = self.speed_mps if flying else 0.0
        return FlightState(time_s, self.position_at(time_s), self.altitude_m, heading, 0.0, 0.0,
                           speed * math.cos(math.radians(heading)), speed * math.sin(math.radians(heading)))

    def latest(self) -> FlightState:
        return self.state_at(self.time_s)


class FixRecorder:
    """Sink and logger that only remember: every fix the localizer made, and those that would have been sent."""

    def __init__(self) -> None:
        self.fixes: list[Fix] = []
        self.sent: list[Fix] = []

    def send(self, fix: Fix) -> None:
        if fix.position is not None:
            self.sent.append(fix)

    def log_fix(self, fix: Fix) -> None:
        self.fixes.append(fix)

    def log_state(self, state: FlightState) -> None:
        pass

    def log_frame(self, frame: object) -> None:
        pass

    def close(self) -> None:
        pass


@dataclass(frozen=True)
class ReplayResult:
    times_s: np.ndarray                 # `(N,)` capture time of each step
    along_m: np.ndarray                 # `(N,)` distance flown, out and back
    positions: tuple[LatLon, ...]       # true position at each step
    fixes: tuple[Fix, ...]
    length_m: float                     # out and back

    @property
    def dropped(self) -> np.ndarray:
        return np.array([fix.position is None for fix in self.fixes])

    @property
    def dropped_share(self) -> float:
        return float(self.dropped.mean())

    def reasons(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for fix in self.fixes:
            counts[fix.reason] = counts.get(fix.reason, 0) + 1
        return counts

    def longest_dropped_run_m(self) -> float:
        """Longest distance flown with consecutive dropped fixes."""
        longest, start = 0.0, None
        for along, dropped in zip(self.along_m, self.dropped):
            if dropped and start is None:
                start = along
            if not dropped and start is not None:
                longest, start = max(longest, along - start), None
        if start is not None:
            longest = max(longest, self.along_m[-1] - start)
        return float(longest)

    def error_m(self) -> np.ndarray:
        """Horizontal error of each sent fix against the truth at its capture time."""
        return np.array([haversine_m(fix.position, truth) for fix, truth in zip(self.fixes, self.positions)
                         if fix.position is not None])


def replay_stub(package: MissionPackage, speed_mps: float, altitude_m: float, period_s: float, noise_m: float,
                seed: int) -> ReplayResult:
    """Fly the package's out-and-back route at `speed_mps`, one runner step per `period_s` of simulated time."""
    flight = PolylineFlight(package.waypoints, speed_mps, altitude_m)
    recorder = FixRecorder()
    parts = RunnerParts(NullCamera(), flight, ArdupilotHeight(), StubLocalizer(flight, noise_m, seed), recorder, recorder)
    runner = OnboardRunner(package, parts, period_s, clock=lambda: flight.time_s, sleep=flight.advance)
    runner.run(max_steps=math.ceil(flight.duration_s / period_s) + 1)
    times = np.array([fix.time_s for fix in recorder.fixes])
    return ReplayResult(times, np.array([flight.along_m(t) for t in times]),
                        tuple(flight.position_at(t) for t in times), tuple(recorder.fixes), flight.length_m)

