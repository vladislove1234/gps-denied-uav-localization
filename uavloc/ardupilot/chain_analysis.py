"""Read a chain flight back: what the runner sent and dropped, and what the vehicle did about it."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from uavloc.ardupilot.chain import FLY_MODE, ChainFlight
from uavloc.ardupilot.flight import EKF_FLAG_POS_HORIZ_ABS, TrackPoint
from uavloc.geo import LatLon, haversine_m

FIXES_NAME = "fixes.jsonl"           # `uavloc.onboard.logger.FIXES_NAME`, repeated so this module stays light
REASON_SENT = "stub"                 # `uavloc.onboard.stub.REASON_FIX`
REASON_BLIND = "blind"
ERROR_PERCENTILE = 95.0
RECOVERY_WINDOW_S = 10.0             # watched after fixes come back, for a failsafe that trails the stretch


@dataclass(frozen=True)
class FixRecord:
    time_s: float                    # capture time on the flight-controller clock (simulated seconds since boot)
    sent: bool
    reason: str


@dataclass(frozen=True)
class BlindRun:
    """One stretch without fixes: from the last fix sent before it to the first sent after it."""

    last_fix_s: float
    next_fix_s: float | None         # None when the flight ended first
    dropped_fixes: int
    length_m: float                  # distance flown between the two fixes
    error_at_last_fix_m: float
    error_peak_m: float
    error_at_next_fix_m: float | None
    position_lost_s: float | None    # after the last fix, until the EKF dropped its absolute position
    failsafe_s: float | None         # after the last fix, until the mode left AUTO
    mode_after: str

    @property
    def gap_s(self) -> float | None:
        return None if self.next_fix_s is None else self.next_fix_s - self.last_fix_s


def find_fix_log(log_dir: Path) -> Path:
    logs = sorted(log_dir.glob(f"run-*/{FIXES_NAME}"))
    if len(logs) != 1:
        raise FileNotFoundError(f"expected one runner log under {log_dir}, found {len(logs)}")
    return logs[0]


def read_fix_log(path: Path) -> list[FixRecord]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    return [FixRecord(r["time_s"], r["reason"] == REASON_SENT, r["reason"]) for r in records]


def ekf_error_m(point: TrackPoint) -> float:
    return haversine_m(point.truth, point.estimate)


def errors_since(track: list[TrackPoint], since_s: float) -> np.ndarray:
    return np.array([ekf_error_m(p) for p in track if p.sim_time_s >= since_s])


def flight_end_s(track: list[TrackPoint]) -> float:
    return track[-1].sim_time_s


def blind_runs(fixes: list[FixRecord], track: list[TrackPoint]) -> list[BlindRun]:
    """Every stretch of dropped fixes, with what the EKF did in it. Fixes after the flight ended are ignored."""
    times = np.array([p.sim_time_s for p in track])
    fixes = [f for f in fixes if times[0] <= f.time_s <= times[-1]]
    runs, last_sent_s, in_run = [], None, False
    dropped = 0
    for fix in fixes:
        if fix.sent:
            if in_run:
                runs.append(describe_run(track, times, last_sent_s, fix.time_s, dropped))
            last_sent_s, in_run, dropped = fix.time_s, False, 0
        elif fix.reason == REASON_BLIND and last_sent_s is not None:
            in_run, dropped = True, dropped + 1
    if in_run:
        runs.append(describe_run(track, times, last_sent_s, None, dropped))
    return [run for run in runs if mode_at(track, times, run.last_fix_s) == FLY_MODE]   # after a failsafe the mission is over


def mode_at(track: list[TrackPoint], times: np.ndarray, time_s: float) -> str:
    return track[min(int(np.searchsorted(times, time_s)), len(track) - 1)].mode


def describe_run(track: list[TrackPoint], times: np.ndarray, last_fix_s: float, next_fix_s: float | None,
                 dropped: int) -> BlindRun:
    start = int(np.searchsorted(times, last_fix_s))
    stop = len(track) if next_fix_s is None else int(np.searchsorted(times, next_fix_s))
    inside = track[start:stop]
    watched_until_s = (times[-1] if next_fix_s is None else next_fix_s) + RECOVERY_WINDOW_S
    watched = track[start:int(np.searchsorted(times, watched_until_s))]
    length_m = sum(haversine_m(a.truth, b.truth) for a, b in zip(inside, inside[1:]))
    lost = next((p.sim_time_s - last_fix_s for p in watched if not p.ekf_flags & EKF_FLAG_POS_HORIZ_ABS), None)
    failsafe = next((p.sim_time_s - last_fix_s for p in watched if p.mode != FLY_MODE), None)
    return BlindRun(last_fix_s, next_fix_s, dropped, length_m, ekf_error_m(track[start]),
                    max(ekf_error_m(p) for p in watched), None if stop >= len(track) else ekf_error_m(track[stop]),
                    lost, failsafe, watched[-1].mode)


def mode_changes(track: list[TrackPoint]) -> list[tuple[float, str]]:
    changes = [(track[0].sim_time_s, track[0].mode)]
    changes += [(b.sim_time_s, b.mode) for a, b in zip(track, track[1:]) if a.mode != b.mode]
    return changes


def positions_at(track: list[TrackPoint], times_s: list[float]) -> list[LatLon]:
    """Simulator truth at the given simulated times (nearest track point)."""
    track_times = np.array([p.sim_time_s for p in track])
    return [track[min(int(np.searchsorted(track_times, t)), len(track) - 1)].truth for t in times_s]


def error_stats(errors: np.ndarray) -> dict[str, float | int]:
    if not len(errors):
        return {"samples": 0}
    return {"samples": len(errors), "max_m": float(errors.max()), "mean_m": float(errors.mean()),
            "p95_m": float(np.percentile(errors, ERROR_PERCENTILE))}


def summarise_flight(flight: ChainFlight, fixes: list[FixRecord], start: LatLon, end: LatLon) -> dict:
    """Everything the notes quote about one flight, as plain numbers."""
    track = flight.track
    armed = [p for p in track if p.armed]
    active = [p for p in armed if p.sim_time_s >= flight.gps1_off_s]
    in_auto = [p for p in active if p.mode == FLY_MODE]
    reasons = Counter(f.reason for f in fixes if armed[0].sim_time_s <= f.time_s <= armed[-1].sim_time_s)
    return {
        "completed": flight.completed,
        "runner_exit_code": flight.runner_exit_code,
        "events": flight.events,
        "mode_changes": mode_changes(armed),
        "airborne_s": armed[-1].sim_time_s - armed[0].sim_time_s,
        "max_height_m": max(p.height_m for p in armed),
        "closest_to_end_m": min(haversine_m(p.truth, end) for p in armed),
        "landed_offset_from_start_m": haversine_m(flight.landed, start),
        "fixes_by_reason": dict(reasons),
        "fix_rate_hz": reasons[REASON_SENT] / (armed[-1].sim_time_s - armed[0].sim_time_s),
        "active_error_all_modes": error_stats(np.array([ekf_error_m(p) for p in active])),
        "active_error_auto_only": error_stats(np.array([ekf_error_m(p) for p in in_auto])),
        "failsafe_texts": [(t, text) for t, text in flight.texts if "ailsafe" in text or "position lost" in text],
        "blind_runs": [asdict(run) | {"gap_s": run.gap_s} for run in blind_runs(fixes, track)],
    }
