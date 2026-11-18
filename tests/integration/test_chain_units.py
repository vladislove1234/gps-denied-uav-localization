"""The parts of the SITL chain that need no simulator: AMSL altitude on GPS_INPUT, blind switch, run analysis."""

from __future__ import annotations

import time

import pytest

from chain_support import synthetic_result  # noqa: F401  (also puts tests/onboard on sys.path)
from onboard_fakes import ORIGIN, ROUTE_NORTH_M, FakeConnection, at, message
from uavloc.ardupilot.chain import FLY_MODE
from uavloc.ardupilot.chain_analysis import FixRecord, blind_runs
from uavloc.ardupilot.flight import EKF_FLAG_POS_HORIZ_ABS, TrackPoint
from uavloc.geo import LatLon
from uavloc.onboard.cli import build_parser, scaled_clocks
from uavloc.onboard.contracts import Fix, FlightState, ReferenceMap
from uavloc.onboard.mavlink_sources import MavlinkFlightState
from uavloc.onboard.sinks import GpsInputSink
from uavloc.onboard.stub import REASON_FIX, StubLocalizer

ALTITUDE_AMSL_M = 340.0
BLIND_EAST_M = 550.0


class FixedTruth:
    def __init__(self, position: LatLon) -> None:
        self.position = position

    def position_at(self, time_s: float) -> LatLon:
        return self.position


class FixedAltitude:
    def altitude_amsl_m(self) -> float:
        return ALTITUDE_AMSL_M


class MissingAltitude:
    def altitude_amsl_m(self) -> float:
        raise LookupError("no sample")


def global_position_with_altitude(time_s: float, amsl_m: float):
    return message("GLOBAL_POSITION_INT", time_boot_ms=round(time_s * 1000), lat=0, lon=0, alt=round(amsl_m * 1000),
                   relative_alt=40_000)


def test_sink_passes_the_amsl_altitude_to_the_sender() -> None:
    sent = []
    sink = GpsInputSink(FakeConnection(), altitude=FixedAltitude(), sender=lambda *args, **kwargs: sent.append(kwargs))
    sink.send(Fix(10.0, ORIGIN, 6.0, 1.0, REASON_FIX))
    assert sent == [{"altitude_amsl_m": ALTITUDE_AMSL_M}]


def test_sink_without_an_altitude_source_keeps_the_old_sender_call() -> None:
    sent = []
    GpsInputSink(FakeConnection(), sender=lambda *args, **kwargs: sent.append((args, kwargs))).send(
        Fix(10.0, ORIGIN, 6.0, 1.0, REASON_FIX))
    assert sent[0][1] == {}


def test_sink_drops_a_fix_when_the_altitude_is_not_known_yet() -> None:
    sink = GpsInputSink(FakeConnection(), altitude=MissingAltitude(), sender=lambda *args, **kwargs: None)
    sink.send(Fix(10.0, ORIGIN, 6.0, 1.0, REASON_FIX))
    assert sink.sent == 0 and sink.skipped_stale == 1


def test_flight_state_reports_the_latest_amsl_altitude() -> None:
    conn = FakeConnection()
    state = MavlinkFlightState(conn, request_streams=False)
    with pytest.raises(LookupError):
        state.altitude_amsl_m()
    conn.queue.extend([global_position_with_altitude(1.0, 300.0), global_position_with_altitude(1.5, ALTITUDE_AMSL_M)])
    assert state.altitude_amsl_m() == pytest.approx(ALTITUDE_AMSL_M)


def test_stub_can_answer_over_blind_stretches() -> None:
    result = synthetic_result()
    reference = ReferenceMap(result.class_map, result.blind_map)
    state = FlightState(10.0, ORIGIN, 40.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    truth = FixedTruth(at(BLIND_EAST_M, ROUTE_NORTH_M))
    assert StubLocalizer(truth).locate(None, state, reference).position is None
    assert StubLocalizer(truth, ignore_blind=True).locate(None, state, reference).position is not None


def test_cli_clocks_run_in_simulated_seconds() -> None:
    clock, sleep = scaled_clocks(8.0)
    assert clock() / 8.0 == pytest.approx(time.monotonic(), abs=0.5)
    assert build_parser().parse_args(["--package", "x", "--connection", "y", "--stub"]).speedup == 1.0


def track_point(time_s: float, mode: str = FLY_MODE) -> TrackPoint:
    return TrackPoint(time_s, LatLon(49.9, 23.9), LatLon(49.9, 23.9), 40.0, mode, True, EKF_FLAG_POS_HORIZ_ABS)


def test_blind_runs_span_from_the_last_sent_fix_to_the_next() -> None:
    track = [track_point(t / 2) for t in range(0, 61)]
    fixes = [FixRecord(t / 2, not 10 <= t <= 19, "blind" if 10 <= t <= 19 else "stub") for t in range(0, 61)]
    runs = blind_runs(fixes, track)
    assert len(runs) == 1
    assert (runs[0].last_fix_s, runs[0].next_fix_s, runs[0].dropped_fixes) == (4.5, 10.0, 10)


def test_blind_runs_after_a_failsafe_are_ignored() -> None:
    track = [track_point(t / 2, FLY_MODE if t < 10 else "LAND") for t in range(0, 61)]
    fixes = [FixRecord(t / 2, not 20 <= t <= 29, "blind" if 20 <= t <= 29 else "stub") for t in range(0, 61)]
    assert blind_runs(fixes, track) == []
