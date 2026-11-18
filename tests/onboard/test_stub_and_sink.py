from __future__ import annotations

import numpy as np
import pytest

from onboard_fakes import ORIGIN, ROUTE_NORTH_M, FakeConnection, at, synthetic_result, stream
from uavloc.geo import LatLon, east_north_m, haversine_m
from uavloc.onboard.contracts import Fix, FlightState, ReferenceMap, bilinear_at
from uavloc.onboard.mavlink_sources import MavlinkFlightState
from uavloc.onboard.sinks import ACCURACY_FLOOR_M, GpsInputSink
from uavloc.onboard.stub import REASON_BLIND, REASON_FIX, REASON_OUTSIDE, StubLocalizer


class FixedTruth:
    def __init__(self, position: LatLon | None) -> None:
        self.position = position
        self.asked: list[float] = []

    def position_at(self, time_s: float) -> LatLon | None:
        self.asked.append(time_s)
        return self.position


def state_at_time(time_s: float = 10.0) -> FlightState:
    return FlightState(time_s, ORIGIN, 40.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@pytest.fixture
def reference() -> ReferenceMap:
    result = synthetic_result()
    return ReferenceMap(result.class_map, result.blind_map)


def test_stub_is_blind_where_the_map_is(reference: ReferenceMap) -> None:
    fix = StubLocalizer(FixedTruth(at(550.0, ROUTE_NORTH_M))).locate(None, state_at_time(), reference)
    assert fix.position is None and fix.reason == REASON_BLIND


def test_stub_has_no_reference_outside_the_raster(reference: ReferenceMap) -> None:
    fix = StubLocalizer(FixedTruth(at(-5000.0, 0.0))).locate(None, state_at_time(), reference)
    assert fix.position is None and fix.reason == REASON_OUTSIDE


def test_stub_noise_matches_the_setting_and_is_seeded(reference: ReferenceMap) -> None:
    truth_point = at(300.0, ROUTE_NORTH_M)

    def errors(seed: int) -> np.ndarray:
        stub = StubLocalizer(FixedTruth(truth_point), noise_m=6.0, seed=seed)
        fixes = [stub.locate(None, state_at_time(), reference) for _ in range(2000)]
        assert all(f.reason == REASON_FIX and f.uncertainty_m == 6.0 for f in fixes)
        return np.array([east_north_m(truth_point, f.position) for f in fixes])

    first = errors(1)
    assert first.std(axis=0) == pytest.approx([6.0, 6.0], rel=0.1)
    assert abs(first.mean()) < 0.5
    assert np.array_equal(first, errors(1)) and not np.array_equal(first, errors(2))


def test_stub_describes_the_moment_before_the_state(reference: ReferenceMap) -> None:
    truth = FixedTruth(at(300.0, ROUTE_NORTH_M))
    fix = StubLocalizer(truth, latency_s=0.5).locate(None, state_at_time(10.0), reference)
    assert truth.asked == [9.5] and fix.time_s == 9.5


class FakeMotion:
    def __init__(self, east_m: float = 0.0, north_m: float = 0.0, stale: bool = False) -> None:
        self.displacement, self.stale = (east_m, north_m), stale

    def fc_time_s(self) -> float:
        return 10.5

    def unix_time_s(self) -> float:
        return 1_800_000_000.5

    def velocity_displacement_m(self, from_s: float, to_s: float) -> tuple[float, float]:
        if self.stale:
            raise LookupError("too old")
        assert (from_s, to_s) == (10.0, 10.5)
        return self.displacement


class RecordingSender:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __call__(self, conn, position, accuracy_m, unix_s, satellites) -> None:
        self.calls.append((position, accuracy_m, unix_s, satellites))


def good_fix(uncertainty_m: float = 6.0) -> Fix:
    return Fix(10.0, at(300.0, ROUTE_NORTH_M), uncertainty_m, 1.0, "stub")


def test_sink_sends_nothing_for_no_fix_and_never_resends() -> None:
    sender = RecordingSender()
    sink = GpsInputSink(None, FakeMotion(), sender=sender)
    sink.send(good_fix())
    sink.send(Fix(10.5, None, 6.0, 0.0, "blind"))
    sink.send(Fix(11.0, None, 6.0, 0.0, "blind"))
    assert len(sender.calls) == 1
    assert (sink.sent, sink.skipped_no_fix) == (1, 2)


def test_sink_carries_the_fix_forward_by_velocity() -> None:
    sender = RecordingSender()
    GpsInputSink(None, FakeMotion(east_m=2.5, north_m=-1.0), sender=sender).send(good_fix())
    position, _, unix_s, _ = sender.calls[0]
    assert east_north_m(good_fix().position, position) == pytest.approx((2.5, -1.0), abs=1e-6)
    assert unix_s == 1_800_000_000.5


def test_sink_floors_the_accuracy() -> None:
    sender = RecordingSender()
    sink = GpsInputSink(None, FakeMotion(), sender=sender)
    sink.send(good_fix(uncertainty_m=0.5))
    sink.send(good_fix(uncertainty_m=12.0))
    assert [call[1] for call in sender.calls] == [ACCURACY_FLOOR_M, 12.0]


def test_sink_drops_a_fix_older_than_its_velocity_history() -> None:
    sender = RecordingSender()
    sink = GpsInputSink(None, FakeMotion(stale=True), sender=sender)
    sink.send(good_fix())
    assert sender.calls == [] and sink.skipped_stale == 1


def test_sink_sends_a_real_gps_input_through_the_connection() -> None:
    conn = FakeConnection()
    stream(conn, 0.0, 2.0, velocity=(0.0, 10.0))          # 10 m/s east
    motion = MavlinkFlightState(conn)
    capture_s = motion.fc_time_s() - 0.5
    fix = Fix(capture_s, ORIGIN, 6.0, 1.0, "stub")
    GpsInputSink(conn, motion).send(fix)
    sent = conn.mav.gps_inputs[0]
    sent_lon = sent[7] / 1e7
    assert haversine_m(ORIGIN, LatLon(sent[6] / 1e7, sent_lon)) == pytest.approx(5.0, abs=0.2)   # 0.5 s x 10 m/s
    assert sent[1] == 1 and sent[15] == 6.0               # gps_id, horizontal accuracy


def test_bilinear_interpolates_between_cell_centres_and_holds_edges() -> None:
    grid = np.zeros((5, 5))
    grid[2, 2] = 1.0
    assert bilinear_at(grid, 2.0, 2.0) == 1.0
    assert bilinear_at(grid, 2.5, 2.0) == 0.5
    assert bilinear_at(grid, -3.0, 9.0) == 0.0
