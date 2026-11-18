from __future__ import annotations

import numpy as np
import pytest

from onboard_fakes import ORIGIN, FakeConnection, attitude, global_position, local_position, stream, system_time
from uavloc.geo import east_north_m
from uavloc.onboard.mavlink_sources import (ArdupilotHeight, MavlinkFlightState, MavlinkRouter, SampleBuffer,
                                            SimTruth, STREAM_MESSAGE_IDS)


def test_buffer_interpolates_holds_and_forgets() -> None:
    buffer = SampleBuffer(span_s=1.0)
    for t in (0.0, 0.5, 1.0, 1.5):
        buffer.add(t, [t * 10.0])
    assert buffer.at(1.25)[0] == pytest.approx(12.5)
    assert buffer.at(9.0)[0] == pytest.approx(15.0)       # held past the newest
    assert buffer.at(0.2) is None                          # older than the span
    assert buffer.oldest_s == 0.5


def test_buffer_interpolates_yaw_across_north() -> None:
    buffer = SampleBuffer(angular=(0,))
    buffer.add(0.0, [350.0])
    buffer.add(1.0, [10.0])
    assert buffer.at(0.5)[0] == pytest.approx(0.0, abs=1e-6)


def test_buffer_integral_is_trapezoidal_and_needs_history() -> None:
    buffer = SampleBuffer()
    for t in (0.0, 1.0, 2.0):
        buffer.add(t, [t, 1.0])
    assert buffer.integral(0.0, 2.0) == pytest.approx([2.0, 2.0])
    assert buffer.integral(1.5, 3.0)[1] == pytest.approx(1.5)   # velocity held at the newest sample
    with pytest.raises(LookupError):
        buffer.integral(-1.0, 1.0)


def test_buffer_restarts_after_a_reboot() -> None:
    buffer = SampleBuffer()
    buffer.add(100.0, [1.0])
    buffer.add(1.0, [2.0])
    assert len(buffer) == 1 and buffer.at(1.0)[0] == 2.0


def test_flight_state_interpolates_and_converts_units() -> None:
    conn = FakeConnection()
    conn.queue.extend([attitude(10.0, yaw_rad=0.0), attitude(10.1, yaw_rad=0.2),
                       local_position(10.0, 3.0, 4.0), local_position(10.1, 3.0, 4.0),
                       global_position(10.0, ORIGIN, relative_alt_m=40.0),
                       global_position(10.1, ORIGIN, relative_alt_m=42.0)])
    source = MavlinkFlightState(conn)
    state = source.state_at(10.05)
    assert state.yaw_deg == pytest.approx(np.degrees(0.1))
    assert (state.velocity_north_mps, state.velocity_east_mps) == (3.0, 4.0)
    assert state.altitude_m == pytest.approx(41.0)
    assert state.position.lat == pytest.approx(ORIGIN.lat)
    assert source.latest().time_s == pytest.approx(10.1)
    assert source.state_at(9.0) is None
    assert ArdupilotHeight().height_m(state) == state.altitude_m


def test_flight_state_is_none_until_all_streams_arrive() -> None:
    conn = FakeConnection()
    conn.queue.append(attitude(1.0))
    source = MavlinkFlightState(conn)
    assert source.latest() is None


def test_streams_are_requested_at_start_up() -> None:
    conn = FakeConnection()
    MavlinkFlightState(conn)
    requested = {int(command[4]): command[5] for command in conn.mav.commands}
    assert requested[STREAM_MESSAGE_IDS["ATTITUDE"]] == pytest.approx(1e6 / 50)
    assert requested[STREAM_MESSAGE_IDS["LOCAL_POSITION_NED"]] == pytest.approx(1e6 / 20)
    assert requested[STREAM_MESSAGE_IDS["GLOBAL_POSITION_INT"]] == pytest.approx(1e6 / 20)


def test_velocity_displacement_at_constant_velocity() -> None:
    conn = FakeConnection()
    stream(conn, 0.0, 2.0, velocity=(3.0, 4.0))     # 3 m/s north, 4 m/s east
    source = MavlinkFlightState(conn)
    east_m, north_m = source.velocity_displacement_m(0.5, 1.0)
    assert (east_m, north_m) == pytest.approx((2.0, 1.5))
    with pytest.raises(ValueError):
        source.velocity_displacement_m(1.0, 0.5)
    with pytest.raises(LookupError):
        source.velocity_displacement_m(-3.0, 1.0)


def test_sources_share_one_connection_without_stealing_messages() -> None:
    conn = FakeConnection()
    stream(conn, 0.0, 1.0, velocity=(1.0, 0.0))
    flight_state, truth = MavlinkFlightState(conn), SimTruth(conn)
    assert flight_state.latest() is not None
    assert truth.position_at(0.5) is not None


def test_sim_truth_reads_simstate_and_interpolates() -> None:
    conn = FakeConnection()
    stream(conn, 0.0, 2.0, velocity=(0.0, 5.0), position=ORIGIN)   # 5 m/s east
    truth = SimTruth(conn)
    east_m, north_m = east_north_m(ORIGIN, truth.position_at(1.01))
    assert east_m == pytest.approx(5.05, abs=0.3)
    assert north_m == pytest.approx(0.0, abs=0.01)
    assert truth.position_at(-1.0) is None


def test_clocks_follow_the_flight_controller() -> None:
    conn = FakeConnection()
    now = {"t": 100.0}
    router = MavlinkRouter(conn, monotonic=lambda: now["t"], wall_clock=lambda: 7.0)
    source = MavlinkFlightState(router)
    assert router.unix_time_s() == 7.0                    # no SYSTEM_TIME yet: local clock
    conn.queue.extend([system_time(1_800_000_000.0, 50.0), attitude(50.0)])
    source.router.poll()
    now["t"] = 100.25
    assert source.fc_time_s() == pytest.approx(50.25)
    assert source.unix_time_s() == pytest.approx(1_800_000_000.25)


def test_fc_time_before_any_message_is_an_error() -> None:
    with pytest.raises(LookupError):
        MavlinkFlightState(FakeConnection()).fc_time_s()
