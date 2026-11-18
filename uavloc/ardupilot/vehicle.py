"""Commands to a connected vehicle: modes, arming, message rates, EKF source set, RC override."""

from __future__ import annotations

import time

from pymavlink import mavutil

from uavloc.ardupilot.link import wait_message

MODE_TIMEOUT_S = 10.0
COMMAND_ACK_TIMEOUT_S = 3.0
US_PER_S = 1_000_000
MAV_CMD_SET_EKF_SOURCE_SET = 42007
RC_RELEASE = 65535                   # RC_CHANNELS_OVERRIDE: leave this channel as it is
RC_CHANNEL_COUNT = 18
PWM_LOW = 1000
PWM_HIGH = 2000


def current_mode(conn: mavutil.mavfile, timeout_s: float = MODE_TIMEOUT_S) -> str:
    return mavutil.mode_string_v10(wait_message(conn, "HEARTBEAT", timeout_s))


def set_mode(conn: mavutil.mavfile, name: str, timeout_s: float = MODE_TIMEOUT_S) -> None:
    deadline = time.monotonic() + timeout_s
    conn.set_mode(conn.mode_mapping()[name])
    while time.monotonic() < deadline:
        heartbeat = conn.recv_match(type="HEARTBEAT", blocking=True, timeout=1.0)
        if heartbeat is not None and mavutil.mode_string_v10(heartbeat) == name:
            return
    raise TimeoutError(f"vehicle did not enter {name} within {timeout_s:g} s")


def is_armed(heartbeat) -> bool:
    return bool(heartbeat.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)


def request_message_rate(conn: mavutil.mavfile, message_id: int, rate_hz: float) -> None:
    conn.mav.command_long_send(conn.target_system, conn.target_component,
                               mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                               message_id, US_PER_S / rate_hz, 0, 0, 0, 0, 0)


def set_ekf_source_set(conn: mavutil.mavfile, source_set: int) -> None:
    """`source_set` is 1-based, as EK3_SRC1_* is set 1."""
    conn.mav.command_long_send(conn.target_system, conn.target_component,
                               MAV_CMD_SET_EKF_SOURCE_SET, 0, source_set, 0, 0, 0, 0, 0, 0)
    ack = wait_message(conn, "COMMAND_ACK", COMMAND_ACK_TIMEOUT_S)
    if ack.result != mavutil.mavlink.MAV_RESULT_ACCEPTED:
        raise RuntimeError(f"EKF source set {source_set} refused, MAV_RESULT {ack.result}")


def override_rc(conn: mavutil.mavfile, channels_pwm: dict[int, int]) -> None:
    """Hold the given RC channels (1-based) at a PWM and leave the others alone. Resend within 3 s to keep it."""
    channels = [RC_RELEASE] * RC_CHANNEL_COUNT
    for channel, pwm in channels_pwm.items():
        channels[channel - 1] = pwm
    conn.mav.rc_channels_override_send(conn.target_system, conn.target_component, *channels)
