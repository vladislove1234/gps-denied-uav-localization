"""MAVLink link to the vehicle: connect, mission protocol, home position, parameters."""

from __future__ import annotations

import math
import time
from collections.abc import Iterable

from pymavlink import mavutil

from uavloc.ardupilot.mission import MissionItem
from uavloc.geo import LatLon, haversine_m

HOME_WARNING_M = 20.0
E7 = 1e7
MISSION_TYPE_MISSION = 0
MESSAGE_RETRIES = 3
PARAM_TOLERANCE = 1e-4
PARAM_TIMEOUT_S = 3.0
MISSION_STEP_TIMEOUT_S = 5.0
MAV_MISSION_ACCEPTED = mavutil.mavlink.MAV_MISSION_ACCEPTED
HOME_POSITION_MESSAGE_ID = 242


def connect(url: str, timeout_s: float) -> mavutil.mavfile:
    conn = mavutil.mavlink_connection(url)
    if conn.wait_heartbeat(timeout=timeout_s) is None:
        raise TimeoutError(f"no MAVLink heartbeat from {url} within {timeout_s:g} s")
    return conn


def wait_message(conn: mavutil.mavfile, message_type: str | Iterable[str], timeout_s: float):
    """Block until a message of the type arrives; raises on timeout so callers never handle None."""
    message = conn.recv_match(type=message_type, blocking=True, timeout=timeout_s)
    if message is None:
        raise TimeoutError(f"no {message_type} within {timeout_s:g} s")
    return message


def to_item(message) -> MissionItem:
    return MissionItem(message.seq, message.frame, message.command, message.current, message.autocontinue,
                       message.param1, message.param2, message.param3, message.param4,
                       message.x / E7, message.y / E7, message.z)


def send_item(conn: mavutil.mavfile, item: MissionItem) -> None:
    conn.mav.mission_item_int_send(
        conn.target_system, conn.target_component, item.seq, item.frame, item.command, item.current,
        item.autocontinue, item.param1, item.param2, item.param3, item.param4,
        round(item.lat * E7), round(item.lon * E7), item.alt, MISSION_TYPE_MISSION)


def upload_mission(conn: mavutil.mavfile, items: list[MissionItem]) -> None:
    conn.mav.mission_count_send(conn.target_system, conn.target_component, len(items), MISSION_TYPE_MISSION)
    while True:
        message = wait_message(conn, ["MISSION_REQUEST_INT", "MISSION_REQUEST", "MISSION_ACK"],
                               MISSION_STEP_TIMEOUT_S)
        if message.get_type() == "MISSION_ACK":
            if message.type != MAV_MISSION_ACCEPTED:
                raise RuntimeError(f"mission upload rejected, MAV_MISSION_RESULT {message.type}")
            return
        send_item(conn, items[message.seq])


def download_mission(conn: mavutil.mavfile) -> list[MissionItem]:
    conn.mav.mission_request_list_send(conn.target_system, conn.target_component, MISSION_TYPE_MISSION)
    count = wait_message(conn, "MISSION_COUNT", MISSION_STEP_TIMEOUT_S).count
    items = []
    for seq in range(count):
        conn.mav.mission_request_int_send(conn.target_system, conn.target_component, seq, MISSION_TYPE_MISSION)
        items.append(to_item(wait_message(conn, "MISSION_ITEM_INT", MISSION_STEP_TIMEOUT_S)))
    conn.mav.mission_ack_send(conn.target_system, conn.target_component, MAV_MISSION_ACCEPTED, MISSION_TYPE_MISSION)
    return items


def vehicle_home(conn: mavutil.mavfile, timeout_s: float) -> LatLon:
    """Home as the vehicle reports it. It exists only once the EKF has an origin and a GPS fix."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        conn.mav.command_long_send(conn.target_system, conn.target_component,
                                   mavutil.mavlink.MAV_CMD_REQUEST_MESSAGE, 0,
                                   HOME_POSITION_MESSAGE_ID, 0, 0, 0, 0, 0, 0)
        message = conn.recv_match(type="HOME_POSITION", blocking=True, timeout=1.0)
        if message is not None:
            return LatLon(message.latitude / E7, message.longitude / E7)
    raise TimeoutError(f"vehicle reported no home position within {timeout_s:g} s")


def home_offset_m(home: LatLon, start: LatLon) -> float:
    return haversine_m(home, start)


def set_param(conn: mavutil.mavfile, name: str, value: float) -> None:
    for _ in range(MESSAGE_RETRIES):
        conn.mav.param_set_send(conn.target_system, conn.target_component, name.encode(), value,
                                mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
        deadline = time.monotonic() + PARAM_TIMEOUT_S
        while time.monotonic() < deadline:
            message = conn.recv_match(type="PARAM_VALUE", blocking=True, timeout=PARAM_TIMEOUT_S)
            if message is not None and message.param_id == name:
                if not math.isclose(message.param_value, value, abs_tol=PARAM_TOLERANCE):
                    raise RuntimeError(f"{name} reads back {message.param_value}, expected {value}")
                return
    raise TimeoutError(f"no PARAM_VALUE for {name}: unknown parameter, or it needs a reboot to appear")


def set_params(conn: mavutil.mavfile, params: dict[str, float]) -> None:
    for name, value in params.items():
        set_param(conn, name, value)


def get_param(conn: mavutil.mavfile, name: str) -> float:
    for _ in range(MESSAGE_RETRIES):
        conn.mav.param_request_read_send(conn.target_system, conn.target_component, name.encode(), -1)
        deadline = time.monotonic() + PARAM_TIMEOUT_S
        while time.monotonic() < deadline:
            message = conn.recv_match(type="PARAM_VALUE", blocking=True, timeout=PARAM_TIMEOUT_S)
            if message is not None and message.param_id == name:
                return message.param_value
    raise TimeoutError(f"no PARAM_VALUE for {name}")
