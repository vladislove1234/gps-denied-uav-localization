from dataclasses import replace

import pytest

from uavloc.ardupilot.gps_input import gps_week_and_ms
from uavloc.ardupilot.mission import (
    MAV_CMD_DO_CHANGE_SPEED, MAV_CMD_NAV_LAND, MAV_CMD_NAV_LOITER_TIME, MAV_CMD_NAV_TAKEOFF,
    MAV_CMD_NAV_WAYPOINT, MAV_FRAME_GLOBAL_RELATIVE_ALT, MissionPlan, mission_differences, mission_items,
    read_waypoints,
    write_waypoints,
)
from uavloc.ardupilot.params import read_param_file, vehicle_params, write_param_file
from uavloc.ardupilot.scripts import lua_script_path
from uavloc.geo import LatLon

START, MID, END = LatLon(49.905, 23.885), LatLon(49.910, 23.900), LatLon(49.915, 23.925)
ALTITUDE_M, SPEED_MPS, HOVER_S = 40.0, 5.0, 12.0


def commands(items):
    return [item.command for item in items]


def test_out_and_back_sequence_without_hover():
    items = mission_items(MissionPlan((START, MID, END), ALTITUDE_M, SPEED_MPS))
    assert commands(items) == [MAV_CMD_NAV_WAYPOINT, MAV_CMD_DO_CHANGE_SPEED, MAV_CMD_NAV_TAKEOFF,
                               MAV_CMD_NAV_WAYPOINT, MAV_CMD_NAV_WAYPOINT,
                               MAV_CMD_NAV_WAYPOINT, MAV_CMD_NAV_LAND]
    assert [item.seq for item in items] == list(range(len(items)))
    assert items[0].current == 1
    assert items[1].param2 == SPEED_MPS
    out, back = items[3:5], items[5:6]
    assert [(item.lat, item.lon) for item in out] == [MID, END]
    assert [(item.lat, item.lon) for item in back] == [MID]
    assert all(item.alt == ALTITUDE_M and item.frame == MAV_FRAME_GLOBAL_RELATIVE_ALT for item in items[2:-1])
    assert (items[-1].lat, items[-1].lon, items[-1].alt) == (START.lat, START.lon, 0.0)


def test_hover_adds_loiter_at_the_end_point():
    items = mission_items(MissionPlan((START, MID, END), ALTITUDE_M, SPEED_MPS, HOVER_S))
    loiter = [item for item in items if item.command == MAV_CMD_NAV_LOITER_TIME]
    assert len(loiter) == 1
    assert (loiter[0].lat, loiter[0].lon, loiter[0].param1) == (END.lat, END.lon, HOVER_S)
    assert items[items.index(loiter[0]) - 1].command == MAV_CMD_NAV_WAYPOINT


def test_straight_route_has_no_return_waypoints():
    items = mission_items(MissionPlan((START, END), ALTITUDE_M, SPEED_MPS))
    assert commands(items)[-2:] == [MAV_CMD_NAV_WAYPOINT, MAV_CMD_NAV_LAND]
    assert commands(items).count(MAV_CMD_NAV_WAYPOINT) == 2     # home placeholder + end point


def test_single_point_is_rejected():
    with pytest.raises(ValueError):
        mission_items(MissionPlan((START,), ALTITUDE_M, SPEED_MPS))


def test_waypoint_file_round_trip(tmp_path):
    items = mission_items(MissionPlan((START, MID, END), ALTITUDE_M, SPEED_MPS, HOVER_S))
    path = tmp_path / "mission.waypoints"
    write_waypoints(path, items)
    assert path.read_text().splitlines()[0] == "QGC WPL 110"
    assert read_waypoints(path) == items


def test_waypoint_file_keeps_centimetre_precision(tmp_path):
    long_lon = LatLon(49.90589831528412, 23.88709216449324)
    items = mission_items(MissionPlan((START, long_lon, END), ALTITUDE_M, SPEED_MPS))
    path = tmp_path / "mission.waypoints"
    write_waypoints(path, items)
    assert mission_differences(items, read_waypoints(path)) == []


def test_waypoint_file_rejects_other_formats(tmp_path):
    path = tmp_path / "bad.waypoints"
    path.write_text("QGC WPL 120\n")
    with pytest.raises(ValueError):
        read_waypoints(path)


def test_param_file_round_trip(tmp_path):
    params = vehicle_params()
    path = tmp_path / "uavloc.param"
    write_param_file(path, params)
    assert read_param_file(path) == params


def test_vehicle_params_fly_on_our_gps_without_velocity():
    params = vehicle_params()
    assert params["GPS2_TYPE"] == 14
    assert params["GPS_AUTO_SWITCH"] == 0
    assert params["EK3_SRC2_POSXY"] == 3 and params["EK3_SRC2_VELXY"] == 0


def test_lua_script_is_packaged():
    assert "GPS_PRIMARY" in lua_script_path().read_text()


def test_gps_week_and_ms():
    week, week_ms = gps_week_and_ms(315_964_800 - 18 + 604_800 + 1.5)
    assert (week, week_ms) == (1, 1500)


def test_mission_differences_ignores_home_rewrite_and_flags_real_changes():
    sent = mission_items(MissionPlan((START, MID, END), ALTITUDE_M, SPEED_MPS))
    received = [replace(item, alt=300.1 if item.seq == 0 else item.alt) for item in sent]
    assert mission_differences(sent, received) == []
    moved = [replace(item, lat=item.lat + 1e-4) if item.seq == 3 else item for item in sent]
    assert mission_differences(sent, moved) == ["item 3: position moved"]
    assert mission_differences(sent, sent[:-1]) == ["7 items sent, 6 received"]
