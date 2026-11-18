"""Out-and-back ArduCopter mission from a planned route, and the QGC WPL 110 text format."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from uavloc.geo import LatLon, haversine_m
from uavloc.planning.config import MissionInputs
from uavloc.planning.routes import Route

MAV_FRAME_GLOBAL = 0
MAV_FRAME_GLOBAL_RELATIVE_ALT = 3
MAV_FRAME_MISSION = 2

MAV_CMD_NAV_WAYPOINT = 16
MAV_CMD_NAV_LOITER_TIME = 19
MAV_CMD_NAV_LAND = 21
MAV_CMD_NAV_TAKEOFF = 22
MAV_CMD_DO_CHANGE_SPEED = 178

SPEED_TYPE_GROUND = 1          # MAV_CMD_DO_CHANGE_SPEED param1: 0 = airspeed, 1 = ground speed
SPEED_UNCHANGED = -1.0         # param3 (throttle) and other "leave as is" fields
WAYPOINT_FILE_HEADER = "QGC WPL 110"
WAYPOINT_FILE_FIELDS = 12
MIN_WAYPOINTS = 2
FLOAT_FORMAT = ".8f"
POSITION_COMMANDS = (MAV_CMD_NAV_WAYPOINT, MAV_CMD_NAV_LOITER_TIME, MAV_CMD_NAV_LAND, MAV_CMD_NAV_TAKEOFF)
HOME_SEQ = 0
POSITION_TOLERANCE_M = 0.1
ALTITUDE_TOLERANCE_M = 0.01
PARAM_TOLERANCE = 1e-6


@dataclass(frozen=True)
class MissionPlan:
    waypoints: tuple[LatLon, ...]        # start, corners, end
    altitude_m: float                    # relative to takeoff
    speed_mps: float
    hover_s: float = 0.0

    @classmethod
    def from_route(cls, route: Route, inputs: MissionInputs) -> MissionPlan:
        return cls(route.waypoints, inputs.altitude_m, inputs.speed_mps, inputs.hover_s)


@dataclass(frozen=True)
class MissionItem:
    seq: int
    frame: int
    command: int
    current: int
    autocontinue: int
    param1: float
    param2: float
    param3: float
    param4: float
    lat: float
    lon: float
    alt: float


def navigation_item(command: int, position: LatLon, alt_m: float, param1: float = 0.0) -> MissionItem:
    return MissionItem(0, MAV_FRAME_GLOBAL_RELATIVE_ALT, command, 0, 1, param1, 0.0, 0.0, 0.0,
                       position.lat, position.lon, alt_m)


def home_item(start: LatLon) -> MissionItem:
    return MissionItem(0, MAV_FRAME_GLOBAL, MAV_CMD_NAV_WAYPOINT, 1, 1, 0.0, 0.0, 0.0, 0.0,
                       start.lat, start.lon, 0.0)


def speed_item(speed_mps: float) -> MissionItem:
    return MissionItem(0, MAV_FRAME_MISSION, MAV_CMD_DO_CHANGE_SPEED, 0, 1,
                       SPEED_TYPE_GROUND, speed_mps, SPEED_UNCHANGED, 0.0, 0.0, 0.0, 0.0)


def mission_items(plan: MissionPlan) -> list[MissionItem]:
    """Home, speed, takeoff at the start, out along the route, optional hover, back reversed, land at the start."""
    if len(plan.waypoints) < MIN_WAYPOINTS:
        raise ValueError(f"a mission needs a start and an end point, got {len(plan.waypoints)} waypoints")
    start, end = plan.waypoints[0], plan.waypoints[-1]
    out = plan.waypoints[1:]
    back = plan.waypoints[-2:0:-1]
    items = [home_item(start), speed_item(plan.speed_mps),
             navigation_item(MAV_CMD_NAV_TAKEOFF, start, plan.altitude_m)]
    items += [navigation_item(MAV_CMD_NAV_WAYPOINT, point, plan.altitude_m) for point in out]
    if plan.hover_s > 0:
        items.append(navigation_item(MAV_CMD_NAV_LOITER_TIME, end, plan.altitude_m, plan.hover_s))
    items += [navigation_item(MAV_CMD_NAV_WAYPOINT, point, plan.altitude_m) for point in back]
    items.append(navigation_item(MAV_CMD_NAV_LAND, start, 0.0))
    return [replace(item, seq=seq) for seq, item in enumerate(items)]


def waypoint_line(item: MissionItem) -> str:
    numbers = [f"{value:{FLOAT_FORMAT}}" for value in
               (item.param1, item.param2, item.param3, item.param4, item.lat, item.lon, item.alt)]
    return "\t".join([str(item.seq), str(item.current), str(item.frame), str(item.command), *numbers,
                      str(item.autocontinue)])


def write_waypoints(path: Path, items: list[MissionItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([WAYPOINT_FILE_HEADER, *map(waypoint_line, items)]) + "\n")


def parse_waypoint_line(line: str) -> MissionItem:
    fields = line.split()
    if len(fields) != WAYPOINT_FILE_FIELDS:
        raise ValueError(f"expected {WAYPOINT_FILE_FIELDS} fields, got {len(fields)}: {line!r}")
    seq, current, frame, command = (int(field) for field in fields[:4])
    p1, p2, p3, p4, lat, lon, alt = (float(field) for field in fields[4:11])
    return MissionItem(seq, frame, command, current, int(fields[11]), p1, p2, p3, p4, lat, lon, alt)


def read_waypoints(path: Path) -> list[MissionItem]:
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    if not lines or lines[0].strip() != WAYPOINT_FILE_HEADER:
        raise ValueError(f"{path} is not a {WAYPOINT_FILE_HEADER} file")
    return [parse_waypoint_line(line) for line in lines[1:]]


def mission_differences(sent: list[MissionItem], received: list[MissionItem]) -> list[str]:
    """What the vehicle changed in a mission, ignoring what ArduPilot rewrites by design.

    It overwrites item 0 with its own home, drops the frame of non-position commands and may set a land
    option in param4, so those are not compared.
    """
    if len(sent) != len(received):
        return [f"{len(sent)} items sent, {len(received)} received"]
    differences = []
    for a, b in zip(sent, received):
        if a.seq == HOME_SEQ:
            continue
        if (a.seq, a.command) != (b.seq, b.command):
            differences.append(f"item {a.seq}: command {a.command} became {b.command}")
            continue
        if any(abs(x - y) > PARAM_TOLERANCE for x, y in ((a.param1, b.param1), (a.param2, b.param2),
                                                         (a.param3, b.param3))):
            differences.append(f"item {a.seq}: parameters changed")
        if a.command in POSITION_COMMANDS:
            if haversine_m(LatLon(a.lat, a.lon), LatLon(b.lat, b.lon)) > POSITION_TOLERANCE_M:
                differences.append(f"item {a.seq}: position moved")
            if a.frame != b.frame or abs(a.alt - b.alt) > ALTITUDE_TOLERANCE_M:
                differences.append(f"item {a.seq}: altitude or frame changed")
    return differences
