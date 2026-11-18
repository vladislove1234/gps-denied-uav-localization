"""Fakes for the onboard tests: a MAVLink connection that replays messages, and tiny sources and sinks."""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import numpy as np

from uavloc.data.satellite import GeoImage
from uavloc.geo import GeoBounds, LatLon, offset_latlon
from uavloc.onboard.contracts import Fix, FlightState, Frame
from uavloc.planning.grid import cost_grid
from uavloc.planning.planner import PlanningResult
from uavloc.planning.routes import BlindStretch, Route

ORIGIN = LatLon(49.9, 23.9)
WORLD_EAST_M = 1200.0
WORLD_NORTH_M = 600.0
CLASS_MPP_M = 2.0
BLIND_CELL_M = 25.0
N_CLASSES = 6
BLIND_FROM_EAST_M = 500.0
BLIND_TO_EAST_M = 700.0
ROUTE_NORTH_M = -300.0
ROUTE_EAST_M = (100.0, 600.0, 1100.0)


def at(east_m: float, north_m: float) -> LatLon:
    """A point `east_m`, `north_m` from `ORIGIN`, the world's north-west corner."""
    return offset_latlon(ORIGIN, east_m, north_m)


def world_corners() -> tuple[LatLon, LatLon]:
    return ORIGIN, at(WORLD_EAST_M, -WORLD_NORTH_M)


def synthetic_result() -> PlanningResult:
    top_left, bottom_right = world_corners()
    rng = np.random.default_rng(0)
    classes = rng.integers(1, 255, (int(WORLD_NORTH_M / CLASS_MPP_M), int(WORLD_EAST_M / CLASS_MPP_M), N_CLASSES), np.uint8)
    class_map = GeoImage(classes, top_left, bottom_right)
    rows, cols = int(WORLD_NORTH_M / BLIND_CELL_M), int(WORLD_EAST_M / BLIND_CELL_M)
    blind = np.zeros((rows, cols, 1), np.float32)
    blind[:, int(BLIND_FROM_EAST_M / BLIND_CELL_M):int(BLIND_TO_EAST_M / BLIND_CELL_M)] = 1.0
    blind_map = GeoImage(blind, top_left, bottom_right)
    area = GeoBounds(top_left.lat, top_left.lon, bottom_right.lat, bottom_right.lon)
    grid = cost_grid(blind_map, class_map, BLIND_CELL_M, (), 0.5)
    return PlanningResult(area, class_map, blind_map, grid, 64.0, [synthetic_route()])


def synthetic_route() -> Route:
    waypoints = tuple(at(east, ROUTE_NORTH_M) for east in ROUTE_EAST_M)
    blind_start_m = BLIND_FROM_EAST_M - ROUTE_EAST_M[0]
    return Route(0.0, waypoints, ROUTE_EAST_M[-1] - ROUTE_EAST_M[0], 0.2,
                 (BlindStretch(blind_start_m, BLIND_TO_EAST_M - ROUTE_EAST_M[0]),), 900.0, 64.0, 960.0)


def synthetic_satellite() -> GeoImage:
    """RGB over a slightly larger extent than the world, on a different pixel grid than the class map."""
    top_left, bottom_right = at(-50.0, 50.0), at(WORLD_EAST_M + 50.0, -WORLD_NORTH_M - 50.0)
    return GeoImage(np.full((340, 650, 3), 120, np.uint8), top_left, bottom_right)


class FakeMav:
    def __init__(self) -> None:
        self.commands: list[tuple] = []
        self.gps_inputs: list[tuple] = []

    def command_long_send(self, *args) -> None:
        self.commands.append(args)

    def gps_input_send(self, *args) -> None:
        self.gps_inputs.append(args)


class FakeConnection:
    """Replays queued messages through `recv_match`, records what is sent."""

    target_system = 1
    target_component = 1

    def __init__(self) -> None:
        self.mav = FakeMav()
        self.queue: deque = deque()

    def recv_match(self, type=None, blocking=False, timeout=None):
        return self.queue.popleft() if self.queue else None

    def wait_heartbeat(self, timeout=None):
        return object()


def message(kind: str, **fields) -> SimpleNamespace:
    return SimpleNamespace(get_type=lambda: kind, **fields)


def attitude(time_s: float, yaw_rad: float = 0.0) -> SimpleNamespace:
    return message("ATTITUDE", time_boot_ms=round(time_s * 1000), roll=0.0, pitch=0.0, yaw=yaw_rad)


def local_position(time_s: float, vx: float, vy: float) -> SimpleNamespace:
    return message("LOCAL_POSITION_NED", time_boot_ms=round(time_s * 1000), x=0.0, y=0.0, z=0.0, vx=vx, vy=vy, vz=0.0)


def global_position(time_s: float, position: LatLon, relative_alt_m: float = 40.0) -> SimpleNamespace:
    return message("GLOBAL_POSITION_INT", time_boot_ms=round(time_s * 1000), lat=round(position.lat * 1e7),
                   lon=round(position.lon * 1e7), alt=0, relative_alt=round(relative_alt_m * 1000))


def simstate(position: LatLon) -> SimpleNamespace:
    return message("SIMSTATE", lat=round(position.lat * 1e7), lng=round(position.lon * 1e7))


def system_time(unix_s: float, time_s: float) -> SimpleNamespace:
    return message("SYSTEM_TIME", time_unix_usec=round(unix_s * 1e6), time_boot_ms=round(time_s * 1000))


def stream(conn: FakeConnection, start_s: float, stop_s: float, velocity: tuple[float, float] = (0.0, 0.0),
           position: LatLon = ORIGIN, hz: float = 20.0) -> None:
    """Queue attitude, velocity, position and simulator truth for a vehicle moving at a constant velocity."""
    for k in range(int(round((stop_s - start_s) * hz)) + 1):
        t = start_s + k / hz
        here = offset_latlon(position, velocity[1] * (t - start_s), velocity[0] * (t - start_s))
        conn.queue.extend([attitude(t), local_position(t, *velocity), global_position(t, here), simstate(here)])


class ScriptedLocalizer:
    def __init__(self, fixes: list[Fix] | None = None, error: Exception | None = None) -> None:
        self.fixes = deque(fixes or [])
        self.error = error
        self.calls: list[tuple[Frame | None, FlightState]] = []

    def locate(self, frame, state, reference) -> Fix:
        self.calls.append((frame, state))
        if self.error:
            raise self.error
        return self.fixes.popleft()


class RecordingSink:
    def __init__(self) -> None:
        self.fixes: list[Fix] = []

    def send(self, fix: Fix) -> None:
        self.fixes.append(fix)


class RecordingLogger:
    def __init__(self) -> None:
        self.fixes, self.states, self.frames, self.closed = [], [], [], False

    def log_fix(self, fix) -> None:
        self.fixes.append(fix)

    def log_state(self, state) -> None:
        self.states.append(state)

    def log_frame(self, frame) -> None:
        self.frames.append(frame)

    def close(self) -> None:
        self.closed = True
