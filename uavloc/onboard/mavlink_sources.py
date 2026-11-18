"""Flight state and simulator truth read from a MAVLink connection, with a short history for interpolation."""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Any, Callable

import numpy as np
from pymavlink import mavutil

from uavloc.geo import LatLon
from uavloc.onboard.contracts import FlightState

BUFFER_S = 2.0                       # history kept: docs/ardupilot-integration.md section 3
ATTITUDE_HZ = 50.0
LOCAL_POSITION_HZ = 20.0
GLOBAL_POSITION_HZ = 20.0
SYSTEM_TIME_HZ = 1.0
SIM_STATE_HZ = 50.0
MAX_MESSAGES_PER_POLL = 5000         # a stalled loop drains its backlog, but never starves the caller
DEGE7 = 1e-7
MM_PER_M = 1000.0
MS_PER_S = 1000.0
US_PER_S = 1e6
YAW_COLUMN = 2                       # attitude columns: roll, pitch, yaw
AMSL_COLUMN = 3                      # position columns: lat, lon, altitude relative to home, altitude AMSL
CLOCK_BACKSTEP_S = 1.0               # a sample this much older than the newest means the flight controller rebooted

MSG_ATTITUDE = "ATTITUDE"
MSG_LOCAL_POSITION = "LOCAL_POSITION_NED"
MSG_GLOBAL_POSITION = "GLOBAL_POSITION_INT"
MSG_SYSTEM_TIME = "SYSTEM_TIME"
MSG_SIMSTATE = "SIMSTATE"
MSG_SIM_STATE = "SIM_STATE"
STREAM_MESSAGE_IDS = {MSG_ATTITUDE: 30, MSG_LOCAL_POSITION: 32, MSG_GLOBAL_POSITION: 33, MSG_SYSTEM_TIME: 2,
                      MSG_SIMSTATE: 164, MSG_SIM_STATE: 108}

Handler = Callable[[Any], None]


class SampleBuffer:
    """Time-ordered samples of a few numbers over the last `span_s`; linear interpolation between them."""

    def __init__(self, angular: tuple[int, ...] = (), span_s: float = BUFFER_S) -> None:
        self.angular = angular           # columns holding degrees that wrap at 360
        self.span_s = span_s
        self._times: deque[float] = deque()
        self._values: deque[np.ndarray] = deque()

    def __len__(self) -> int:
        return len(self._times)

    @property
    def oldest_s(self) -> float:
        return self._times[0]

    @property
    def newest_s(self) -> float:
        return self._times[-1]

    def add(self, time_s: float, values: list[float]) -> None:
        if self._times and time_s < self.newest_s - CLOCK_BACKSTEP_S:
            self._times.clear()
            self._values.clear()
        if self._times and time_s <= self.newest_s:
            return
        self._times.append(time_s)
        self._values.append(np.asarray(values, dtype=float))
        while self._times[0] < time_s - self.span_s:
            self._times.popleft()
            self._values.popleft()

    def _columns(self) -> np.ndarray:
        columns = np.stack(self._values)
        for index in self.angular:
            columns[:, index] = np.degrees(np.unwrap(np.radians(columns[:, index])))
        return columns

    def at(self, time_s: float) -> np.ndarray | None:
        """Interpolated values; the newest sample is held for later times, `None` for earlier than the oldest."""
        if not self._times or time_s < self.oldest_s:
            return None
        times, columns = np.asarray(self._times), self._columns()
        out = np.array([np.interp(time_s, times, columns[:, i]) for i in range(columns.shape[1])])
        for index in self.angular:
            out[index] %= 360.0
        return out

    def integral(self, from_s: float, to_s: float) -> np.ndarray:
        """Trapezoid integral of every column over `[from_s, to_s]`; the newest sample is held past its time."""
        if not self._times or from_s < self.oldest_s:
            raise LookupError(f"no samples back to {from_s:.3f} s")
        times, columns = np.asarray(self._times), np.stack(self._values)
        inside = times[(times > from_s) & (times < to_s)]
        grid = np.concatenate([[from_s], inside, [to_s]])
        values = np.stack([np.interp(grid, times, columns[:, i]) for i in range(columns.shape[1])], axis=1)
        return ((values[1:] + values[:-1]) / 2 * np.diff(grid)[:, None]).sum(axis=0)


class MavlinkRouter:
    """The one reader of a connection: several sources subscribe to message types, `poll()` drains and dispatches.

    Without it, two sources reading the same connection would take each other's messages.
    """

    def __init__(self, conn: Any, monotonic: Callable[[], float] = time.monotonic,
                 wall_clock: Callable[[], float] = time.time) -> None:
        self.conn = conn
        self._monotonic = monotonic
        self._wall_clock = wall_clock
        self._handlers: dict[str, list[Handler]] = {}
        self.last_boot_time_s: float | None = None
        self._boot_arrival_s = 0.0
        self._unix_offset_s: float | None = None

    @classmethod
    def for_connection(cls, conn: Any) -> MavlinkRouter:
        if not hasattr(conn, "_uavloc_router"):
            conn._uavloc_router = cls(conn)
        return conn._uavloc_router

    def subscribe(self, message_type: str, handler: Handler) -> None:
        self._handlers.setdefault(message_type, []).append(handler)

    def request_interval(self, message_type: str, rate_hz: float) -> None:
        """SET_MESSAGE_INTERVAL for a message the flight controller does not stream by default."""
        self.conn.mav.command_long_send(
            self.conn.target_system, self.conn.target_component, mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
            STREAM_MESSAGE_IDS[message_type], US_PER_S / rate_hz, 0, 0, 0, 0, 0)

    def poll(self) -> int:
        """Dispatch every message waiting on the connection; returns how many."""
        count = 0
        while count < MAX_MESSAGES_PER_POLL:
            message = self.conn.recv_match(blocking=False)
            if message is None:
                break
            count += 1
            self._dispatch(message)
        return count

    def _dispatch(self, message: Any) -> None:
        kind = message.get_type()
        if hasattr(message, "time_boot_ms"):
            self.last_boot_time_s = message.time_boot_ms / MS_PER_S
            self._boot_arrival_s = self._monotonic()
        if kind == MSG_SYSTEM_TIME and message.time_unix_usec > 0:
            self._unix_offset_s = message.time_unix_usec / US_PER_S - message.time_boot_ms / MS_PER_S
        for handler in self._handlers.get(kind, ()):
            handler(message)

    def fc_time_s(self) -> float:
        """Flight-controller time now: the newest message's time plus how long ago it arrived."""
        if self.last_boot_time_s is None:
            raise LookupError("no message with a timestamp has arrived yet")
        return self.last_boot_time_s + self._monotonic() - self._boot_arrival_s

    def unix_time_s(self) -> float:
        """Unix time from the flight controller's SYSTEM_TIME; the local clock until the flight controller has one."""
        if self._unix_offset_s is None:
            return self._wall_clock()
        return self.fc_time_s() + self._unix_offset_s


def connect(url: str, timeout_s: float) -> Any:
    """Open `url` and wait for the first heartbeat.

    The same as `uavloc.ardupilot.link.connect`, kept here because that module imports the mission planner,
    which pulls torch onto the Pi.
    """
    conn = mavutil.mavlink_connection(url)
    if conn.wait_heartbeat(timeout=timeout_s) is None:
        raise TimeoutError(f"no MAVLink heartbeat from {url} within {timeout_s:g} s")
    return conn


def router_of(source: Any) -> MavlinkRouter:
    return source if isinstance(source, MavlinkRouter) else MavlinkRouter.for_connection(source)


class MavlinkFlightState:
    """`FlightStateSource` and `MotionSource` over ATTITUDE, LOCAL_POSITION_NED and GLOBAL_POSITION_INT.

    `source` is a pymavlink connection or the `MavlinkRouter` already reading it.
    """

    def __init__(self, source: Any, request_streams: bool = True) -> None:
        self.router = router_of(source)
        self._attitude = SampleBuffer(angular=(YAW_COLUMN,))   # roll, pitch, yaw
        self._velocity = SampleBuffer()                       # north, east
        self._position = SampleBuffer()                       # lat, lon, altitude relative to home, altitude AMSL
        self.router.subscribe(MSG_ATTITUDE, self._on_attitude)
        self.router.subscribe(MSG_LOCAL_POSITION, self._on_local_position)
        self.router.subscribe(MSG_GLOBAL_POSITION, self._on_global_position)
        if request_streams:
            for message_type, rate_hz in ((MSG_ATTITUDE, ATTITUDE_HZ), (MSG_LOCAL_POSITION, LOCAL_POSITION_HZ),
                                          (MSG_GLOBAL_POSITION, GLOBAL_POSITION_HZ), (MSG_SYSTEM_TIME, SYSTEM_TIME_HZ)):
                self.router.request_interval(message_type, rate_hz)

    def _on_attitude(self, message: Any) -> None:
        self._attitude.add(message.time_boot_ms / MS_PER_S,
                           [math.degrees(message.roll), math.degrees(message.pitch), math.degrees(message.yaw)])

    def _on_local_position(self, message: Any) -> None:
        self._velocity.add(message.time_boot_ms / MS_PER_S, [message.vx, message.vy])

    def _on_global_position(self, message: Any) -> None:
        self._position.add(message.time_boot_ms / MS_PER_S,
                           [message.lat * DEGE7, message.lon * DEGE7, message.relative_alt / MM_PER_M,
                            message.alt / MM_PER_M])

    def state_at(self, time_s: float) -> FlightState | None:
        self.router.poll()
        attitude, velocity, position = (b.at(time_s) for b in (self._attitude, self._velocity, self._position))
        if attitude is None or velocity is None or position is None:
            return None
        lat, lon, altitude_m, _ = position
        has_position = not (lat == 0.0 and lon == 0.0)
        return FlightState(time_s, LatLon(lat, lon) if has_position else None, float(altitude_m),
                           yaw_deg=float(attitude[YAW_COLUMN]), roll_deg=float(attitude[0]), pitch_deg=float(attitude[1]),
                           velocity_north_mps=float(velocity[0]), velocity_east_mps=float(velocity[1]))

    def latest(self) -> FlightState | None:
        self.router.poll()
        buffers = (self._attitude, self._velocity, self._position)
        if not all(len(b) for b in buffers):
            return None
        return self.state_at(max(b.newest_s for b in buffers))

    def altitude_amsl_m(self) -> float:
        self.router.poll()
        if not len(self._position):
            raise LookupError("no GLOBAL_POSITION_INT has arrived yet")
        return float(self._position.at(self._position.newest_s)[AMSL_COLUMN])

    def fc_time_s(self) -> float:
        self.router.poll()
        return self.router.fc_time_s()

    def unix_time_s(self) -> float:
        self.router.poll()
        return self.router.unix_time_s()

    def velocity_displacement_m(self, from_s: float, to_s: float) -> tuple[float, float]:
        """`(east_m, north_m)` flown between two flight-controller times, integrating velocity.

        Velocity, not position differences: the EKF position jumps when an earlier fix is fused, the velocity does not.
        """
        if to_s < from_s:
            raise ValueError(f"to_s={to_s} is before from_s={from_s}")
        self.router.poll()
        north_m, east_m = self._velocity.integral(from_s, to_s)
        return float(east_m), float(north_m)


class ArdupilotHeight:
    """Height from ArduPilot's altitude above home (GPS + barometer)."""

    def height_m(self, state: FlightState) -> float:
        return state.altitude_m


class SimTruth:
    """`TruthSource` from the simulator's SIMSTATE (or SIM_STATE) messages. Exists only in SITL, for the stub."""

    def __init__(self, source: Any, request_streams: bool = True) -> None:
        self.router = router_of(source)
        self._position = SampleBuffer()
        self.router.subscribe(MSG_SIMSTATE, self._on_simstate)
        self.router.subscribe(MSG_SIM_STATE, self._on_sim_state)
        if request_streams:
            self.router.request_interval(MSG_SIMSTATE, SIM_STATE_HZ)

    def _add(self, lat: float, lon: float) -> None:
        # SIMSTATE has no timestamp; the newest timestamped message before it dates it, within one ATTITUDE period
        if self.router.last_boot_time_s is not None:
            self._position.add(self.router.last_boot_time_s, [lat, lon])

    def _on_simstate(self, message: Any) -> None:
        self._add(message.lat * DEGE7, message.lng * DEGE7)

    def _on_sim_state(self, message: Any) -> None:
        self._add(message.lat_int * DEGE7, message.lon_int * DEGE7)

    def position_at(self, time_s: float) -> LatLon | None:
        self.router.poll()
        values = self._position.at(time_s)
        return None if values is None else LatLon(float(values[0]), float(values[1]))
