"""A live SITL flight: telemetry, track recording and a GPS_INPUT feed driven by simulator truth."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from pymavlink import mavutil

from uavloc.ardupilot.gps_input import send_gps_input
from uavloc.ardupilot.link import E7
from uavloc.ardupilot.vehicle import is_armed, override_rc, request_message_rate
from uavloc.geo import LatLon, offset_latlon

MM_PER_M = 1000.0
MS_PER_S = 1000.0
RECV_TIMEOUT_S = 1.0
TELEMETRY_RATE_HZ = 10.0
SLOW_RATE_HZ = 2.0
EKF_FLAG_POS_HORIZ_ABS = 16          # EKF_STATUS_FLAGS bit: absolute horizontal position estimate is valid
STREAMED_MESSAGES = {
    "GLOBAL_POSITION_INT": TELEMETRY_RATE_HZ,
    "SIMSTATE": TELEMETRY_RATE_HZ,
    "GPS2_RAW": SLOW_RATE_HZ,
    "EKF_STATUS_REPORT": SLOW_RATE_HZ,
    "MISSION_CURRENT": SLOW_RATE_HZ,
}


RC_RESEND_S = 1.0                    # an override lapses after 3 s (RC_OVERRIDE_TIME)
ARM_TIMEOUT_S = 120.0
ARM_RETRY_S = 3.0
FORCE_ARM_MAGIC = 0.0                # param2: 21196 would skip the pre-arm checks; we never do


class ArmingRefused(RuntimeError):
    pass


@dataclass(frozen=True)
class FeedConfig:
    rate_hz: float = TELEMETRY_RATE_HZ     # simulated time; the GPS counts as healthy only under 215 ms average interval
    noise_std_m: float = 0.0
    horiz_accuracy_m: float = 3.0
    bias_north_m: float = 0.0
    seed: int = 0


@dataclass(frozen=True)
class TrackPoint:
    sim_time_s: float
    truth: LatLon
    estimate: LatLon
    height_m: float
    mode: str
    armed: bool
    ekf_flags: int


class FlightSession:
    """Reads vehicle messages, records the track and feeds our position to GPS 2 while `feeding` is on."""

    def __init__(self, conn: mavutil.mavfile, feed: FeedConfig = FeedConfig()) -> None:
        self.conn = conn
        self.feed = feed
        self.feeding = False
        self.track: list[TrackPoint] = []
        self.texts: list[tuple[float, str]] = []
        self.gps2_fixes: list[LatLon] = []
        self.sim_time_s = 0.0
        self.mission_seq = 0
        self._rng = np.random.default_rng(feed.seed)
        self._truth: LatLon | None = None
        self._estimate = LatLon(0.0, 0.0)
        self._height_m = 0.0
        self._altitude_amsl_m = 0.0
        self._mode = ""
        self._armed = False
        self._ekf_flags = 0
        self._next_feed_s = 0.0
        self._next_rc_s = 0.0
        self.rc_hold: dict[int, int] = {}
        self.last_fix_s: float | None = None
        self._clock_offset_s = time.time()

    def request_streams(self) -> None:
        for name, rate_hz in STREAMED_MESSAGES.items():
            request_message_rate(self.conn, getattr(mavutil.mavlink, f"MAVLINK_MSG_ID_{name}"), rate_hz)

    def step(self) -> None:
        """Handle one incoming message."""
        message = self.conn.recv_match(blocking=True, timeout=RECV_TIMEOUT_S)
        if message is None:
            return
        kind = message.get_type()
        if kind == "SIMSTATE":
            self._truth = LatLon(message.lat / E7, message.lng / E7)
            self._feed_if_due()
        elif kind == "GLOBAL_POSITION_INT":
            self.sim_time_s = message.time_boot_ms / MS_PER_S
            self._estimate = LatLon(message.lat / E7, message.lon / E7)
            self._height_m = message.relative_alt / MM_PER_M
            self._altitude_amsl_m = message.alt / MM_PER_M
            self._record()
            self._resend_rc_if_due()
        elif kind == "HEARTBEAT" and message.type != mavutil.mavlink.MAV_TYPE_GCS:
            self._mode = mavutil.mode_string_v10(message)
            self._armed = is_armed(message)
        elif kind == "EKF_STATUS_REPORT":
            self._ekf_flags = message.flags
        elif kind == "MISSION_CURRENT":
            self.mission_seq = message.seq
        elif kind == "GPS2_RAW" and message.fix_type >= mavutil.mavlink.GPS_FIX_TYPE_3D_FIX:
            self.gps2_fixes.append(LatLon(message.lat / E7, message.lon / E7))
        elif kind == "STATUSTEXT":
            self.texts.append((self.sim_time_s, message.text))

    def _record(self) -> None:
        if self._truth is not None:
            self.track.append(TrackPoint(self.sim_time_s, self._truth, self._estimate, self._height_m,
                                         self._mode, self._armed, self._ekf_flags))

    def _resend_rc_if_due(self) -> None:
        if self.rc_hold and self.sim_time_s >= self._next_rc_s:
            override_rc(self.conn, self.rc_hold)
            self._next_rc_s = self.sim_time_s + RC_RESEND_S

    def _feed_if_due(self) -> None:
        if not self.feeding or self.sim_time_s < self._next_feed_s:
            return
        self._next_feed_s = self.sim_time_s + 1.0 / self.feed.rate_hz
        east_m, north_m = self._rng.normal(0.0, self.feed.noise_std_m, 2) if self.feed.noise_std_m else (0.0, 0.0)
        send_gps_input(self.conn, offset_latlon(self._truth, east_m, north_m + self.feed.bias_north_m), self.feed.horiz_accuracy_m,
                       self._clock_offset_s + self.sim_time_s, altitude_amsl_m=self._altitude_amsl_m)
        self.last_fix_s = self.sim_time_s

    def run_until(self, done: Callable[[TrackPoint], bool], timeout_sim_s: float) -> TrackPoint:
        """Pump messages until `done(latest track point)`; raises if simulated time runs out first."""
        deadline_s = self.sim_time_s + timeout_sim_s
        while self.sim_time_s < deadline_s:
            self.step()
            if self.track and done(self.track[-1]):
                return self.track[-1]
        raise TimeoutError(f"condition not met within {timeout_sim_s:g} simulated s; "
                           f"mode {self._mode}, last messages {[text for _, text in self.texts[-4:]]}")

    def run_for(self, duration_sim_s: float) -> None:
        end_s = self.sim_time_s + duration_sim_s
        while self.sim_time_s < end_s:
            self.step()

    def arm(self, timeout_sim_s: float = ARM_TIMEOUT_S) -> None:
        """Arm with every pre-arm check on, feeding GPS 2 meanwhile; raise with the vehicle's reasons."""
        deadline_s = self.sim_time_s + timeout_sim_s
        while self.sim_time_s < deadline_s:
            self.conn.mav.command_long_send(self.conn.target_system, self.conn.target_component,
                                            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
                                            1, FORCE_ARM_MAGIC, 0, 0, 0, 0, 0)
            retry_s = self.sim_time_s + ARM_RETRY_S
            while self.sim_time_s < retry_s:
                self.step()
                if self._armed:
                    return
        raise ArmingRefused(f"not armed after {timeout_sim_s:g} simulated s; "
                            f"last messages: {[text for _, text in self.texts[-5:]]}")
