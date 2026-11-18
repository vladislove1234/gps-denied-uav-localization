"""Where fixes go: the flight controller's second GPS."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from uavloc.ardupilot.gps_input import DEFAULT_SATELLITES, send_gps_input
from uavloc.geo import LatLon, offset_latlon
from uavloc.onboard.contracts import AltitudeSource, Fix, MotionSource

ACCURACY_FLOOR_M = 3.0   # docs/ardupilot-integration.md section 5: the EKF should not trust a fix tighter than this

GpsSender = Callable[..., None]   # (conn, position, accuracy_m, unix_s, satellites[, altitude_amsl_m=])


@dataclass(frozen=True)
class GpsSinkConfig:
    accuracy_floor_m: float = ACCURACY_FLOOR_M
    satellites: int = DEFAULT_SATELLITES


class GpsInputSink:
    """Sends each accepted fix as `GPS_INPUT`, carried forward from capture time to send time.

    A no-fix sends nothing, and the last fix is never sent twice: the EKF would count each copy as new evidence.
    `motion=None` skips the carry-forward and stamps the fix with the local clock, for bench tests without a link.
    `altitude` supplies the AMSL height of the message: ArduPilot compares GPS 1 and GPS 2 in 3D before arming and
    refuses to arm when they differ by hundreds of metres. Without it the message carries altitude 0.
    """

    def __init__(self, conn: Any, motion: MotionSource | None = None, config: GpsSinkConfig = GpsSinkConfig(),
                 sender: GpsSender = send_gps_input, altitude: AltitudeSource | None = None) -> None:
        self.conn = conn
        self.motion = motion
        self.altitude = altitude
        self.config = config
        self.sender = sender
        self.sent = 0
        self.skipped_no_fix = 0
        self.skipped_stale = 0
        self.last_sent_position: LatLon | None = None

    def send(self, fix: Fix) -> None:
        if fix.position is None:
            self.skipped_no_fix += 1
            return
        try:
            position, unix_s = self._carried_forward(fix)
            extra = {} if self.altitude is None else {"altitude_amsl_m": self.altitude.altitude_amsl_m()}
        except LookupError:
            self.skipped_stale += 1   # the velocity history does not reach back to the capture: drop, never guess
            return
        accuracy_m = max(fix.uncertainty_m, self.config.accuracy_floor_m)
        self.sender(self.conn, position, accuracy_m, unix_s, self.config.satellites, **extra)
        self.sent += 1
        self.last_sent_position = position

    def _carried_forward(self, fix: Fix) -> tuple[LatLon, float]:
        if self.motion is None:
            return fix.position, time.time()
        east_m, north_m = self.motion.velocity_displacement_m(fix.time_s, self.motion.fc_time_s())
        return offset_latlon(fix.position, east_m, north_m), self.motion.unix_time_s()

