"""Our position as a MAVLink GPS_INPUT for GPS 2 (docs/ardupilot-integration.md sections 2 and 5)."""

from __future__ import annotations

from pymavlink import mavutil

from uavloc.geo import LatLon

GPS_EPOCH_UNIX_S = 315_964_800
GPS_LEAP_S = 18
SECONDS_PER_WEEK = 604_800
MS_PER_S = 1000
DEGREES_TO_E7 = 1e7

GPS_ID_SECOND = 1                    # AP_GPS_MAV drops a message whose gps_id is not its own instance
FIX_TYPE_3D = 3                      # below 3 the EKF ignores the fix
DEFAULT_SATELLITES = 10              # fewer than 6 inflates the EKF's position noise
HDOP = 1.0

# Ignore vertical dop, both velocities, speed accuracy and vertical accuracy: we measure position only. With the
# velocity flags set the driver leaves the velocity at zero, which is why EK3 source set 2 takes no GPS velocity.
IGNORE_FLAGS = (mavutil.mavlink.GPS_INPUT_IGNORE_FLAG_VDOP
                | mavutil.mavlink.GPS_INPUT_IGNORE_FLAG_VEL_HORIZ
                | mavutil.mavlink.GPS_INPUT_IGNORE_FLAG_VEL_VERT
                | mavutil.mavlink.GPS_INPUT_IGNORE_FLAG_SPEED_ACCURACY
                | mavutil.mavlink.GPS_INPUT_IGNORE_FLAG_VERTICAL_ACCURACY)


def gps_week_and_ms(time_s: float) -> tuple[int, int]:
    """`(week, millisecond of week)` for UNIX seconds."""
    gps_s = time_s - GPS_EPOCH_UNIX_S + GPS_LEAP_S
    return int(gps_s // SECONDS_PER_WEEK), int(gps_s % SECONDS_PER_WEEK * MS_PER_S)


def send_gps_input(conn: mavutil.mavfile, position: LatLon, horiz_accuracy_m: float, time_s: float,
                   satellites: int = DEFAULT_SATELLITES, altitude_amsl_m: float = 0.0) -> None:
    week, week_ms = gps_week_and_ms(time_s)
    conn.mav.gps_input_send(
        0, GPS_ID_SECOND, IGNORE_FLAGS, week_ms, week, FIX_TYPE_3D,
        int(round(position.lat * DEGREES_TO_E7)), int(round(position.lon * DEGREES_TO_E7)), altitude_amsl_m,
        HDOP, 0.0, 0.0, 0.0, 0.0, 0.0, horiz_accuracy_m, 0.0, satellites)
