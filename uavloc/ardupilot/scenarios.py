"""SITL verification scenarios: a full mission, GPS_INPUT reaching GPS 2, and IMU-only coasting."""

from __future__ import annotations

from dataclasses import dataclass, replace

from uavloc.ardupilot.flight import EKF_FLAG_POS_HORIZ_ABS, FlightSession, TrackPoint
from uavloc.ardupilot.link import connect, set_params, upload_mission, vehicle_home
from uavloc.ardupilot.mission import MissionItem
from uavloc.ardupilot.sitl import SitlConfig, start_sitl
from uavloc.ardupilot.vehicle import set_mode
from uavloc.geo import LatLon, haversine_m

GPS_SETTLE_S = 10.0                  # simulated seconds of fixes before arming, so GPS 2 is healthy
AIRBORNE_HEIGHT_M = 5.0
GPS_CHECK_S = 4.0
GPS_CHECK_BIAS_M = 30.0              # below the 50 m pre-arm limit between GPS 1 and 2
GPS_CHECK_TOLERANCE_M = 0.5
SWITCH_CHANNEL = 9
SWITCH_HIGH_PWM = 2000
SWITCH_TIMEOUT_S = 10.0
SIM_REAL_GPS_ENABLE = "SIM_GPS1_ENABLE"
AUTO_OPTIONS_ARM_AND_TAKEOFF = 3     # bit 0 allow arming in Auto, bit 1 take off without raising throttle
AUTO_OPTIONS_PARAM = "AUTO_OPTIONS"
CRUISE_SETTLE_S = 20.0
COAST_TIMEOUT_S = 240.0
FIRST_ROUTE_WAYPOINT_SEQ = 3         # home, speed, takeoff, then the route
CONNECT_TIMEOUT_S = 30.0
HOME_TIMEOUT_S = 120.0
EKF_ACTION_PARAM = "FS_EKF_ACTION"


@dataclass(frozen=True)
class MissionFlight:
    track: list[TrackPoint]
    texts: list[tuple[float, str]]
    landed: LatLon
    closest_to_end_m: float
    max_height_m: float
    duration_sim_s: float

    def landed_offset_m(self, start: LatLon) -> float:
        return haversine_m(self.landed, start)


@dataclass(frozen=True)
class GpsInputCheck:
    sent_bias_north_m: float
    fixes_seen: int
    max_error_m: float

    @property
    def passed(self) -> bool:
        return self.fixes_seen > 0 and self.max_error_m <= GPS_CHECK_TOLERANCE_M


@dataclass(frozen=True)
class CoastResult:
    stop_sim_s: float
    position_lost_s: float | None        # after the last fix, until the EKF drops its absolute position
    failsafe_s: float | None             # after the last fix, until the mode left AUTO; None if it never did
    mode_after_failsafe: str
    error_m: list[tuple[float, float]]   # (seconds after the last fix, EKF position error vs truth)
    texts: list[tuple[float, str]]


def enable_auto_takeoff(session: FlightSession) -> None:
    set_params(session.conn, {AUTO_OPTIONS_PARAM: AUTO_OPTIONS_ARM_AND_TAKEOFF})


def check_gps_input(session: FlightSession) -> GpsInputCheck:
    """Feed a deliberately offset position and see the same offset in GPS2_RAW."""
    session.feeding = True
    session.run_for(GPS_SETTLE_S)
    session.gps2_fixes.clear()
    biased = replace(session.feed, bias_north_m=GPS_CHECK_BIAS_M)
    unbiased, session.feed = session.feed, biased
    session.run_for(GPS_CHECK_S)
    seen, truth = list(session.gps2_fixes), session.track[-1].truth
    session.feed = unbiased
    errors_m = [abs(haversine_m(fix, truth) - GPS_CHECK_BIAS_M) for fix in seen[1:]]
    session.run_for(GPS_SETTLE_S)
    return GpsInputCheck(GPS_CHECK_BIAS_M, len(seen), max(errors_m, default=float("inf")))


def fly_mission(session: FlightSession, start: LatLon, end: LatLon, timeout_sim_s: float) -> MissionFlight:
    """Fly the uploaded mission in AUTO on whichever GPS is primary, from arming to disarming after landing."""
    session.feeding = True
    session.run_for(GPS_SETTLE_S)
    set_mode(session.conn, "AUTO")
    session.arm()
    first = len(session.track)
    session.run_until(lambda p: p.height_m > AIRBORNE_HEIGHT_M, timeout_sim_s)
    session.run_until(lambda p: not p.armed, timeout_sim_s)
    flown = session.track[first:]
    return MissionFlight(flown, list(session.texts), flown[-1].truth,
                         min(haversine_m(p.truth, end) for p in flown), max(p.height_m for p in flown),
                         flown[-1].sim_time_s - flown[0].sim_time_s)


def switch_to_vision(session: FlightSession) -> None:
    """Flip the transmitter switch the Lua script watches, then wait for its confirmation."""
    session.rc_hold = {SWITCH_CHANNEL: SWITCH_HIGH_PWM}
    deadline_s = session.sim_time_s + SWITCH_TIMEOUT_S
    while session.sim_time_s < deadline_s:
        session.step()
        if any("Vision GPS: ON" in text for _, text in session.texts):
            return
    raise TimeoutError("the Lua script did not confirm 'Vision GPS: ON': is the script loaded and RC9_OPTION 300?")


def measure_coast(session: FlightSession, use_vision: bool, observe_s: float = COAST_TIMEOUT_S) -> CoastResult:
    """Fly AUTO to cruise, take the real GPS away, stop our fixes, and time how long the EKF holds on."""
    session.feeding = True
    session.run_for(GPS_SETTLE_S)
    set_mode(session.conn, "AUTO")
    session.arm()
    session.run_until(lambda p: session.mission_seq >= FIRST_ROUTE_WAYPOINT_SEQ and p.height_m > AIRBORNE_HEIGHT_M,
                      COAST_TIMEOUT_S)
    if use_vision:
        switch_to_vision(session)
        set_params(session.conn, {SIM_REAL_GPS_ENABLE: 0})
    session.run_for(CRUISE_SETTLE_S)
    if use_vision:
        session.feeding = False
        stop_s = session.last_fix_s
    else:
        set_params(session.conn, {SIM_REAL_GPS_ENABLE: 0})
        stop_s = session.sim_time_s
    first = len(session.track)
    session.run_until(lambda p: p.mode != "AUTO" or p.sim_time_s - stop_s >= observe_s, observe_s + COAST_TIMEOUT_S)
    return coast_result(session, stop_s, first)


def coast_result(session: FlightSession, stop_s: float, first: int) -> CoastResult:
    after = session.track[first:]
    lost = next((p.sim_time_s - stop_s for p in after if not p.ekf_flags & EKF_FLAG_POS_HORIZ_ABS), None)
    failsafe = after[-1].sim_time_s - stop_s if after[-1].mode != "AUTO" else None
    errors = [(p.sim_time_s - stop_s, haversine_m(p.truth, p.estimate)) for p in after]
    return CoastResult(stop_s, lost, failsafe, after[-1].mode, errors,
                       [(t - stop_s, text) for t, text in session.texts if t >= stop_s])


def run_coast_test(config: SitlConfig, items: list[MissionItem], use_vision: bool, ekf_action: int,
                   observe_s: float) -> CoastResult:
    """A fresh SITL flying `items`; `ekf_action` 0 reports the EKF failsafe only, so drift can be watched."""
    with start_sitl(config) as url:
        conn = connect(url, CONNECT_TIMEOUT_S)
        session = FlightSession(conn)
        session.request_streams()
        set_params(conn, {AUTO_OPTIONS_PARAM: AUTO_OPTIONS_ARM_AND_TAKEOFF, EKF_ACTION_PARAM: ekf_action})
        vehicle_home(conn, HOME_TIMEOUT_S)
        upload_mission(conn, items)
        return measure_coast(session, use_vision, observe_s)
