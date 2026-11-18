"""The v1 chain in SITL: the onboard runner as a separate process feeds GPS 2 while the vehicle flies the mission."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from uavloc.ardupilot.flight import ArmingRefused, EKF_FLAG_POS_HORIZ_ABS, FlightSession, TrackPoint
from uavloc.ardupilot.link import connect, download_mission, set_params, upload_mission, vehicle_home
from uavloc.ardupilot.mission import MissionItem, mission_differences
from uavloc.ardupilot.scenarios import (AIRBORNE_HEIGHT_M, AUTO_OPTIONS_ARM_AND_TAKEOFF, AUTO_OPTIONS_PARAM,
                                        CONNECT_TIMEOUT_S, EKF_ACTION_PARAM, FIRST_ROUTE_WAYPOINT_SEQ, GPS_SETTLE_S,
                                        HOME_TIMEOUT_S, SIM_REAL_GPS_ENABLE, switch_to_vision)
from uavloc.ardupilot.sitl import SitlConfig, start_sitl
from uavloc.ardupilot.vehicle import set_mode
from uavloc.geo import LatLon, haversine_m

RUNNER_MODULE = "uavloc.onboard"
RUNNER_STOP_TIMEOUT_S = 10.0
RUNNER_LOG_NAME = "runner.log"
ARM_MODE = "ALT_HOLD"                # Copter checks the GPS only for modes that need a position
FLY_MODE = "AUTO"
AGREE_SAMPLES = 5                    # GPS2_RAW messages (2 Hz) that must sit close to the truth before the switch
AGREE_TOLERANCE_M = 15.0
AGREE_TIMEOUT_S = 60.0
LANDED_HEIGHT_M = 0.5                # without a position the vehicle may touch down and stay armed
POST_LANDING_S = 5.0
MODE_REFUSAL_WAIT_S = 3.0
EKF_ALIGN_TIMEOUT_S = 120.0          # AUTO is refused until the EKF has a position, even if arming in ALT_HOLD worked
SWITCH_SETTLE_S = 10.0               # fly on our fixes with the real GPS still on, before it is switched off


@dataclass(frozen=True)
class RunnerConfig:
    """How the onboard runner is started, as on the Pi but with `--stub` and the simulator's clock."""

    package: Path
    connection: str
    log_dir: Path
    speedup: float
    period_s: float
    noise_m: float
    seed: int = 0
    no_blind: bool = False


@dataclass(frozen=True)
class ChainScenario:
    """One flight of the chain: a fresh SITL, the mission, and the runner that feeds GPS 2."""

    sitl: SitlConfig
    items: list[MissionItem]
    runner: RunnerConfig
    ekf_action: int                        # FS_EKF_ACTION: 1 lands on a lost position, 0 only reports it
    timeout_sim_s: float


@dataclass(frozen=True)
class ChainFlight:
    track: list[TrackPoint]
    texts: list[tuple[float, str]]
    events: list[tuple[float, str]]        # (simulated s, what happened) in the order it happened
    completed: bool                        # touched down before the timeout
    switch_s: float | None                 # simulated time the Lua switch confirmed "Vision GPS: ON"
    gps1_off_s: float | None               # simulated time the simulated real GPS was switched off
    runner_exit_code: int | None = None
    home: LatLon | None = None             # as the vehicle reports it before the flight
    mission_differences: list[str] = field(default_factory=list)

    @property
    def landed(self) -> LatLon:
        return self.track[-1].truth


def runner_command(config: RunnerConfig) -> list[str]:
    command = [sys.executable, "-m", RUNNER_MODULE, "--package", str(config.package), "--connection",
               config.connection, "--stub", "--log-dir", str(config.log_dir), "--speedup", f"{config.speedup:g}",
               "--period-s", f"{config.period_s:g}", "--noise-m", f"{config.noise_m:g}", "--seed", str(config.seed)]
    return command + (["--no-blind"] if config.no_blind else [])


@contextmanager
def running_runner(config: RunnerConfig, console_log: Path) -> Iterator[subprocess.Popen]:
    """Start the runner as its own process; stop it with SIGTERM on exit, as systemd would."""
    config.log_dir.mkdir(parents=True, exist_ok=True)
    with open(console_log, "wb") as log:
        process = subprocess.Popen(runner_command(config), stdout=log, stderr=subprocess.STDOUT)
        try:
            yield process
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=RUNNER_STOP_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


class FixAgreement:
    """True once the last `samples` GPS 2 fixes the vehicle reports are within `tolerance_m` of the simulator truth."""

    def __init__(self, session: FlightSession, samples: int = AGREE_SAMPLES,
                 tolerance_m: float = AGREE_TOLERANCE_M) -> None:
        self.session = session
        self.samples = samples
        self.tolerance_m = tolerance_m
        self.errors_m: list[float] = []
        self._seen = len(session.gps2_fixes)

    def __call__(self, point: TrackPoint) -> bool:
        fixes = self.session.gps2_fixes
        self.errors_m += [haversine_m(fix, point.truth) for fix in fixes[self._seen:]]
        self._seen = len(fixes)
        recent = self.errors_m[-self.samples:]
        return len(recent) == self.samples and float(np.mean(recent)) <= self.tolerance_m


def enter_mode(session: FlightSession, name: str) -> None:
    """Change mode; on refusal raise with the vehicle's own reasons from STATUSTEXT."""
    try:
        set_mode(session.conn, name)
    except TimeoutError as error:
        session.run_for(MODE_REFUSAL_WAIT_S)
        raise TimeoutError(f"{error}; vehicle said {[text for _, text in session.texts[-5:]]}") from error


def fly_chain(session: FlightSession, timeout_sim_s: float) -> ChainFlight:
    """Shadow takeoff on the simulated GPS, switch to our fixes, drop the simulated GPS, fly the mission out and back.

    The runner feeds GPS 2 on its own port; this session only watches and flips the switch. Arming happens in a mode
    that needs no position, so the arming check does not require GPS 2 at 5 Hz.
    """
    events: list[tuple[float, str]] = []

    def mark(label: str) -> None:
        events.append((session.sim_time_s, label))

    session.run_until(lambda p: bool(p.ekf_flags & EKF_FLAG_POS_HORIZ_ABS), EKF_ALIGN_TIMEOUT_S)
    mark("EKF has an absolute position")
    session.run_for(GPS_SETTLE_S)
    enter_mode(session, ARM_MODE)
    session.arm()
    mark(f"armed in {ARM_MODE}")
    first = len(session.track)
    enter_mode(session, FLY_MODE)
    mark(f"mode {FLY_MODE}")
    session.run_until(lambda p: session.mission_seq >= FIRST_ROUTE_WAYPOINT_SEQ and p.height_m > AIRBORNE_HEIGHT_M,
                      timeout_sim_s)
    mark("airborne on the simulated GPS (shadow)")
    session.run_until(FixAgreement(session), AGREE_TIMEOUT_S)
    mark("our fixes agree with the truth")
    switch_to_vision(session)
    switch_s = session.sim_time_s
    mark("switch to our fixes (GPS_PRIMARY 1, source set 2)")
    session.run_for(SWITCH_SETTLE_S)
    set_params(session.conn, {SIM_REAL_GPS_ENABLE: 0})
    gps1_off_s = session.sim_time_s
    mark("simulated GPS 1 off: flying on our fixes only")
    try:
        session.run_until(lambda p: not p.armed or p.height_m < LANDED_HEIGHT_M, timeout_sim_s)
        mark("touched down")
        session.run_for(POST_LANDING_S)
        completed = True
    except TimeoutError:
        completed = False
        mark("timeout before touching down")
    return ChainFlight(session.track[first:], list(session.texts), events, completed, switch_s, gps1_off_s)


@contextmanager
def prepared_vehicle(scenario: ChainScenario, console_log: Path) -> Iterator[tuple[FlightSession, subprocess.Popen, LatLon, list[str]]]:
    """SITL up, parameters set, mission uploaded and checked, runner started; everything stopped on exit."""
    with start_sitl(scenario.sitl) as url:
        conn = connect(url, CONNECT_TIMEOUT_S)
        session = FlightSession(conn)
        session.request_streams()
        set_params(conn, scenario.sitl.extra_params)
        set_params(conn, {AUTO_OPTIONS_PARAM: AUTO_OPTIONS_ARM_AND_TAKEOFF, EKF_ACTION_PARAM: scenario.ekf_action})
        home = vehicle_home(conn, HOME_TIMEOUT_S)
        upload_mission(conn, scenario.items)
        differences = mission_differences(scenario.items, download_mission(conn))
        with running_runner(scenario.runner, console_log) as process:
            yield session, process, home, differences


def run_chain(scenario: ChainScenario, console_log: Path) -> ChainFlight:
    """The whole chain once: returns the flight; the runner's own log is in `scenario.runner.log_dir`."""
    with prepared_vehicle(scenario, console_log) as (session, process, home, differences):
        flight = fly_chain(session, scenario.timeout_sim_s)
    return replace(flight, runner_exit_code=process.returncode, home=home, mission_differences=differences)


@dataclass(frozen=True)
class ArmingProbe:
    period_s: float
    armed: bool
    texts: list[str]


def probe_arming(scenario: ChainScenario, console_log: Path, wait_sim_s: float) -> ArmingProbe:
    """Try to arm in AUTO, which checks GPS 2, while the runner sends a fix every `runner.period_s`."""
    with prepared_vehicle(scenario, console_log) as (session, _, _, _):
        session.run_until(lambda p: bool(p.ekf_flags & EKF_FLAG_POS_HORIZ_ABS), EKF_ALIGN_TIMEOUT_S)
        session.run_for(GPS_SETTLE_S)
        enter_mode(session, FLY_MODE)
        try:
            session.arm(wait_sim_s)
            armed = True
        except ArmingRefused:
            armed = False
        return ArmingProbe(scenario.runner.period_s, armed, [text for _, text in session.texts])
