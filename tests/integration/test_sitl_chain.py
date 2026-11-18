"""Smoke test of the v1 chain: package -> SITL -> separate runner process -> mission out and back on our fixes."""

from __future__ import annotations

import os
from pathlib import Path

from chain_support import requires_sitl
from uavloc.ardupilot.chain import ChainScenario, RunnerConfig, run_chain
from uavloc.ardupilot.chain_analysis import REASON_SENT, find_fix_log, read_fix_log, summarise_flight
from uavloc.ardupilot.mission import read_waypoints
from uavloc.ardupilot.params import vehicle_params
from uavloc.ardupilot.sitl import SitlConfig, default_ardupilot_dir, telem2_url
from uavloc.onboard.package import load_package

SPEEDUP = float(os.environ.get("UAVLOC_SITL_SPEEDUP", "16"))
HOME_ALT_M = 300.0
HEADING_DEG = 90.0
RUNNER_PERIOD_S = 0.5
NOISE_M = 6.0
TIMEOUT_SIM_S = 900.0
LANDING_TOLERANCE_M = 20.0
FIX_ERROR_SIGMAS = 3.0                # every fix error is a draw from a 6 m per-axis normal


@requires_sitl
def test_chain_flies_out_and_back_on_runner_fixes(free_sitl_port, package_zip: Path, tmp_path: Path):
    package = load_package(package_zip)
    start, end = package.waypoints[0], package.waypoints[len(package.waypoints) // 2]
    sitl = SitlConfig(default_ardupilot_dir(), start, HOME_ALT_M, HEADING_DEG, SPEEDUP, vehicle_params(),
                      work_dir=tmp_path / "sitl")
    runner = RunnerConfig(package_zip, telem2_url(), tmp_path / "runner-logs", SPEEDUP, RUNNER_PERIOD_S, NOISE_M,
                          no_blind=True)
    scenario = ChainScenario(sitl, read_waypoints(package.root / "ardupilot" / "mission.waypoints"), runner,
                             ekf_action=1, timeout_sim_s=TIMEOUT_SIM_S)

    flight = run_chain(scenario, tmp_path / "runner.log")
    fixes = read_fix_log(find_fix_log(runner.log_dir))
    summary = summarise_flight(flight, fixes, start, end)

    assert flight.completed
    assert flight.runner_exit_code == 0
    assert flight.switch_s is not None and flight.gps1_off_s is not None
    assert not flight.mission_differences
    assert summary["fixes_by_reason"][REASON_SENT] > 0
    assert summary["closest_to_end_m"] < LANDING_TOLERANCE_M
    assert summary["landed_offset_from_start_m"] < LANDING_TOLERANCE_M
    assert summary["active_error_auto_only"]["max_m"] < FIX_ERROR_SIGMAS * NOISE_M * 2
    assert {mode for _, mode in summary["mode_changes"]} <= {"ALT_HOLD", "AUTO", "LAND"}
