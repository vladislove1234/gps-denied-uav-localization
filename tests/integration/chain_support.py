"""Shared setup of the SITL tests: availability, the synthetic world of the onboard tests, a flyable package."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "onboard"))   # the synthetic world of the onboard tests

from onboard_fakes import synthetic_result, synthetic_route, synthetic_satellite  # noqa: E402
from uavloc.ardupilot.mission import MissionPlan, mission_items, write_waypoints  # noqa: E402
from uavloc.ardupilot.params import vehicle_params, write_param_file  # noqa: E402
from uavloc.ardupilot.scripts import lua_script_path  # noqa: E402
from uavloc.ardupilot.sitl import SITL_BINARY, default_ardupilot_dir  # noqa: E402
from uavloc.onboard.package_writer import PackageRequest, write_package  # noqa: E402
from uavloc.planning.config import MissionInputs, PlannerConfig  # noqa: E402
from uavloc.simulation.camera import CameraConfig  # noqa: E402

CREATED = datetime(2026, 11, 2, 12, 0, tzinfo=timezone.utc)
FLIGHT_SPEED_MPS = 10.0              # a short mission: the synthetic route is 1 km one way


def sitl_available() -> bool:
    return (default_ardupilot_dir() / SITL_BINARY).exists()


requires_sitl = pytest.mark.skipif(not sitl_available(), reason="ArduPilot SITL is not built (see builds/ardupilot-mission)")



def build_package(tmp_path: Path) -> Path:
    route = synthetic_route()
    inputs = MissionInputs(route.waypoints[0], route.waypoints[-1], speed_mps=FLIGHT_SPEED_MPS)
    source = tmp_path / "ardupilot-src"
    source.mkdir()
    write_waypoints(source / "mission.waypoints", mission_items(MissionPlan.from_route(route, inputs)))
    write_param_file(source / "uavloc.param", vehicle_params())
    model = tmp_path / "drone.onnx"
    model.write_bytes(b"onnx bytes")
    files = {"mission.waypoints": source / "mission.waypoints", "uavloc.param": source / "uavloc.param",
             "vision-gps-switch.lua": lua_script_path()}
    request = PackageRequest(synthetic_result(), route, inputs, PlannerConfig(), synthetic_satellite(), model,
                             CameraConfig(), files, mission_id="smoke", created_utc=CREATED)
    return write_package(tmp_path / "packages", request)
