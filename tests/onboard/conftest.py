from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from onboard_fakes import synthetic_result, synthetic_route, synthetic_satellite
from uavloc.onboard.package import ARDUPILOT_FILE_NAMES
from uavloc.onboard.package_writer import PackageRequest, write_package
from uavloc.planning.config import MissionInputs, PlannerConfig
from uavloc.simulation.camera import CameraConfig

CREATED = datetime(2026, 11, 2, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def request_for_package(tmp_path: Path) -> PackageRequest:
    model = tmp_path / "drone.onnx"
    model.write_bytes(b"onnx bytes")
    ardupilot = {}
    for name in ARDUPILOT_FILE_NAMES:
        ardupilot[name] = tmp_path / f"src-{name}"
        ardupilot[name].write_text(f"content of {name}")
    route = synthetic_route()
    inputs = MissionInputs(route.waypoints[0], route.waypoints[-1])
    return PackageRequest(synthetic_result(), route, inputs, PlannerConfig(), synthetic_satellite(), model,
                          CameraConfig(), ardupilot, mission_id="test", created_utc=CREATED)


@pytest.fixture
def package_zip(tmp_path: Path, request_for_package: PackageRequest) -> Path:
    return write_package(tmp_path / "out", request_for_package)
