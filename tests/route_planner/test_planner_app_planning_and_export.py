from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from planner_fakes import START, BrokenNetwork
from tools.route_planner.config import AppConfig
from tools.route_planner.export import (PARAMS_NAME, WAYPOINTS_NAME, ExportRequest, UploadReport, export_route)
from tools.route_planner.planning_job import ELEVATION_STAGE, OVERLAY_STAGE, PlanOutcome, PlannerEngine
from uavloc.ardupilot.link import HOME_WARNING_M
from uavloc.ardupilot.mission import MissionPlan, mission_items, read_waypoints
from uavloc.ardupilot.params import read_param_file, vehicle_params
from uavloc.ardupilot.scripts import SWITCH_SCRIPT_NAME
from uavloc.geo import LatLon, offset_latlon
from uavloc.onboard.package import load_package
from uavloc.planning.config import MissionInputs

FIXED_TIME = datetime(2026, 11, 2, 12, 0, tzinfo=timezone.utc)


def test_plan_reports_stages_and_renders_layers(engine: PlannerEngine, inputs: MissionInputs) -> None:
    stages: list[str] = []
    outcome = engine.plan(inputs, lambda stage, done, total: stages.append(stage))
    assert {ELEVATION_STAGE, OVERLAY_STAGE} <= set(stages)
    assert len(outcome.elevation) == len(outcome.result.routes)
    assert all(check is not None for check in outcome.elevation)
    assert outcome.class_overlay.leaflet_bounds[0][0] < outcome.class_overlay.leaflet_bounds[1][0]
    assert outcome.blind_overlay.data_url.startswith("data:image/png")


def test_failed_elevation_does_not_fail_the_plan(app_config: AppConfig, inputs: MissionInputs) -> None:
    outcome = PlannerEngine(app_config, elevation_source=BrokenNetwork()).plan(inputs)
    assert outcome.elevation == (None,) * len(outcome.result.routes)
    assert "no network" in outcome.elevation_error


def test_area_too_large_raises(engine: PlannerEngine) -> None:
    far = MissionInputs(START, LatLon(START.lat + 0.5, START.lon + 0.5))
    with pytest.raises(ValueError, match="exceeds"):
        engine.plan(far)


def export_request(outcome: PlanOutcome, app_config: AppConfig) -> ExportRequest:
    return ExportRequest(outcome.result, outcome.result.routes[-1], outcome.inputs, outcome.config,
                         app_config.tile_source_key, app_config.tile_cache, app_config.drone_onnx)


def test_export_writes_every_file_and_a_loadable_package(outcome: PlanOutcome, app_config: AppConfig,
                                                         tmp_path: Path) -> None:
    request = export_request(outcome, app_config)
    files = export_route(tmp_path, request, FIXED_TIME)
    assert all(path.is_file() for path in files.paths())
    assert files.folder.parent == tmp_path
    items = read_waypoints(files.waypoints)
    assert len(items) == len(mission_items(MissionPlan.from_route(request.route, request.inputs)))
    assert read_param_file(files.params) == pytest.approx(vehicle_params())
    assert files.lua_script.name == SWITCH_SCRIPT_NAME
    with zipfile.ZipFile(files.package_zip) as archive:
        names = archive.namelist()
    assert any(name.endswith("manifest.json") for name in names)
    assert any(name.endswith(f"ardupilot/{WAYPOINTS_NAME}") for name in names)
    assert any(name.endswith(f"ardupilot/{PARAMS_NAME}") for name in names)
    package = load_package(files.package_zip)
    assert package.manifest["route"]["weight"] == request.route.weight
    assert json.dumps(package.manifest)


def test_exporting_twice_at_the_same_second_refuses_to_overwrite(outcome: PlanOutcome, app_config: AppConfig,
                                                                 tmp_path: Path) -> None:
    request = export_request(outcome, app_config)
    export_route(tmp_path, request, FIXED_TIME)
    with pytest.raises(FileExistsError):
        export_route(tmp_path, request, FIXED_TIME)


def test_upload_report_warns_beyond_the_home_limit() -> None:
    near = UploadReport(10, START, 3.0, ())
    far = UploadReport(10, offset_latlon(START, HOME_WARNING_M + 5, 0), HOME_WARNING_M + 5, ())
    unknown = UploadReport(10, None, None, ("item 3: position moved",))
    assert not near.home_far_from_start and "read back identical" in near.summary()
    assert far.home_far_from_start and "WARNING" in far.summary()
    assert not unknown.home_far_from_start and "no home" in unknown.summary()
    assert "item 3" in unknown.summary()
