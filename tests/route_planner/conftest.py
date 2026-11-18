"""One real Birky plan for the whole suite: its tiles and class blocks are cached, so it takes seconds."""

from __future__ import annotations

import pytest

from planner_fakes import END, START, FlatGround
from tools.route_planner.config import AppConfig
from tools.route_planner.planning_job import PlanOutcome, PlannerEngine
from uavloc.planning.config import MissionInputs
from uavloc.track import REPO_ROOT


@pytest.fixture(scope="session")
def app_config() -> AppConfig:
    config = AppConfig.defaults(REPO_ROOT)
    needed = [config.satellite_onnx, config.drone_onnx, config.tile_cache / "esri-current"]
    if not all(path.exists() for path in needed):
        pytest.skip("the cached Birky case (models and tiles) is not on this machine")
    return config


@pytest.fixture(scope="session")
def engine(app_config: AppConfig) -> PlannerEngine:
    return PlannerEngine(app_config, elevation_source=FlatGround())


@pytest.fixture(scope="session")
def inputs() -> MissionInputs:
    return MissionInputs(START, END)


@pytest.fixture(scope="session")
def outcome(engine: PlannerEngine, inputs: MissionInputs) -> PlanOutcome:
    return engine.plan(inputs)
