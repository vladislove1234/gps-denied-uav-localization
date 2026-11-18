"""One planning pass: area → class map → localizability → cost grid → route alternatives."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from uavloc.data.satellite import GeoImage
from uavloc.data.tiles import TileSource
from uavloc.geo import GeoBounds, footprint_m
from uavloc.matching.correlator import half_range_for_footprint_m
from uavloc.planning.classmap import ProgressHook, Segmenter, corridor_class_map
from uavloc.planning.config import MissionInputs, PlannerConfig
from uavloc.planning.drift import max_blind_distance_m
from uavloc.planning.grid import CostGrid, cost_grid
from uavloc.planning.localizability import LocalizabilityModel
from uavloc.planning.routes import Route, plan_routes

M2_PER_KM2 = 1e6
CACHE_KEY_HEX_CHARS = 16


@dataclass(frozen=True)
class PlanningResult:
    area: GeoBounds
    class_map: GeoImage          # `(H, W, C)` uint8 probabilities x 255
    blind_map: GeoImage          # `(rows, cols, 1)` float32 over `area`, 1 = blind
    grid: CostGrid
    blind_limit_m: float
    routes: list[Route]


def planning_area(inputs: MissionInputs, config: PlannerConfig) -> GeoBounds:
    area = GeoBounds.around([inputs.start, inputs.end], config.area_margin_m)
    width_m, height_m = area.size_m
    if width_m * height_m / M2_PER_KM2 > config.max_area_km2:
        raise ValueError(f"planning area {width_m * height_m / M2_PER_KM2:.1f} km² exceeds "
                         f"{config.max_area_km2:.0f} km²: move the points closer or raise max_area_km2")
    return area


def blind_limit_m(inputs: MissionInputs, config: PlannerConfig) -> float:
    """Drift must stay inside the search window's short half-range, the axis that runs out first."""
    footprint = footprint_m(inputs.altitude_m, config.matcher.fov_w_deg, config.matcher.aspect_w_over_h)
    half_range_m = min(half_range_for_footprint_m(config.matcher, footprint))
    return max_blind_distance_m(inputs.speed_mps, half_range_m, config.ins, config.drift_sigmas, config.ekf_coast_s)


def cached_blind_map(model: LocalizabilityModel, class_map: GeoImage, area: GeoBounds, altitude_m: float,
                     config: PlannerConfig, cache_dir: Path, progress: ProgressHook | None) -> GeoImage:
    """The localizability map is the slow step; it is cached per area, class map, altitude and model settings."""
    key_fields = {"model": type(model).__name__, "settings": repr(vars(model)), "altitude_m": altitude_m,
                  "spacing_m": config.sample_spacing_m, "area": area, "top_left": class_map.top_left,
                  "bottom_right": class_map.bottom_right, "shape": class_map.pixels.shape,
                  "class_map_sum": int(class_map.pixels[::7, ::7].sum(dtype=np.int64))}
    key = hashlib.sha256(json.dumps(key_fields, sort_keys=True, default=str).encode()).hexdigest()
    path = cache_dir / f"{key[:CACHE_KEY_HEX_CHARS]}.npy"
    if path.exists():
        return GeoImage(np.load(path), area.top_left, area.bottom_right)
    blind = model.blind_map(class_map, area, altitude_m, config.sample_spacing_m, progress)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, blind.pixels)
    return blind


def plan(inputs: MissionInputs, config: PlannerConfig, source: TileSource, segmenter: Segmenter,
         localizability: LocalizabilityModel, tile_cache: Path, planner_cache: Path,
         progress: ProgressHook | None = None) -> PlanningResult:
    area = planning_area(inputs, config)
    class_map = corridor_class_map(area, config.zoom, source, segmenter, tile_cache,
                                   planner_cache / "class-maps", progress)
    blind_map = cached_blind_map(localizability, class_map, area, inputs.altitude_m, config,
                                 planner_cache / "localizability", progress)
    grid = cost_grid(blind_map, class_map, config.cell_m, config.blocked_classes, config.blocked_share)
    limit_m = blind_limit_m(inputs, config)
    return PlanningResult(area, class_map, blind_map, grid, limit_m, plan_routes(grid, inputs, config, limit_m))


def config_record(inputs: MissionInputs, config: PlannerConfig) -> dict:
    """Everything that decided a plan, JSON-ready, for the mission manifest."""
    return {"inputs": asdict(inputs), "planner": asdict(config)}
