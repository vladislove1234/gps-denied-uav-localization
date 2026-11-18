"""One planning run as the app performs it: plan, check elevation per route, render the map layers. No Qt."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from tools.route_planner.config import AppConfig
from tools.route_planner.overlays import MapOverlay, blind_overlay, class_overlay
from tools.route_planner.tile_sources import tile_source
from uavloc.features.onnx_segmenter import OnnxSegmenter
from uavloc.planning.config import MissionInputs, PlannerConfig
from uavloc.planning.elevation import ElevationCheck, ElevationSource, OpenMeteoElevation, check_elevation
from uavloc.planning.localizability import SelfSimilarity
from uavloc.planning.planner import PlanningResult, plan

ELEVATION_STAGE = "elevation"
OVERLAY_STAGE = "map layers"
ProgressCallback = Callable[[str, int, int], None]


@dataclass(frozen=True)
class PlanOutcome:
    inputs: MissionInputs
    config: PlannerConfig
    result: PlanningResult
    elevation: tuple[ElevationCheck | None, ...]    # per route; None where the check could not run
    elevation_error: str | None
    class_overlay: MapOverlay
    blind_overlay: MapOverlay


class PlannerEngine:
    """Loads the satellite segmenter once and plans as many missions as asked."""

    def __init__(self, app_config: AppConfig, planner_config: PlannerConfig | None = None,
                 elevation_source: ElevationSource | None = None) -> None:
        self.app_config = app_config
        self.config = planner_config or PlannerConfig()
        self.elevation_source = elevation_source or OpenMeteoElevation()
        self._segmenter: OnnxSegmenter | None = None

    @property
    def segmenter(self) -> OnnxSegmenter:
        if self._segmenter is None:
            self._segmenter = OnnxSegmenter(self.app_config.satellite_onnx)
        return self._segmenter

    def check_routes(self, result: PlanningResult, altitude_m: float,
                     progress: ProgressCallback | None) -> tuple[tuple[ElevationCheck | None, ...], str | None]:
        """Elevation is advisory and needs the network, so a failure becomes a message, not a failed plan."""
        checks: list[ElevationCheck | None] = []
        try:
            for done, route in enumerate(result.routes):
                if progress:
                    progress(ELEVATION_STAGE, done, len(result.routes))
                checks.append(check_elevation(route.waypoints, altitude_m, self.elevation_source))
        except (OSError, ValueError, KeyError) as error:
            return (None,) * len(result.routes), f"{type(error).__name__}: {error}"
        return tuple(checks), None

    def plan(self, inputs: MissionInputs, progress: ProgressCallback | None = None) -> PlanOutcome:
        config = self.config
        localizability = SelfSimilarity(config.matcher, config.match_gsd_m, config.blind_s_max_m, config.blind_c_min)
        result = plan(inputs, config, tile_source(self.app_config.tile_source_key).source, self.segmenter,
                      localizability, self.app_config.tile_cache, self.app_config.planner_cache, progress)
        checks, error = self.check_routes(result, inputs.altitude_m, progress)
        if progress:
            progress(OVERLAY_STAGE, 0, 1)
        outcome = PlanOutcome(inputs, config, result, checks, error,
                              class_overlay(result.class_map), blind_overlay(result.blind_map))
        if progress:
            progress(OVERLAY_STAGE, 1, 1)
        return outcome
