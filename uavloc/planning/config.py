"""What a mission asks for, and the planner's tunables."""

from __future__ import annotations

from dataclasses import dataclass, field

from uavloc.geo import LatLon
from uavloc.matching.correlator import CorrelatorConfig
from uavloc.simulation.ins import InsConfig

DEFAULT_ALTITUDE_M = 40.0          # where the matcher was validated (2025 flight, simulator)
DEFAULT_SPEED_MPS = 5.0            # docs/ardupilot-integration.md: <= 5 m/s for the first flights on our fixes
DEFAULT_MAX_FLIGHT_TIME_S = 20 * 60.0

# The onboard matcher's settings, from the 007 baseline (007.1: weighted NCC, rect window x1.7, 3% satellite
# border crop). Temperature and gate thresholds sit inside 007's per-fold tuned range (T 0.25-2, s_max 2-5 m,
# c_min 0.01); they set what "a fix is likely" means for the planner.
PLANNING_MATCHER = CorrelatorConfig(
    search_scale=1.7, fov_w_deg=60.0, sat_feature_crop_pct=0.03, matching="weighted-ncc",
    class_weights=(0.0, 1.0, 1.0, 0.5, 0.0, 0.5), temperature=0.5)
BLIND_S_MAX_M = 5.0
BLIND_C_MIN = 0.01


@dataclass(frozen=True)
class MissionInputs:
    start: LatLon
    end: LatLon
    altitude_m: float = DEFAULT_ALTITUDE_M      # relative to takeoff
    speed_mps: float = DEFAULT_SPEED_MPS
    max_flight_time_s: float = DEFAULT_MAX_FLIGHT_TIME_S
    hover_s: float = 0.0                        # at the end point, before turning back


@dataclass(frozen=True)
class PlannerConfig:
    matcher: CorrelatorConfig = PLANNING_MATCHER
    match_gsd_m: float = 0.385                  # the matcher's common grid (007 MATCH_GSD_M)
    zoom: int = 18                              # Esri z18, ~0.38 m/px: the scale the satellite segmenter saw
    area_margin_m: float = 500.0                # room for detours around the start-end box
    max_area_km2: float = 25.0
    sample_spacing_m: float = 25.0              # localizability grid
    blind_s_max_m: float = BLIND_S_MAX_M        # posterior long-axis std above which a fix is unlikely
    blind_c_min: float = BLIND_C_MIN            # weighted-class coverage below which a fix is unlikely
    heuristic_min_coverage: float = 0.1         # class-mix fallback: weighted-class share needed in the footprint
    cell_m: float = 10.0                        # A* grid
    route_weights: tuple[float, ...] = (0.0, 2.0, 5.0, 20.0)   # blind-cell penalty, shortest first
    blocked_classes: tuple[str, ...] = ()
    blocked_share: float = 0.5                  # a cell is impassable when a blocked class covers this share
    straighten_tolerance: float = 0.05          # a shortcut may cost this fraction more than the cell path
    duplicate_distance_m: float = 30.0          # routes closer than this on average count as one
    drift_sigmas: float = 2.0                   # blind limit: drift at this many sigma must fit the search window
    # IMU-only coasting before ArduCopter declares position lost and lands: 7.5-8.1 s in SITL (no wind),
    # builds/ardupilot-mission/results.json; a real airframe will coast shorter
    ekf_coast_s: float | None = 8.0
    ins: InsConfig = field(default_factory=InsConfig)
    flight_reserve: float = 0.2                 # share of max flight time kept back
