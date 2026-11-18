"""Export a planned route for ArduPilot and the Pi, and send it to a vehicle. No Qt, so it can be tested."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from tools.route_planner.tile_sources import tile_source
from uavloc.ardupilot.link import HOME_WARNING_M, connect, download_mission, home_offset_m, set_params, \
    upload_mission, vehicle_home
from uavloc.ardupilot.mission import MissionPlan, mission_differences, mission_items, write_waypoints
from uavloc.ardupilot.params import vehicle_params, write_param_file
from uavloc.ardupilot.scripts import SWITCH_SCRIPT_NAME, lua_script_path
from uavloc.data.tiles import geo_mosaic
from uavloc.geo import LatLon
from uavloc.onboard.package_writer import PackageRequest, write_package
from uavloc.planning.config import MissionInputs, PlannerConfig
from uavloc.planning.planner import PlanningResult
from uavloc.planning.routes import Route
from uavloc.simulation.camera import CameraConfig

WAYPOINTS_NAME = "mission.waypoints"
PARAMS_NAME = "uavloc.param"
FOLDER_STAMP_FORMAT = "%Y%m%dT%H%M%SZ"
CONNECT_TIMEOUT_S = 10.0
HOME_TIMEOUT_S = 40.0      # SITL sets its home about 20 s after the first connection; a real FC with a fix answers at once


@dataclass(frozen=True)
class ExportRequest:
    result: PlanningResult
    route: Route
    inputs: MissionInputs
    config: PlannerConfig
    tile_source_key: str
    tile_cache: Path
    drone_model: Path


@dataclass(frozen=True)
class ExportedFiles:
    folder: Path
    waypoints: Path
    params: Path
    lua_script: Path
    package_zip: Path

    def paths(self) -> list[Path]:
        return [self.waypoints, self.params, self.lua_script, self.package_zip]


def export_folder(out_dir: Path, route: Route, created: datetime) -> Path:
    return out_dir / f"route-w{route.weight:g}-{created.strftime(FOLDER_STAMP_FORMAT)}"


def mission_id(route: Route, created: datetime) -> str:
    """Named here rather than left to `write_package`, so the package folder matches the export folder."""
    return f"{created.strftime(FOLDER_STAMP_FORMAT)}-w{route.weight:g}"


def export_route(out_dir: Path, request: ExportRequest, created: datetime | None = None) -> ExportedFiles:
    """Write the three ArduPilot files and the Pi mission package into a new folder under `out_dir`."""
    created = created or datetime.now(timezone.utc)
    folder = export_folder(out_dir, request.route, created)
    folder.mkdir(parents=True, exist_ok=False)
    plan = MissionPlan.from_route(request.route, request.inputs)
    waypoints, params, lua = folder / WAYPOINTS_NAME, folder / PARAMS_NAME, folder / SWITCH_SCRIPT_NAME
    write_waypoints(waypoints, mission_items(plan))
    write_param_file(params, vehicle_params())
    lua.write_bytes(lua_script_path().read_bytes())
    satellite = geo_mosaic(tile_source(request.tile_source_key).source, request.result.area, request.config.zoom,
                           request.tile_cache)
    package = write_package(folder, PackageRequest(
        request.result, request.route, request.inputs, request.config, satellite, request.drone_model,
        CameraConfig(fov_w_deg=request.config.matcher.fov_w_deg),
        {WAYPOINTS_NAME: waypoints, PARAMS_NAME: params, SWITCH_SCRIPT_NAME: lua},
        mission_id=mission_id(request.route, created), created_utc=created))
    return ExportedFiles(folder, waypoints, params, lua, package)


@dataclass(frozen=True)
class UploadReport:
    items_sent: int
    home: LatLon | None
    home_offset_m: float | None
    differences: tuple[str, ...]          # what the vehicle stored differently from what was sent

    @property
    def home_far_from_start(self) -> bool:
        return self.home_offset_m is not None and self.home_offset_m > HOME_WARNING_M

    def summary(self) -> str:
        lines = [f"{self.items_sent} mission items uploaded"]
        if self.home is None:
            lines.append("vehicle reported no home yet (no GPS fix?): check the start before arming")
        elif self.home_far_from_start:
            lines.append(f"WARNING: vehicle home is {self.home_offset_m:.0f} m from the planned start "
                         f"(limit {HOME_WARNING_M:.0f} m)")
        else:
            lines.append(f"vehicle home is {self.home_offset_m:.1f} m from the planned start")
        lines += [f"vehicle stored it differently: {d}" for d in self.differences] or ["read back identical"]
        return "\n".join(lines)


def upload_route(connection: str, route: Route, inputs: MissionInputs) -> UploadReport:
    """Upload only the mission (parameters are loaded once, separately), read it back and compare."""
    items = mission_items(MissionPlan.from_route(route, inputs))
    conn = connect(connection, CONNECT_TIMEOUT_S)
    try:
        try:
            home = vehicle_home(conn, HOME_TIMEOUT_S)
        except TimeoutError:
            home = None
        upload_mission(conn, items)
        differences = mission_differences(items, download_mission(conn))
    finally:
        conn.close()
    offset = home_offset_m(home, route.waypoints[0]) if home else None
    return UploadReport(len(items), home, offset, tuple(differences))


def write_vehicle_params(connection: str) -> int:
    """Set every parameter of `vehicle_params()` on the vehicle; returns how many."""
    params = vehicle_params()
    conn = connect(connection, CONNECT_TIMEOUT_S)
    try:
        set_params(conn, params)
    finally:
        conn.close()
    return len(params)
