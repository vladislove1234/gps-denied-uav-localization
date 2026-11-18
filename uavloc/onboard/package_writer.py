"""Writing a mission package on the laptop: corridor crop, GeoTIFFs, manifest, zip."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from uavloc.data.satellite import GeoImage
from uavloc.geo import GeoBounds, LatLon
from uavloc.onboard.corridor import CorridorWidth, corridor_mask, corridor_width, point_at_distance
from uavloc.onboard.package import (ARDUPILOT_DIR, ARDUPILOT_FILE_NAMES, BLIND_PATH, CAMERA_PATH, CLASSES_PATH,
                                    FORMAT_VERSION, LOCALIZER_PATH, MANIFEST_NAME, MODEL_PATH, PACKAGE_PREFIX,
                                    REQUIRED_FILES, SATELLITE_PATH, sha256_of)
from uavloc.onboard.raster import crop_to_bounds, write_geotiff
from uavloc.planning.config import MissionInputs, PlannerConfig
from uavloc.planning.planner import PlanningResult, config_record
from uavloc.planning.routes import Route
from uavloc.simulation.camera import CameraConfig

JPEG_QUALITY = 90
MISSION_ID_HASH_CHARS = 6
UTC_STAMP_FORMAT = "%Y%m%dT%H%M%SZ"
OUTSIDE_CORRIDOR_BLIND = 1.0


@dataclass(frozen=True)
class PackageRequest:
    """Everything `write_package` needs besides the output directory."""

    result: PlanningResult
    route: Route
    inputs: MissionInputs
    config: PlannerConfig
    satellite: GeoImage                        # RGB, any extent that covers the route's corridor
    drone_model: Path
    camera: CameraConfig
    ardupilot_files: dict[str, Path]           # keys are `ARDUPILOT_FILE_NAMES`
    mission_id: str | None = None              # default: UTC stamp + a hash of the waypoints
    created_utc: datetime | None = None


def out_and_back(route: Route) -> list[LatLon]:
    return [*route.waypoints, *reversed(route.waypoints[:-1])]


def make_mission_id(waypoints: list[LatLon], created: datetime) -> str:
    digest = hashlib.sha256(json.dumps(waypoints).encode()).hexdigest()[:MISSION_ID_HASH_CHARS]
    return f"{created.strftime(UTC_STAMP_FORMAT)}-{digest}"


def stretch_records(route: Route) -> list[dict]:
    return [{"start_m": s.start_m, "end_m": s.end_m, "length_m": s.length_m,
             "start": list(point_at_distance(route.waypoints, s.start_m)),
             "end": list(point_at_distance(route.waypoints, s.end_m))} for s in route.blind_stretches]


def raster_record(image: GeoImage) -> dict:
    return {"top_left": list(image.top_left), "bottom_right": list(image.bottom_right),
            "shape": list(image.pixels.shape)}


def corridor_rasters(request: PackageRequest, width: CorridorWidth) -> tuple[GeoImage, GeoImage, GeoImage]:
    """Class map, blind map and satellite cropped to the route's corridor; pixels outside the strip are blanked
    (classes 0, blind 1, satellite black) so the file compresses to the strip and the Pi sees no map there."""
    waypoints = request.route.waypoints
    bounds = GeoBounds.around(list(waypoints), width.half_width_m)
    classes = crop_to_bounds(request.result.class_map, bounds)
    blind = crop_to_bounds(request.result.blind_map, bounds)
    satellite = crop_to_bounds(request.satellite, bounds)
    classes_pixels, satellite_pixels = classes.pixels.copy(), satellite.pixels.copy()
    blind_pixels = blind.pixels.copy()
    classes_pixels[~corridor_mask(classes, waypoints, width.half_width_m)] = 0
    satellite_pixels[~corridor_mask(satellite, waypoints, width.half_width_m)] = 0
    blind_pixels[~corridor_mask(blind, waypoints, width.half_width_m)] = OUTSIDE_CORRIDOR_BLIND
    return classes.with_pixels(classes_pixels), blind.with_pixels(blind_pixels), satellite.with_pixels(satellite_pixels)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def copy_into(root: Path, relative: str, source: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"{source} for {relative} does not exist")
    (root / relative).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, root / relative)


def build_manifest(request: PackageRequest, width: CorridorWidth, mission_id: str, created: datetime,
                   rasters: tuple[GeoImage, GeoImage, GeoImage], hashes: dict[str, str]) -> dict:
    route, classes, blind, satellite = request.route, *rasters
    return {
        "format_version": FORMAT_VERSION,
        "mission_id": mission_id,
        "created_utc": created.isoformat(),
        "planner": config_record(request.inputs, request.config),
        "route": {
            "weight": route.weight, "waypoints_out_and_back": [list(p) for p in out_and_back(route)],
            "length_m": route.length_m, "flight_time_s": route.flight_time_s, "blind_share": route.blind_share,
            "blind_limit_m": route.blind_limit_m, "longest_blind_m": route.longest_blind_m,
            "exceeds_blind_limit": route.exceeds_blind_limit, "blind_stretches": stretch_records(route)},
        "corridor": {**asdict(width), "half_width_m": width.half_width_m},
        "reference": {"classes": raster_record(classes), "blind": raster_record(blind),
                      "satellite": raster_record(satellite)},
        "files": hashes,
    }


def write_package(out_dir: Path, request: PackageRequest) -> Path:
    """Write `out_dir/mission-<id>/` and `out_dir/mission-<id>.zip`; returns the zip."""
    missing = set(ARDUPILOT_FILE_NAMES) - set(request.ardupilot_files)
    if missing:
        raise ValueError(f"ardupilot_files lacks {sorted(missing)}")
    created = request.created_utc or datetime.now(timezone.utc)
    mission_id = request.mission_id or make_mission_id(out_and_back(request.route), created)
    root = out_dir / f"{PACKAGE_PREFIX}{mission_id}"
    if root.exists():
        raise FileExistsError(f"{root} already exists")
    width = corridor_width(request.result.blind_limit_m, request.inputs, request.config)
    rasters = corridor_rasters(request, width)
    classes, blind, satellite = rasters
    write_geotiff(root / CLASSES_PATH, classes)
    write_geotiff(root / BLIND_PATH, blind)
    (root / SATELLITE_PATH).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(satellite.pixels).save(root / SATELLITE_PATH, quality=JPEG_QUALITY)
    copy_into(root, MODEL_PATH, request.drone_model)
    write_json(root / CAMERA_PATH, asdict(request.camera))
    write_json(root / LOCALIZER_PATH, {
        "matcher": asdict(request.config.matcher), "match_gsd_m": request.config.match_gsd_m,
        "blind_s_max_m": request.config.blind_s_max_m, "blind_c_min": request.config.blind_c_min})
    for name in ARDUPILOT_FILE_NAMES:
        copy_into(root, f"{ARDUPILOT_DIR}/{name}", request.ardupilot_files[name])
    hashes = {relative: sha256_of(root / relative) for relative in REQUIRED_FILES}
    write_json(root / MANIFEST_NAME, build_manifest(request, width, mission_id, created, rasters, hashes))
    return zip_folder(root)


def zip_folder(root: Path) -> Path:
    archive = root.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                zf.write(path, str(Path(root.name) / path.relative_to(root)))
    return archive
