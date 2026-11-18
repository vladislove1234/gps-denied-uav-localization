from __future__ import annotations

import json
import shutil
import zipfile
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from onboard_fakes import ROUTE_EAST_M, ROUTE_NORTH_M, at
from uavloc.onboard.corridor import corridor_width
from uavloc.onboard.package import (CLASSES_PATH, FORMAT_VERSION, MANIFEST_NAME, PackageFormatError,
                                    PackageIntegrityError, load_package, newest_package)
from uavloc.onboard.package_writer import PackageRequest, write_package
from uavloc.planning.drift import drift_sigma_m


def pixel_index(image, point) -> tuple[int, int]:
    x, y = image.to_px(point)
    return int(y), int(x)


def test_round_trip_from_zip(package_zip: Path, request_for_package: PackageRequest) -> None:
    package = load_package(package_zip)
    manifest = package.manifest
    assert manifest["format_version"] == FORMAT_VERSION
    assert package.mission_id == "test"
    assert len(package.waypoints) == 2 * len(request_for_package.route.waypoints) - 1
    assert package.waypoints[0] == package.waypoints[-1] == request_for_package.route.waypoints[0]
    assert package.reference.classes.pixels.dtype == np.uint8
    assert package.reference.classes.pixels.shape[2] == 6
    assert package.reference.blind.pixels.dtype == np.float32
    assert package.model_path.read_bytes() == b"onnx bytes"
    assert package.camera_settings == asdict(request_for_package.camera)
    assert package.localizer_settings["match_gsd_m"] == request_for_package.config.match_gsd_m
    assert package.ardupilot_file("uavloc.param").read_text() == "content of uavloc.param"
    assert manifest["route"]["blind_stretches"][0]["length_m"] == pytest.approx(200.0)


def test_georeference_survives_the_geotiff(package_zip: Path, request_for_package: PackageRequest) -> None:
    classes = load_package(package_zip).reference.classes
    source = request_for_package.result.class_map
    probe = at(300.0, ROUTE_NORTH_M)
    x, y = classes.to_px(probe)
    sx, sy = source.to_px(probe)
    assert classes.pixels[int(y), int(x)].tolist() == source.pixels[int(sy), int(sx)].tolist()


def test_blind_map_and_corridor(package_zip: Path, request_for_package: PackageRequest) -> None:
    reference = load_package(package_zip).reference
    assert not reference.is_blind(at(300.0, ROUTE_NORTH_M))
    assert reference.is_blind(at(550.0, ROUTE_NORTH_M))
    assert reference.is_blind(at(300.0, ROUTE_NORTH_M + 250.0))           # outside the strip
    assert reference.is_blind(at(-5000.0, 0.0))                            # outside the raster
    assert not reference.contains(at(-5000.0, 0.0))
    width = corridor_width(64.0, request_for_package.inputs, request_for_package.config)
    classes = reference.classes
    half_m = width.half_width_m
    inside = classes.pixels[pixel_index(classes, at(ROUTE_EAST_M[0], ROUTE_NORTH_M + half_m * 0.5))]
    corner = classes.pixels[pixel_index(classes, at(ROUTE_EAST_M[0] - half_m * 0.9, ROUTE_NORTH_M + half_m * 0.9))]
    assert inside.min() > 0 and corner.max() == 0


def test_corridor_width_terms(request_for_package: PackageRequest) -> None:
    config, inputs = request_for_package.config, request_for_package.inputs
    width = corridor_width(64.0, inputs, config)
    assert width.drift_m == pytest.approx(config.drift_sigmas * drift_sigma_m(64.0, inputs.speed_mps, config.ins))
    assert width.half_width_m == pytest.approx(width.drift_m + width.footprint_half_diagonal_m + width.search_half_range_m)
    assert request_for_package.inputs.altitude_m == 40.0
    assert width.footprint_half_diagonal_m == pytest.approx(np.hypot(46.19, 46.19 * 9 / 16) / 2, rel=1e-3)


def test_package_is_loadable_from_folder(package_zip: Path) -> None:
    assert load_package(package_zip.with_suffix("")).mission_id == "test"


def test_tampered_file_is_rejected(package_zip: Path, tmp_path: Path) -> None:
    folder = package_zip.with_suffix("")
    with (folder / CLASSES_PATH).open("r+b") as handle:
        handle.seek(100)
        handle.write(b"\xff\xff")
    with pytest.raises(PackageIntegrityError, match="classes.tif"):
        load_package(folder)


def test_tampered_zip_is_rejected(package_zip: Path, tmp_path: Path) -> None:
    forged = tmp_path / "in" / package_zip.name
    forged.parent.mkdir()
    with zipfile.ZipFile(package_zip) as source, zipfile.ZipFile(forged, "w") as target:
        for item in source.infolist():
            data = source.read(item.filename)
            target.writestr(item.filename, b"swapped model" if item.filename.endswith("drone.onnx") else data)
    with pytest.raises(PackageIntegrityError, match="drone.onnx"):
        load_package(forged)


def test_missing_file_and_wrong_version(package_zip: Path) -> None:
    folder = package_zip.with_suffix("")
    manifest_path = folder / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    manifest_path.write_text(json.dumps({**manifest, "format_version": FORMAT_VERSION + 1}))
    with pytest.raises(PackageFormatError, match="format version"):
        load_package(folder)
    manifest_path.write_text(json.dumps(manifest))
    (folder / "models" / "drone.onnx").unlink()
    with pytest.raises(PackageIntegrityError, match="missing"):
        load_package(folder)


def test_not_a_package(tmp_path: Path) -> None:
    junk = tmp_path / "mission-junk.zip"
    junk.write_text("not a zip")
    with pytest.raises(PackageFormatError):
        load_package(junk)
    with pytest.raises(FileNotFoundError):
        load_package(tmp_path / "absent.zip")


def test_unsafe_zip_member_is_rejected(tmp_path: Path) -> None:
    evil = tmp_path / "in" / "mission-evil.zip"
    evil.parent.mkdir()
    with zipfile.ZipFile(evil, "w") as zf:
        zf.writestr("../escaped.txt", "x")
    with pytest.raises(PackageFormatError, match="unsafe"):
        load_package(evil)
    assert not (tmp_path / "escaped.txt").exists()


def test_missing_ardupilot_file_is_an_error(tmp_path: Path, request_for_package: PackageRequest) -> None:
    files = dict(request_for_package.ardupilot_files)
    del files["uavloc.param"]
    with pytest.raises(ValueError, match="uavloc.param"):
        write_package(tmp_path / "out", replace(request_for_package, ardupilot_files=files))


def test_existing_package_is_not_overwritten(package_zip: Path, tmp_path: Path, request_for_package: PackageRequest) -> None:
    with pytest.raises(FileExistsError):
        write_package(tmp_path / "out", request_for_package)


def test_newest_package_skips_invalid_and_picks_latest_valid(tmp_path: Path, request_for_package: PackageRequest) -> None:
    directory = tmp_path / "missions"
    older = write_package(directory, replace(request_for_package, mission_id="older"))
    newer = write_package(directory, replace(request_for_package, mission_id="newer",
                                             created_utc=request_for_package.created_utc + timedelta(hours=1)))
    newest = write_package(directory, replace(request_for_package, mission_id="newest",
                                              created_utc=request_for_package.created_utc + timedelta(hours=2)))
    shutil.rmtree(newest.with_suffix(""))
    with zipfile.ZipFile(newest, "w") as zf:      # newest zip is a truncated copy
        zf.writestr("mission-newest/manifest.json", json.dumps({"format_version": FORMAT_VERSION, "files": {},
                                                                  "created_utc": datetime.now(timezone.utc).isoformat()}))
    assert newest_package(directory) == newer
    assert older.exists()


def test_newest_package_with_nothing_valid(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        newest_package(tmp_path)
