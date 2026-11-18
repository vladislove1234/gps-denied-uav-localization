"""The mission package a drone loads: a versioned, data-only folder (zipped for transfer).

Reading side only, so it stays light enough for the Pi (no torch). Writing is in `package_writer`.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

from uavloc.geo import LatLon
from uavloc.onboard.contracts import ReferenceMap
from uavloc.onboard.raster import read_geotiff

FORMAT_VERSION = 1
MANIFEST_NAME = "manifest.json"
PACKAGE_PREFIX = "mission-"
CLASSES_PATH = "reference/classes.tif"
BLIND_PATH = "reference/blind.tif"
SATELLITE_PATH = "reference/satellite.jpg"
MODEL_PATH = "models/drone.onnx"
CAMERA_PATH = "config/camera.json"
LOCALIZER_PATH = "config/localizer.json"
ARDUPILOT_DIR = "ardupilot"
ARDUPILOT_FILE_NAMES = ("mission.waypoints", "uavloc.param", "vision-gps-switch.lua")
REQUIRED_FILES = (CLASSES_PATH, BLIND_PATH, SATELLITE_PATH, MODEL_PATH, CAMERA_PATH, LOCALIZER_PATH,
                  *(f"{ARDUPILOT_DIR}/{name}" for name in ARDUPILOT_FILE_NAMES))
HASH_CHUNK_BYTES = 1 << 20


class PackageError(Exception):
    """The package cannot be used."""


class PackageFormatError(PackageError):
    """Unknown or missing format version, or a manifest that is not valid."""


class PackageIntegrityError(PackageError):
    """A file is missing or its sha256 does not match the manifest."""


@dataclass(frozen=True)
class MissionPackage:
    root: Path
    manifest: dict
    reference: ReferenceMap
    satellite_corners: tuple[LatLon, LatLon]   # top-left, bottom-right of `reference/satellite.jpg`

    @property
    def mission_id(self) -> str:
        return self.manifest["mission_id"]

    @property
    def created_utc(self) -> str:
        return self.manifest["created_utc"]

    @property
    def waypoints(self) -> tuple[LatLon, ...]:
        """Out and back: start, legs' corners, end, the corners again in reverse, start."""
        return tuple(LatLon(*p) for p in self.manifest["route"]["waypoints_out_and_back"])

    @property
    def model_path(self) -> Path:
        return self.root / MODEL_PATH

    @property
    def satellite_path(self) -> Path:
        return self.root / SATELLITE_PATH

    @property
    def manifest_path(self) -> Path:
        return self.root / MANIFEST_NAME

    @property
    def camera_settings(self) -> dict:
        """`config/camera.json`: the fields of `uavloc.simulation.camera.CameraConfig`."""
        return json.loads((self.root / CAMERA_PATH).read_text())

    @property
    def localizer_settings(self) -> dict:
        return json.loads((self.root / LOCALIZER_PATH).read_text())

    def ardupilot_file(self, name: str) -> Path:
        return self.root / ARDUPILOT_DIR / name


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path) -> dict:
    """The manifest of a zip or folder, without extracting anything."""
    try:
        if path.is_dir():
            text = (path / MANIFEST_NAME).read_text()
        else:
            with zipfile.ZipFile(path) as zf:
                text = zf.read(f"{path.stem}/{MANIFEST_NAME}").decode()
        return json.loads(text)
    except (OSError, KeyError, zipfile.BadZipFile, json.JSONDecodeError) as error:
        raise PackageFormatError(f"{path}: cannot read {MANIFEST_NAME}: {error}") from error


def extract_zip(archive: Path) -> Path:
    """Unpack next to the zip as a folder of the same stem; refuses members that would escape it."""
    target = archive.parent.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            if not (target / member).resolve().is_relative_to(target):
                raise PackageFormatError(f"{archive}: unsafe path {member!r}")
        zf.extractall(target)
    return archive.with_suffix("")


def verify_folder(root: Path) -> dict:
    """Check format version, required files and every sha256; returns the manifest."""
    manifest = read_manifest(root)
    version = manifest.get("format_version")
    if version != FORMAT_VERSION:
        raise PackageFormatError(f"{root}: format version {version!r}, this software reads {FORMAT_VERSION}")
    hashes = manifest.get("files", {})
    for relative in REQUIRED_FILES:
        if relative not in hashes:
            raise PackageIntegrityError(f"{root}: manifest lists no {relative}")
    for relative, expected in hashes.items():
        path = root / relative
        if not path.is_file():
            raise PackageIntegrityError(f"{root}: {relative} is missing")
        if sha256_of(path) != expected:
            raise PackageIntegrityError(f"{root}: {relative} does not match its sha256 (corrupt copy?)")
    return manifest


def verified_root(path: Path) -> tuple[Path, dict]:
    """The folder of a package (a zip is unpacked next to itself) and its manifest, after verifying every file."""
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        root = path if path.is_dir() else extract_zip(path)
    except zipfile.BadZipFile as error:
        raise PackageFormatError(f"{path} is not a zip file") from error
    return root, verify_folder(root)


def load_package(path: Path) -> MissionPackage:
    """Load a package from its zip or its folder, after verifying every file."""
    root, manifest = verified_root(path)
    reference = ReferenceMap(read_geotiff(root / CLASSES_PATH), read_geotiff(root / BLIND_PATH))
    corners = manifest["reference"]["satellite"]
    return MissionPackage(root, manifest, reference,
                          (LatLon(*corners["top_left"]), LatLon(*corners["bottom_right"])))


def package_candidates(directory: Path) -> list[Path]:
    """Zips and folders named `mission-*` in `directory`, newest manifest first; a folder that sits next to its
    zip counts once, as the zip."""
    zips = sorted(directory.glob(f"{PACKAGE_PREFIX}*.zip"))
    folders = [p for p in sorted(directory.glob(f"{PACKAGE_PREFIX}*")) if p.is_dir() and p.with_suffix(".zip") not in zips]
    dated = []
    for path in [*zips, *folders]:
        try:
            dated.append((read_manifest(path).get("created_utc", ""), path))
        except PackageError:
            continue
    return [path for _, path in sorted(dated, key=lambda pair: pair[0], reverse=True)]


def newest_package(directory: Path) -> Path:
    """The newest package in `directory` that verifies; raises FileNotFoundError when there is none."""
    problems = []
    for path in package_candidates(directory):
        try:
            verified_root(path)
            return path
        except PackageError as error:
            problems.append(str(error))
    raise FileNotFoundError(f"no valid mission package in {directory}" + (f" ({'; '.join(problems)})" if problems else ""))
