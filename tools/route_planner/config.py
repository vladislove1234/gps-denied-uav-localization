"""Every path and default the app uses, in one place. Nothing else builds a path from the repo root."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

from uavloc.track import REPO_ROOT

DEFAULT_TILE_SOURCE_KEY = "esri-current"
DEFAULT_CONNECTION = "tcp:127.0.0.1:5760"       # ArduPilot SITL; a real flight controller is "/dev/tty.usbmodem...,115200"
DEFAULT_VIEW_LAT = 49.91                          # first test area, near Birky
DEFAULT_VIEW_LON = 23.9
DEFAULT_VIEW_ZOOM = 13
SETTINGS_FILE_NAME = "settings.json"
SECRETS_FILE_NAME = ".env"                        # GOOGLE_MAPS_API_KEY=...
IS_FROZEN = getattr(sys, "frozen", False)         # inside a PyInstaller bundle
# Read-only resources (models, web/, lua) sit in the bundle; caches and exports must outlive it, so they go to app-data.
RESOURCE_ROOT = Path(sys._MEIPASS) if IS_FROZEN else REPO_ROOT   # type: ignore[attr-defined]
PATH_FIELDS = frozenset({"tile_cache", "planner_cache", "satellite_onnx", "drone_onnx", "output_dir"})


def user_data_dir() -> Path:
    """Same folder as `QStandardPaths.AppDataLocation` for this app's organisation and name."""
    if sys.platform == "darwin":
        base = Path.home() / "Library/Application Support"
    elif sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData/Roaming"))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
    return base / "uav-localization" / "route-planner"


def secrets_file() -> Path:
    """The bundle carries no `.env`, so the packaged app reads its keys from the app-data folder."""
    return (user_data_dir() if IS_FROZEN else REPO_ROOT) / SECRETS_FILE_NAME


@dataclass(frozen=True)
class AppConfig:
    tile_cache: Path
    planner_cache: Path
    satellite_onnx: Path
    drone_onnx: Path
    output_dir: Path
    tile_source_key: str = DEFAULT_TILE_SOURCE_KEY
    connection: str = DEFAULT_CONNECTION
    view_lat: float = DEFAULT_VIEW_LAT
    view_lon: float = DEFAULT_VIEW_LON
    view_zoom: int = DEFAULT_VIEW_ZOOM

    @classmethod
    def defaults(cls, root: Path = RESOURCE_ROOT) -> AppConfig:
        data_root = user_data_dir() if IS_FROZEN and root == RESOURCE_ROOT else root / "data/derived"
        return cls(
            tile_cache=data_root / "tile-cache",
            planner_cache=data_root / "planner-cache",
            satellite_onnx=root / "pocs/lraspp-pi5/models/lraspp-satellite.onnx",
            drone_onnx=root / "pocs/lraspp-pi5/models/lraspp-drone.onnx",
            output_dir=data_root / "route-planner-exports")

    @classmethod
    def load(cls, settings_path: Path, root: Path = RESOURCE_ROOT) -> AppConfig:
        """Defaults, overridden by the JSON settings file when it exists. Relative paths resolve against `root`."""
        config = cls.defaults(root)
        if not settings_path.exists():
            return config
        overrides = json.loads(settings_path.read_text())
        unknown = set(overrides) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"{settings_path}: unknown setting(s) {sorted(unknown)}")
        for name in PATH_FIELDS & overrides.keys():
            overrides[name] = root / overrides[name]   # an absolute value replaces the root
        return replace(config, **overrides)

    def save(self, settings_path: Path) -> None:
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings_path.write_text(json.dumps({k: str(v) if isinstance(v, Path) else v
                                             for k, v in asdict(self).items()}, indent=2))
