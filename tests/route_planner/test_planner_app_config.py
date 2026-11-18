from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.route_planner.config import AppConfig
from tools.route_planner import tile_sources
from tools.route_planner.tile_sources import TILE_SOURCES, tile_source


def test_defaults_are_built_from_the_given_root(tmp_path: Path) -> None:
    config = AppConfig.defaults(tmp_path)
    assert config.tile_cache == tmp_path / "data/derived/tile-cache"
    assert config.satellite_onnx == tmp_path / "pocs/lraspp-pi5/models/lraspp-satellite.onnx"


def test_missing_settings_file_gives_defaults(tmp_path: Path) -> None:
    assert AppConfig.load(tmp_path / "nope.json", tmp_path) == AppConfig.defaults(tmp_path)


def test_settings_override_paths_and_values(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"tile_cache": "elsewhere/tiles", "output_dir": "/abs/out",
                                    "connection": "/dev/ttyACM0,115200"}))
    config = AppConfig.load(settings, tmp_path)
    assert config.tile_cache == tmp_path / "elsewhere/tiles"
    assert config.output_dir == Path("/abs/out")
    assert config.connection == "/dev/ttyACM0,115200"
    assert config.planner_cache == tmp_path / "data/derived/planner-cache"


def test_unknown_setting_raises(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"tile_chache": "typo"}))
    with pytest.raises(ValueError, match="tile_chache"):
        AppConfig.load(settings, tmp_path)


def test_save_then_load_round_trips(tmp_path: Path) -> None:
    config = AppConfig.defaults(tmp_path / "root")
    config.save(tmp_path / "app" / "settings.json")
    assert AppConfig.load(tmp_path / "app" / "settings.json", tmp_path / "other") == config


def test_esri_is_registered_with_attribution_and_leaflet_placeholders() -> None:
    options = tile_source("esri-current").leaflet_options()
    assert set(TILE_SOURCES) == {"esri-current", "google-satellite"}
    assert "Esri" in options["attribution"]
    assert all(f"{{{key}}}" in options["template"] for key in "zxy")
    assert options["maxZoom"] > options["maxNativeZoom"]


def test_google_builds_a_session_url_with_leaflet_placeholders(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tile_sources, "load_api_key", lambda env_file: "KEY")
    monkeypatch.setattr(tile_sources, "create_session", lambda api_key: f"session-for-{api_key}")
    google = TILE_SOURCES["google-satellite"]
    source = google.build(google.label)
    options = source.leaflet_options()
    assert source.source.key == "google-satellite"
    assert "session=session-for-KEY" in options["template"] and "key=KEY" in options["template"]
    assert all(f"{{{key}}}" in options["template"] for key in "zxy")
    assert source.source.url(19, 1, 2).startswith("https://tile.googleapis.com/v1/2dtiles/19/1/2?")
    assert "Google" in options["attribution"]


def test_unknown_tile_source_raises() -> None:
    with pytest.raises(KeyError):
        tile_source("osm")
