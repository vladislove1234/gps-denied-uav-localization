"""The imagery layers the map can show. Add an entry to `TILE_SOURCES` to offer another one."""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass

from tools.route_planner.config import secrets_file
from uavloc.data.google_maps import create_session, load_api_key, satellite_tile_template
from uavloc.data.tiles import ESRI_WORLD_IMAGERY, XyzTileSource

ESRI_MAX_NATIVE_ZOOM = 19       # beyond this Esri returns "no data" tiles; Leaflet upscales instead
GOOGLE_SATELLITE_KEY = "google-satellite"   # the cache folder pocs/google-satellite already filled
GOOGLE_MAX_NATIVE_ZOOM = 19     # native over rural Ukraine (pocs/google-satellite); Google upsamples above it
GOOGLE_ATTRIBUTION = "Imagery © Google"
EXTRA_ZOOM_LEVELS = 1


@dataclass(frozen=True)
class MapTileSource:
    label: str
    source: XyzTileSource
    max_native_zoom: int

    @property
    def max_zoom(self) -> int:
        return self.max_native_zoom + EXTRA_ZOOM_LEVELS

    def leaflet_options(self) -> dict:
        """What the map page needs to build its `L.tileLayer`; Esri's `{z}/{y}/{x}` placeholders are Leaflet's own."""
        return {"template": self.source.template, "attribution": self.source.attribution,
                "maxNativeZoom": self.max_native_zoom, "maxZoom": self.max_zoom}


@dataclass(frozen=True)
class TileSourceOption:
    label: str
    build: Callable[[str], MapTileSource]   # takes the label


def esri_world_imagery(label: str) -> MapTileSource:
    return MapTileSource(label, ESRI_WORLD_IMAGERY, ESRI_MAX_NATIVE_ZOOM)


def google_satellite(label: str) -> MapTileSource:
    """Needs `GOOGLE_MAPS_API_KEY` (environment or `secrets_file()`) and the network.

    The tile URL carries a session token created here.
    """
    api_key = load_api_key(secrets_file())
    template = satellite_tile_template(api_key, create_session(api_key))
    return MapTileSource(label, XyzTileSource(GOOGLE_SATELLITE_KEY, template, GOOGLE_ATTRIBUTION),
                         GOOGLE_MAX_NATIVE_ZOOM)


TILE_SOURCES: dict[str, TileSourceOption] = {
    ESRI_WORLD_IMAGERY.key: TileSourceOption("Esri World Imagery", esri_world_imagery),
    GOOGLE_SATELLITE_KEY: TileSourceOption("Google satellite", google_satellite),
}


@functools.cache
def tile_source(key: str) -> MapTileSource:
    """Built once per run; a Google session lasts about two weeks, far longer than the app stays open."""
    if key not in TILE_SOURCES:
        raise KeyError(f"unknown tile source {key!r}, known: {sorted(TILE_SOURCES)}")
    option = TILE_SOURCES[key]
    return option.build(option.label)
