"""Web Mercator XYZ tiles: exact tile geometry, cached fetching, and mosaics as `GeoImage`.

Promoted from `pocs/map-derived-labels/tiles.py` on its second use (the route planner).
The on-disk cache layout is the PoC's, `<cache>/<source key>/<z>/<x>/<y>.jpg`, so tiles
the datasets already fetched are reused. An empty file records a tile the server lacks.
"""

from __future__ import annotations

import io
import math
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol

import numpy as np
from PIL import Image

from uavloc.data.satellite import GeoImage
from uavloc.geo import EARTH_RADIUS_M, GeoBounds, LatLon

TILE_PX = 256
EARTH_CIRCUMFERENCE_M = 2 * math.pi * EARTH_RADIUS_M
USER_AGENT = "uav-localization-research/0.1 (academic; contact via repo)"
HTTP_NO_TILE = (400, 404)
FETCH_RETRIES = 5         # with linear backoff: ~20 s of waiting rides out a DNS or Wi-Fi blip
FETCH_BACKOFF_S = 2.0
FETCH_TIMEOUT_S = 30
FETCH_WORKERS = 8
RESAMPLE_CHUNK_ROWS = 512
_throttle = threading.Semaphore(FETCH_WORKERS)


class TileSource(Protocol):
    """An XYZ imagery endpoint. `key` names its cache folder."""

    key: str

    def url(self, zoom: int, x: int, y: int) -> str: ...


@dataclass(frozen=True)
class XyzTileSource:
    """URL template with `{z}`, `{x}`, `{y}` placeholders."""

    key: str
    template: str
    attribution: str = ""

    def url(self, zoom: int, x: int, y: int) -> str:
        return self.template.format(z=zoom, x=x, y=y)


ESRI_WORLD_IMAGERY = XyzTileSource(
    "esri-current",
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    "Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community",
)


def global_px(point: LatLon, zoom: int) -> tuple[float, float]:
    """Exact Web Mercator pixel `(x, y)` at `zoom`, origin at the world's top-left."""
    n = TILE_PX * 2 ** zoom
    x = (point.lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(math.radians(point.lat))) / math.pi) / 2.0 * n
    return x, y


def latlon_of_global_px(x: float, y: float, zoom: int) -> LatLon:
    n = TILE_PX * 2 ** zoom
    return LatLon(math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n)))), x / n * 360.0 - 180.0)


def ground_resolution_m(lat: float, zoom: int) -> float:
    """Metres per pixel at this latitude and zoom."""
    return EARTH_CIRCUMFERENCE_M * math.cos(math.radians(lat)) / (TILE_PX * 2 ** zoom)


@dataclass(frozen=True)
class TileRange:
    """A rectangular block of tiles at one zoom."""

    zoom: int
    x0: int
    y0: int
    nx: int
    ny: int

    @classmethod
    def covering(cls, bounds: GeoBounds, zoom: int) -> TileRange:
        left, top = global_px(bounds.top_left, zoom)
        right, bottom = global_px(bounds.bottom_right, zoom)
        x0, y0 = math.floor(left / TILE_PX), math.floor(top / TILE_PX)
        return cls(zoom, x0, y0, math.ceil(right / TILE_PX) - x0, math.ceil(bottom / TILE_PX) - y0)

    @property
    def shape_px(self) -> tuple[int, int]:
        return self.ny * TILE_PX, self.nx * TILE_PX

    @property
    def bounds(self) -> GeoBounds:
        top_left = latlon_of_global_px(self.x0 * TILE_PX, self.y0 * TILE_PX, self.zoom)
        bottom_right = latlon_of_global_px((self.x0 + self.nx) * TILE_PX, (self.y0 + self.ny) * TILE_PX, self.zoom)
        return GeoBounds(top_left.lat, top_left.lon, bottom_right.lat, bottom_right.lon)

    def tiles(self) -> Iterator[tuple[int, int]]:
        for y in range(self.y0, self.y0 + self.ny):
            for x in range(self.x0, self.x0 + self.nx):
                yield x, y


def fetch_tile(source: TileSource, zoom: int, x: int, y: int, cache_dir: Path) -> np.ndarray | None:
    """`(256, 256, 3)` uint8 from the cache or the server; None where the server has no imagery."""
    path = cache_dir / source.key / str(zoom) / str(x) / f"{y}.jpg"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_download(source.url(zoom, x, y)))
    data = path.read_bytes()
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB")) if data else None


def _download(url: str) -> bytes:
    """Tile bytes, or b"" when the server says the tile does not exist."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(FETCH_RETRIES):
        try:
            with _throttle, urllib.request.urlopen(request, timeout=FETCH_TIMEOUT_S) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code in HTTP_NO_TILE:
                return b""
            if attempt == FETCH_RETRIES - 1:
                raise
        except OSError:   # URLError, and a dropped connection (RemoteDisconnected), which urllib does not wrap
            if attempt == FETCH_RETRIES - 1:
                raise
        time.sleep(FETCH_BACKOFF_S * (attempt + 1))
    raise RuntimeError("unreachable")


def mosaic_px(source: TileSource, tiles: TileRange, cache_dir: Path) -> np.ndarray:
    """Stitched `(H, W, 3)` uint8 in Web Mercator pixels; missing tiles stay black."""
    out = np.zeros((*tiles.shape_px, 3), np.uint8)

    def one(tile: tuple[int, int]) -> tuple[tuple[int, int], np.ndarray | None]:
        return tile, fetch_tile(source, tiles.zoom, *tile, cache_dir)

    with ThreadPoolExecutor(FETCH_WORKERS) as pool:
        for (x, y), patch in pool.map(one, list(tiles.tiles())):
            if patch is not None:
                row, col = (y - tiles.y0) * TILE_PX, (x - tiles.x0) * TILE_PX
                out[row:row + TILE_PX, col:col + TILE_PX] = patch
    return out


def mercator_to_geoimage(pixels: np.ndarray, tiles: TileRange) -> GeoImage:
    """Resample Web Mercator rows so latitude is linear down the raster, as `GeoImage` assumes.

    Columns are already linear in longitude. Over 10 km the Mercator row error is metres, so it is not ignored.
    """
    bounds = tiles.bounds
    h = pixels.shape[0]
    lats = bounds.north - (np.arange(h) + 0.5) / h * (bounds.north - bounds.south)
    n = TILE_PX * 2 ** tiles.zoom
    merc_y = (1.0 - np.arcsinh(np.tan(np.radians(lats))) / np.pi) / 2.0 * n
    source_row = np.clip(merc_y - tiles.y0 * TILE_PX - 0.5, 0, h - 1)
    below = np.floor(source_row).astype(int)
    above = np.minimum(below + 1, h - 1)
    frac = (source_row - below).astype(np.float32).reshape(-1, *([1] * (pixels.ndim - 1)))
    out = np.empty_like(pixels)
    for start in range(0, h, RESAMPLE_CHUNK_ROWS):   # chunks keep a many-channel corridor out of float64 memory
        rows = slice(start, start + RESAMPLE_CHUNK_ROWS)
        blended = pixels[below[rows]] * (1 - frac[rows]) + pixels[above[rows]] * frac[rows]
        out[rows] = np.round(blended) if np.issubdtype(pixels.dtype, np.integer) else blended
    return GeoImage(out, bounds.top_left, bounds.bottom_right)


def geo_mosaic(source: TileSource, bounds: GeoBounds, zoom: int, cache_dir: Path) -> GeoImage:
    """Tile-aligned RGB raster covering `bounds`."""
    tiles = TileRange.covering(bounds, zoom)
    return mercator_to_geoimage(mosaic_px(source, tiles, cache_dir), tiles)
