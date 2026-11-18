"""Web Mercator XYZ tiles, with the georeference that comes free with them.

The whole reason this PoC exists. ``pocs/map-derived-labels`` georeferenced a
Google Earth screen export by fitting an affine to four pushpins, and that fit
carries a 3.4 m residual no amount of care removes (bug-011). A tile has no such
problem: ``(z, x, y)`` *defines* a box in Web Mercator, so pixel -> lat/lon is
closed form and exact. There is nothing to fit and nothing to measure.

``TileGrid`` duck-types ``geo.Bounds`` and ``geo.Affine2D`` -- it offers
``to_pixel``, ``to_latlon`` and ``mpp`` -- so ``osm.rasterise`` and
``align.offset`` from the other PoC take it unchanged.

One real subtlety: Web Mercator pixel y is **not** linear in latitude. Over
800 m that is a sub-pixel effect, but ``to_pixel`` uses the exact formula
anyway, because getting it right costs one ``asinh``.
"""

from __future__ import annotations

import hashlib
import math
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

TILE = 256
EARTH_CIRCUMFERENCE = 2 * math.pi * 6378137.0
USER_AGENT = "uav-localization-research/0.1 (academic; contact via repo)"


def lonlat_to_global_px(lat: float, lon: float, z: int) -> tuple[float, float]:
    """Exact Web Mercator pixel coordinate at zoom ``z``, origin top-left."""
    n = TILE * (2 ** z)
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    return x, y


def global_px_to_lonlat(x: float, y: float, z: int) -> tuple[float, float]:
    n = TILE * (2 ** z)
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n))))
    return lat, lon


def resolution(lat: float, z: int) -> float:
    """Ground metres per pixel at this latitude and zoom."""
    return EARTH_CIRCUMFERENCE * math.cos(math.radians(lat)) / (TILE * 2 ** z)


@dataclass(frozen=True)
class TileGrid:
    """A rectangular block of tiles, and the exact geometry of the mosaic."""

    z: int
    x0: int
    y0: int
    nx: int
    ny: int

    @classmethod
    def covering(cls, lat: float, lon: float, width_m: float, height_m: float,
                 z: int) -> "TileGrid":
        """Smallest tile block centred on ``lat, lon`` covering the given extent."""
        res = resolution(lat, z)
        cx, cy = lonlat_to_global_px(lat, lon, z)
        half_w = width_m / res / 2.0
        half_h = height_m / res / 2.0
        x0 = int(math.floor((cx - half_w) / TILE))
        y0 = int(math.floor((cy - half_h) / TILE))
        x1 = int(math.ceil((cx + half_w) / TILE))
        y1 = int(math.ceil((cy + half_h) / TILE))
        return cls(z, x0, y0, x1 - x0, y1 - y0)

    @property
    def shape(self) -> tuple[int, int]:
        return self.ny * TILE, self.nx * TILE

    @property
    def origin(self) -> tuple[float, float]:
        return self.x0 * TILE, self.y0 * TILE

    def tiles(self):
        for ty in range(self.y0, self.y0 + self.ny):
            for tx in range(self.x0, self.x0 + self.nx):
                yield tx, ty

    # -- the geo.Bounds / geo.Affine2D interface -------------------------------

    def to_pixel(self, lat: float, lon: float, shape=None) -> tuple[float, float]:
        gx, gy = lonlat_to_global_px(lat, lon, self.z)
        ox, oy = self.origin
        return gx - ox, gy - oy

    def to_latlon(self, x: float, y: float, shape=None) -> tuple[float, float]:
        ox, oy = self.origin
        return global_px_to_lonlat(x + ox, y + oy, self.z)

    def mpp(self, shape=None) -> tuple[float, float]:
        """Metres per pixel at the mosaic centre.

        Web Mercator is conformal, so x and y scales are equal at a given
        latitude; they differ across the mosaic by cos(lat), which over 600 m
        of latitude here is 0.008%. Reporting the centre value is honest.
        """
        lat, _ = self.centre()
        r = resolution(lat, self.z)
        return r, r

    def centre(self) -> tuple[float, float]:
        h, w = self.shape
        return self.to_latlon(w / 2.0, h / 2.0)

    def bounds(self):
        """A ``geo.Bounds`` covering the mosaic -- for the Overpass query."""
        from geo import Bounds
        h, w = self.shape
        top, left = self.to_latlon(0, 0)
        bottom, right = self.to_latlon(w, h)
        return Bounds(top=top, left=left, bottom=bottom, right=right)

    def image_bounds(self, shape=None, margin_deg: float = 0.001):
        from geo import Bounds
        b = self.bounds()
        return Bounds(top=b.top + margin_deg, left=b.left - margin_deg,
                      bottom=b.bottom - margin_deg, right=b.right + margin_deg)


# -- fetching -----------------------------------------------------------------

_throttle = threading.Semaphore(8)


class TileSource:
    """A named XYZ endpoint. ``key`` namespaces the on-disk cache."""

    def __init__(self, key: str, template: str):
        self.key = key
        self.template = template

    def url(self, z: int, x: int, y: int) -> str:
        return (self.template
                .replace("{z}", str(z)).replace("{level}", str(z))
                .replace("{y}", str(y)).replace("{row}", str(y))
                .replace("{x}", str(x)).replace("{col}", str(x)))

    def __repr__(self) -> str:
        return f"TileSource({self.key})"


ESRI_CURRENT = TileSource(
    "esri-current",
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
)


def tile_bytes(source: TileSource, z: int, x: int, y: int, cache_dir: Path,
               retries: int = 3) -> bytes | None:
    """One tile, from disk if we have it. ``None`` means the server has no imagery."""
    path = cache_dir / source.key / str(z) / str(x) / f"{y}.jpg"
    if path.exists():
        data = path.read_bytes()
        return data if data else None
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(source.url(z, x, y),
                                     headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with _throttle:
                with urllib.request.urlopen(request, timeout=30) as response:
                    data = response.read()
            path.write_bytes(data)
            return data
        except urllib.error.HTTPError as exc:
            if exc.code in (400, 404):
                path.write_bytes(b"")      # cache the absence too
                return None
            if attempt == retries - 1:
                raise
        except Exception:
            if attempt == retries - 1:
                raise
    return None


def mosaic(source: TileSource, grid: TileGrid, cache_dir: Path,
           workers: int = 8) -> np.ndarray:
    """Fetch every tile in ``grid`` and stitch. Missing tiles come back black."""
    out = np.zeros((*grid.shape, 3), dtype=np.uint8)
    coords = list(grid.tiles())

    def one(tile):
        tx, ty = tile
        data = tile_bytes(source, grid.z, tx, ty, cache_dir)
        if not data:
            return tile, None
        import io
        return tile, np.array(Image.open(io.BytesIO(data)).convert("RGB"))

    with ThreadPoolExecutor(workers) as pool:
        for (tx, ty), patch in pool.map(one, coords):
            if patch is None:
                continue
            r = (ty - grid.y0) * TILE
            c = (tx - grid.x0) * TILE
            out[r:r + TILE, c:c + TILE] = patch
    return out


def tile_hash(source: TileSource, z: int, x: int, y: int, cache_dir: Path) -> str:
    data = tile_bytes(source, z, x, y, cache_dir)
    return hashlib.sha1(data).hexdigest()[:12] if data else "missing"
