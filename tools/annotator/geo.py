"""Web Mercator grid of one location: lat/lon <-> image pixel, exactly.

Copied from ``pocs/map-derived-labels/tiles.py`` (the PoC stays frozen as the
record). A location image is a mosaic of ``nx * ny`` XYZ tiles, so the
georeference is closed form -- nothing is fitted. ``web/js/geo.js`` implements
the same two formulas; ``checks/m0_geo.ipynb`` holds them to each other.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

TILE = 256
EARTH_CIRCUMFERENCE = 2 * math.pi * 6378137.0


def lonlat_to_global_px(lat: float, lon: float, z: int) -> tuple[float, float]:
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
    return EARTH_CIRCUMFERENCE * math.cos(math.radians(lat)) / (TILE * 2 ** z)


@dataclass(frozen=True)
class Grid:
    z: int
    x0: int
    y0: int
    nx: int
    ny: int

    @classmethod
    def from_meta(cls, meta: dict) -> "Grid":
        return cls(**meta["grid"])

    @classmethod
    def from_location(cls, loc_dir: Path) -> "Grid":
        return cls.from_meta(json.loads((loc_dir / "meta.json").read_text()))

    @property
    def shape(self) -> tuple[int, int]:
        """(height, width) of the mosaic."""
        return self.ny * TILE, self.nx * TILE

    def to_px(self, lat: float, lon: float) -> tuple[float, float]:
        gx, gy = lonlat_to_global_px(lat, lon, self.z)
        return gx - self.x0 * TILE, gy - self.y0 * TILE

    def to_latlon(self, x: float, y: float) -> tuple[float, float]:
        return global_px_to_lonlat(x + self.x0 * TILE, y + self.y0 * TILE, self.z)

    # duck-types the PoC TileGrid, so ported PoC code takes it unchanged
    def to_pixel(self, lat: float, lon: float, shape=None) -> tuple[float, float]:
        return self.to_px(lat, lon)

    def mpp(self) -> float:
        h, w = self.shape
        lat, _ = self.to_latlon(w / 2.0, h / 2.0)
        return resolution(lat, self.z)

    def bounds(self) -> dict:
        h, w = self.shape
        top, left = self.to_latlon(0, 0)
        bottom, right = self.to_latlon(w, h)
        return {"top": top, "left": left, "bottom": bottom, "right": right}

    # GeoJSON rings are [lon, lat]
    def ring_to_px(self, ring: list) -> list[tuple[float, float]]:
        return [self.to_px(lat, lon) for lon, lat in ring]

    def ring_to_lonlat(self, pts) -> list[list[float]]:
        out = []
        for x, y in pts:
            lat, lon = self.to_latlon(float(x), float(y))
            out.append([lon, lat])
        return out


@dataclass(frozen=True)
class PixelGrid:
    """Identity "georeference" for pixel-only images (drone frames): stored
    coordinates ARE frame pixels, ``[x, y]`` in place of ``[lon, lat]``."""
    h: int
    w: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.h, self.w

    def to_px(self, a: float, b: float) -> tuple[float, float]:
        return b, a                       # called as to_px(lat=y, lon=x)

    def to_latlon(self, x: float, y: float) -> tuple[float, float]:
        return y, x

    def ring_to_px(self, ring: list) -> list[tuple[float, float]]:
        return [(float(x), float(y)) for x, y in ring]

    def ring_to_lonlat(self, pts) -> list[list[float]]:
        return [[float(x), float(y)] for x, y in pts]


def grid_for(meta: dict):
    """Web Mercator grid for a map location, identity for a pixel-only one."""
    if meta.get("crs") == "pixel" or "grid" not in meta:
        return PixelGrid(*meta["shape"])
    return Grid.from_meta(meta)
