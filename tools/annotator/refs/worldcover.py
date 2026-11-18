"""ESA WorldCover 2021, 10 m. Ported from ``pocs/class-label-sources/worldcover.py``.

Read over HTTP range requests straight from the public S3 COGs, nearest-
neighbour onto the location grid -- so every class edge is a 10 m staircase.
Useful as a coarse woody prior; also offers water and cropland (-> field
overlay) because they come for free.
"""

from __future__ import annotations

import os

import numpy as np

from .base import Adapter, Location, RefResult, register

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")

BASE = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"
WOODY = (10, 20)          # tree cover + shrubland
WATER = (80, 90)          # permanent water + herbaceous wetland
CROP = (40,)


def tile_name(lat: float, lon: float) -> str:
    y = int(np.floor(lat / 3.0) * 3)
    x = int(np.floor(lon / 3.0) * 3)
    return f"{'N' if y >= 0 else 'S'}{abs(y):02d}{'E' if x >= 0 else 'W'}{abs(x):03d}"


def url_for(lat: float, lon: float) -> str:
    return f"{BASE}/ESA_WorldCover_10m_2021_v200_{tile_name(lat, lon)}_Map.tif"


def sample(grid, shape: tuple[int, int], pad: float = 0.01) -> np.ndarray:
    import rasterio
    from rasterio.windows import Window, from_bounds

    height, width = shape
    top, left = grid.to_latlon(0, 0)
    bottom, right = grid.to_latlon(width, height)
    lat_rows = np.array([grid.to_latlon(0, y + 0.5)[0] for y in range(height)])
    lon_cols = np.array([grid.to_latlon(x + 0.5, 0)[1] for x in range(width)])

    out = np.zeros((height, width), np.uint8)
    seen = set()
    for corner in ((top, left), (top, right), (bottom, left), (bottom, right)):
        name = tile_name(*corner)
        if name in seen:
            continue
        seen.add(name)
        with rasterio.open(url_for(*corner)) as src:
            window = from_bounds(left - pad, bottom - pad, right + pad, top + pad,
                                 src.transform).round_offsets().round_lengths()
            window = window.intersection(Window(0, 0, src.width, src.height))
            if window.width < 1 or window.height < 1:
                continue
            patch = src.read(1, window=window)
            wt = src.window_transform(window)
        cols = np.floor((lon_cols - wt.c) / wt.a).astype(int)
        rows = np.floor((lat_rows - wt.f) / wt.e).astype(int)
        inside = ((cols >= 0) & (cols < patch.shape[1]))[None, :] & \
                 ((rows >= 0) & (rows < patch.shape[0]))[:, None]
        block = patch[np.clip(rows, 0, patch.shape[0] - 1)[:, None],
                      np.clip(cols, 0, patch.shape[1] - 1)[None, :]]
        out = np.where(inside & (out == 0), block, out)
    return out


@register
class WorldCover(Adapter):
    name = "worldcover"
    version = "1"
    provides = ("woody", "water", "field")
    params = {"product": "ESA WorldCover v200 2021", "woody": WOODY, "water": WATER, "field": CROP}

    def run(self, loc: Location) -> RefResult:
        cover = sample(loc.grid, loc.shape)
        return RefResult(masks={
            "woody": np.isin(cover, WOODY),
            "water": np.isin(cover, WATER),
            "field": np.isin(cover, CROP),
        }, info={"covered_frac": float((cover > 0).mean())})
