"""ESA WorldCover 10 m land cover, sampled onto a z18 Web Mercator mosaic.

Probe 2 reaches for a raster map rather than a model after probe 1 failed: a
land-cover product cannot draw a wrong parcel boundary, because it draws no
boundaries at all. It is coarse -- one 10 m cell is 26 px at z18 -- and that is
the trade being tested, not a defect to work around.

Access is public HTTPS with range requests over the COGs, so only the window a
location needs is fetched; the 3-degree tiles are 77 MB each and never land on
disk whole. Set `GDAL_DISABLE_READDIR_ON_OPEN` or every open re-lists the
bucket prefix.

WorldCover is a plain EPSG:4326 grid, so no reprojection library is needed:
`TileGrid.to_latlon` gives each output pixel's coordinate and the source is
sampled nearest-neighbour, which is the only correct choice for a categorical
raster.
"""

from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")

import rasterio  # noqa: E402
from rasterio.windows import from_bounds  # noqa: E402

BASE = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"

# The published legend. Three of these map onto our class set; the rest are
# background as far as this project is concerned.
LEGEND = {
    10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built",
    60: "bare", 70: "snow", 80: "water", 90: "wetland", 95: "mangrove",
    100: "moss",
}

# Our five classes, as far as WorldCover can speak to them. `track` has no
# equivalent at 10 m and is deliberately absent.
WOODY = (10, 20)          # tree cover + shrubland
WATER = (80, 90)          # permanent water + herbaceous wetland
BUILT = (50,)


def tile_name(lat: float, lon: float) -> str:
    """The 3-degree tile containing `lat, lon`, in WorldCover's naming."""
    y = int(np.floor(lat / 3.0) * 3)
    x = int(np.floor(lon / 3.0) * 3)
    return f"{'N' if y >= 0 else 'S'}{abs(y):02d}{'E' if x >= 0 else 'W'}{abs(x):03d}"


def url_for(lat: float, lon: float) -> str:
    return f"{BASE}/ESA_WorldCover_10m_2021_v200_{tile_name(lat, lon)}_Map.tif"


def sample(grid, shape: tuple[int, int], pad: float = 0.01) -> np.ndarray:
    """WorldCover class ids resampled onto the mosaic's pixel grid.

    Returns a `shape`-sized uint8 array of legend values, 0 where no source
    tile covers the pixel.
    """
    height, width = shape[:2]
    top, left = grid.to_latlon(0, 0)
    bottom, right = grid.to_latlon(width, height)

    # One output pixel -> one lat/lon, then nearest-neighbour into the source.
    lat_edges = np.array([grid.to_latlon(0, y)[0] for y in range(height)])
    lon_edges = np.array([grid.to_latlon(x, 0)[1] for x in range(width)])

    out = np.zeros((height, width), np.uint8)
    seen = set()
    for corner in ((top, left), (top, right), (bottom, left), (bottom, right)):
        name = tile_name(*corner)
        if name in seen:
            continue
        seen.add(name)

        with rasterio.open(url_for(*corner)) as src:
            window = from_bounds(
                left - pad, bottom - pad, right + pad, top + pad, src.transform
            ).round_offsets().round_lengths()
            window = window.intersection(
                rasterio.windows.Window(0, 0, src.width, src.height)
            )
            if window.width < 1 or window.height < 1:
                continue
            patch = src.read(1, window=window)
            wt = src.window_transform(window)

        # Affine is north-up and axis-aligned for this product, so the inverse
        # is two divisions rather than a general transform.
        cols = np.clip(((lon_edges - wt.c) / wt.a).astype(int), -1, patch.shape[1] - 1)
        rows = np.clip(((lat_edges - wt.f) / wt.e).astype(int), -1, patch.shape[0] - 1)
        inside = (cols >= 0)[None, :] & (rows >= 0)[:, None]
        block = patch[np.clip(rows, 0, None)[:, None], np.clip(cols, 0, None)[None, :]]
        out = np.where(inside & (out == 0), block, out)

    return out


def class_masks(cover: np.ndarray) -> dict[str, np.ndarray]:
    """WorldCover ids folded onto the classes this project uses."""
    return {
        "woody": np.isin(cover, WOODY),
        "water": np.isin(cover, WATER),
        "built": np.isin(cover, BUILT),
    }
