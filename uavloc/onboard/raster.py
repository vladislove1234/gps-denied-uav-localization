"""Cropping and GeoTIFF files for the rasters a mission package carries."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds

from uavloc.data.satellite import GeoImage
from uavloc.geo import GeoBounds, LatLon

GEOTIFF_CRS = "EPSG:4326"
TIFF_BLOCK_PX = 256
DEFLATE_PREDICTOR_INTEGER = 2
DEFLATE_PREDICTOR_FLOAT = 3


def crop_to_bounds(image: GeoImage, bounds: GeoBounds) -> GeoImage:
    """Pixel-aligned crop that covers `bounds` (grown to whole pixels, clipped to the raster)."""
    height, width = image.pixels.shape[:2]
    x0, y0 = image.to_px(bounds.top_left)
    x1, y1 = image.to_px(bounds.bottom_right)
    x0, y0 = max(math.floor(x0), 0), max(math.floor(y0), 0)
    x1, y1 = min(math.ceil(x1), width), min(math.ceil(y1), height)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"{bounds} does not overlap the raster")
    return GeoImage(image.pixels[y0:y1, x0:x1], image.to_latlon(x0, y0), image.to_latlon(x1, y1))


def write_geotiff(path: Path, image: GeoImage) -> None:
    """EPSG:4326 GeoTIFF, one band per channel; the affine transform is the `GeoImage`'s linear lat/lon map."""
    height, width, bands = image.pixels.shape
    transform = from_bounds(image.top_left.lon, image.bottom_right.lat, image.bottom_right.lon,
                            image.top_left.lat, width, height)
    predictor = DEFLATE_PREDICTOR_FLOAT if np.issubdtype(image.pixels.dtype, np.floating) else DEFLATE_PREDICTOR_INTEGER
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=height, width=width, count=bands,
                       dtype=image.pixels.dtype, crs=GEOTIFF_CRS, transform=transform, compress="deflate",
                       predictor=predictor, tiled=True, blockxsize=TIFF_BLOCK_PX, blockysize=TIFF_BLOCK_PX) as dataset:
        dataset.write(np.moveaxis(image.pixels, -1, 0))


def read_geotiff(path: Path) -> GeoImage:
    with rasterio.open(path) as dataset:
        bounds = dataset.bounds
        pixels = np.moveaxis(dataset.read(), 0, -1)
    return GeoImage(np.ascontiguousarray(pixels), LatLon(bounds.top, bounds.left), LatLon(bounds.bottom, bounds.right))
