"""Class-map and blind-map rasters as PNG data URLs for `L.imageOverlay`. No Qt, so it can be tested."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from uavloc.data.satellite import GeoImage
from uavloc.features.classes import BACKGROUND
from uavloc.plotting.correlation import SEG_PALETTE

CLASS_MAX_SIDE_PX = 1600         # the class map can be ~10k px; the screen cannot show that
BLIND_MAX_SIDE_PX = 800          # the blind map has one cell per sample spacing, so it is already coarse
OPAQUE = 255
TRANSPARENT = 0
BLIND_COLOUR = (214, 39, 40)     # same red as the blind stretches on the route
CLEAR_COLOUR = (44, 160, 44)


@dataclass(frozen=True)
class MapOverlay:
    data_url: str
    south: float
    west: float
    north: float
    east: float
    width_px: int
    height_px: int

    @property
    def leaflet_bounds(self) -> list[list[float]]:
        return [[self.south, self.west], [self.north, self.east]]


def png_data_url(rgba: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(rgba).save(buffer, format="PNG", optimize=False)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def overlay_of(raster: GeoImage, rgba: np.ndarray) -> MapOverlay:
    top_left, bottom_right = raster.top_left, raster.bottom_right
    height_px, width_px = rgba.shape[:2]
    return MapOverlay(png_data_url(rgba), bottom_right.lat, top_left.lon, top_left.lat, bottom_right.lon,
                      width_px, height_px)


def class_overlay(class_map: GeoImage, max_side_px: int = CLASS_MAX_SIDE_PX) -> MapOverlay:
    """Most likely class per pixel in the segmenter palette; background is transparent so the imagery shows."""
    stride = max(-(-max(class_map.pixels.shape[:2]) // max_side_px), 1)
    labels = class_map.pixels[::stride, ::stride].argmax(-1)
    rgba = np.empty((*labels.shape, 4), np.uint8)
    rgba[..., :3] = SEG_PALETTE[labels]
    rgba[..., 3] = np.where(labels == BACKGROUND, TRANSPARENT, OPAQUE)
    return overlay_of(class_map, rgba)


def blind_overlay(blind_map: GeoImage, max_side_px: int = BLIND_MAX_SIDE_PX) -> MapOverlay:
    """Red where a fix is unlikely, green where it is likely, smoothly interpolated between samples."""
    blind = blind_map.pixels[..., 0].astype(np.float32)
    scale = max_side_px / max(blind.shape)
    if scale > 1:
        size = (round(blind.shape[1] * scale), round(blind.shape[0] * scale))
        blind = cv2.resize(blind, size, interpolation=cv2.INTER_LINEAR)
    blind = np.clip(blind, 0.0, 1.0)[..., None]
    rgb = blind * np.array(BLIND_COLOUR, np.float32) + (1 - blind) * np.array(CLEAR_COLOUR, np.float32)
    rgba = np.empty((*blind.shape[:2], 4), np.uint8)
    rgba[..., :3] = np.round(rgb).astype(np.uint8)
    rgba[..., 3] = OPAQUE
    return overlay_of(blind_map, rgba)
