from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

from tools.route_planner.overlays import (BLIND_COLOUR, CLEAR_COLOUR, blind_overlay, class_overlay)
from uavloc.data.satellite import GeoImage
from uavloc.features.classes import BACKGROUND, CLASSES
from uavloc.geo import LatLon
from uavloc.plotting.correlation import SEG_PALETTE

TOP_LEFT, BOTTOM_RIGHT = LatLon(50.0, 24.0), LatLon(49.99, 24.02)
BUILDING = CLASSES.index("building")


def decode(data_url: str) -> np.ndarray:
    prefix = "data:image/png;base64,"
    assert data_url.startswith(prefix)
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(data_url[len(prefix):]))))


def test_class_overlay_bounds_follow_the_raster_and_size_is_capped() -> None:
    pixels = np.zeros((900, 1500, len(CLASSES)), np.uint8)
    overlay = class_overlay(GeoImage(pixels, TOP_LEFT, BOTTOM_RIGHT), max_side_px=300)
    assert overlay.leaflet_bounds == [[49.99, 24.0], [50.0, 24.02]]
    assert max(overlay.width_px, overlay.height_px) <= 300
    assert decode(overlay.data_url).shape == (overlay.height_px, overlay.width_px, 4)


def test_class_overlay_colours_classes_and_hides_background() -> None:
    pixels = np.zeros((10, 20, len(CLASSES)), np.uint8)
    pixels[..., BACKGROUND] = 255
    pixels[:, 10:, BACKGROUND] = 0
    pixels[:, 10:, BUILDING] = 255
    rgba = decode(class_overlay(GeoImage(pixels, TOP_LEFT, BOTTOM_RIGHT)).data_url)
    assert rgba[0, 0, 3] == 0
    assert tuple(rgba[0, 15]) == (*SEG_PALETTE[BUILDING], 255)


def test_small_class_map_is_not_resampled() -> None:
    overlay = class_overlay(GeoImage(np.zeros((30, 40, len(CLASSES)), np.uint8), TOP_LEFT, BOTTOM_RIGHT))
    assert (overlay.height_px, overlay.width_px) == (30, 40)


def test_blind_overlay_is_red_where_blind_green_where_clear() -> None:
    blind = np.zeros((20, 20, 1), np.float32)
    blind[:, 10:] = 1.0
    overlay = blind_overlay(GeoImage(blind, TOP_LEFT, BOTTOM_RIGHT), max_side_px=20)
    rgba = decode(overlay.data_url)
    assert tuple(rgba[10, 1, :3]) == CLEAR_COLOUR
    assert tuple(rgba[10, 18, :3]) == BLIND_COLOUR
    assert overlay.leaflet_bounds == [[49.99, 24.0], [50.0, 24.02]]


def test_blind_overlay_is_upsampled_smoothly() -> None:
    blind = np.zeros((10, 10, 1), np.float32)
    blind[:, 5:] = 1.0
    overlay = blind_overlay(GeoImage(blind, TOP_LEFT, BOTTOM_RIGHT), max_side_px=100)
    assert (overlay.height_px, overlay.width_px) == (100, 100)
    middle = decode(overlay.data_url)[50, :, 0]
    assert middle[0] < middle[50] < middle[-1]
