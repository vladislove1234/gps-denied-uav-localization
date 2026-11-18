"""Synthetic class worlds for filter acceptance checks (027).

A class-id raster is rendered as flat colours for the camera, and `ColourClassFeatures`
reads the colours back as one-hot class maps, so a check controls exactly what the
heatmap can and cannot tell apart (a single road, two identical blocks, nothing).
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from uavloc.data.satellite import GeoImage
from uavloc.geo import LatLon, offset_latlon

CLASS_COLOURS = np.array([[60, 120, 60], [200, 60, 60], [230, 230, 230], [20, 70, 20], [180, 160, 90], [40, 60, 200]],
                         np.uint8)   # background, building, track, woody, field, water


def class_world(class_ids: np.ndarray, center: LatLon, mpp: float) -> tuple[GeoImage, GeoImage]:
    """`(colours, one-hot probabilities)` of an `(H, W)` class-id raster centred on `center`."""
    height_m, width_m = class_ids.shape[0] * mpp, class_ids.shape[1] * mpp
    top_left = offset_latlon(center, -width_m / 2, height_m / 2)
    bottom_right = offset_latlon(center, width_m / 2, -height_m / 2)
    one_hot = np.eye(len(CLASS_COLOURS), dtype=np.float32)[class_ids]
    return GeoImage(CLASS_COLOURS[class_ids], top_left, bottom_right), GeoImage(one_hot, top_left, bottom_right)


def paint(class_ids: np.ndarray, mpp: float, class_id: int, east_m: tuple[float, float],
          north_m: tuple[float, float]) -> None:
    """Fill the box `east_m` x `north_m` (metres from the raster centre) with `class_id`, in place."""
    height, width = class_ids.shape
    cols = [int(round(width / 2 + e / mpp)) for e in east_m]
    rows = [int(round(height / 2 - n / mpp)) for n in north_m]
    class_ids[max(min(rows), 0):max(rows), max(min(cols), 0):max(cols)] = class_id


class ColourClassFeatures:
    """Drone side: nearest class colour → one-hot; satellite side: a crop of the one-hot world."""

    def __init__(self, probabilities: GeoImage, match_gsd_m: float) -> None:
        self.probabilities = probabilities
        self.match_gsd_m = match_gsd_m

    def drone(self, image: Image.Image, width_m: float, height_m: float) -> torch.Tensor:
        size = (round(width_m / self.match_gsd_m), round(height_m / self.match_gsd_m))
        rgb = np.asarray(image.resize(size, Image.NEAREST)).astype(np.int32)
        distance = ((rgb[..., None, :] - CLASS_COLOURS.astype(np.int32)) ** 2).sum(-1)
        one_hot = np.eye(len(CLASS_COLOURS), dtype=np.float32)[distance.argmin(-1)]
        return torch.from_numpy(one_hot.transpose(2, 0, 1).copy())[None]

    def satellite_patch(self, center: LatLon, yaw_deg: float, width_m: float, height_m: float) -> torch.Tensor:
        patch = self.probabilities.crop_rotated(center, yaw_deg, width_m, height_m, out_mpp=self.match_gsd_m)
        return torch.from_numpy(patch.transpose(2, 0, 1).copy())[None]
