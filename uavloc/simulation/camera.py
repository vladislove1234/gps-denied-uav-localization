"""A nadir drone camera rendered from a georeferenced raster."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from PIL import Image

from uavloc.data.satellite import GeoImage
from uavloc.geo import footprint_m
from uavloc.matching.correlator import CAMERA_ASPECT_W_OVER_H
from uavloc.simulation.route import Pose

ViewTransform = Callable[[Image.Image], Image.Image]   # e.g. blur, brightness, a style change


class DroneCamera(Protocol):
    def view(self, pose: Pose) -> Image.Image: ...

    def footprint_m(self, altitude_m: float) -> tuple[float, float]: ...


@dataclass(frozen=True)
class CameraConfig:
    fov_w_deg: float = 60.0
    aspect_w_over_h: float = CAMERA_ASPECT_W_OVER_H
    image_width_px: int | None = None   # None: the raster's own resolution (no invented detail)


class RasterCamera:
    """Heading-up view at the true pose. The raster's GSD bounds the detail, whatever `image_width_px` is."""

    def __init__(self, raster: GeoImage, config: CameraConfig, transform: ViewTransform | None = None) -> None:
        self.raster = raster
        self.config = config
        self.transform = transform

    def footprint_m(self, altitude_m: float) -> tuple[float, float]:
        return footprint_m(altitude_m, self.config.fov_w_deg, self.config.aspect_w_over_h)

    def view(self, pose: Pose) -> Image.Image:
        width_m, height_m = self.footprint_m(pose.altitude_m)
        out_mpp = width_m / self.config.image_width_px if self.config.image_width_px else None
        image = Image.fromarray(self.raster.crop_rotated(pose.position, pose.yaw_deg, width_m, height_m, out_mpp))
        return self.transform(image) if self.transform else image
