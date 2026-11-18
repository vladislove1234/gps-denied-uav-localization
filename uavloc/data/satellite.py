"""A north-up georeferenced raster: RGB imagery or per-class probability maps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from uavloc.geo import METERS_PER_DEG, LatLon, meters_per_deg_lon, rotate_to_north_up

CV2_MAX_WARP_CHANNELS = 4


@dataclass(frozen=True)
class GeoImage:
    """`pixels` is `(H, W, C)`; lat/lon maps linearly onto pixels between the corners."""

    pixels: np.ndarray
    top_left: LatLon
    bottom_right: LatLon

    @classmethod
    def load(cls, path: Path, top_left: LatLon, bottom_right: LatLon) -> GeoImage:
        return cls(np.asarray(Image.open(path).convert("RGB")), top_left, bottom_right)

    @property
    def size_m(self) -> tuple[float, float]:
        mid_lat = (self.top_left.lat + self.bottom_right.lat) / 2
        return ((self.bottom_right.lon - self.top_left.lon) * meters_per_deg_lon(mid_lat),
                (self.top_left.lat - self.bottom_right.lat) * METERS_PER_DEG)

    @property
    def mpp_xy(self) -> tuple[float, float]:
        width_m, height_m = self.size_m
        return width_m / self.pixels.shape[1], height_m / self.pixels.shape[0]

    @property
    def mpp(self) -> float:
        return sum(self.mpp_xy) / 2

    def to_px(self, point: LatLon) -> tuple[float, float]:
        h, w = self.pixels.shape[:2]
        return ((point.lon - self.top_left.lon) / (self.bottom_right.lon - self.top_left.lon) * w,
                (self.top_left.lat - point.lat) / (self.top_left.lat - self.bottom_right.lat) * h)

    def to_latlon(self, x: float, y: float) -> LatLon:
        h, w = self.pixels.shape[:2]
        return LatLon(self.top_left.lat - y / h * (self.top_left.lat - self.bottom_right.lat),
                      self.top_left.lon + x / w * (self.bottom_right.lon - self.top_left.lon))

    def with_pixels(self, pixels: np.ndarray) -> GeoImage:
        """Same ground, different raster (e.g. class probabilities at another resolution)."""
        return GeoImage(pixels, self.top_left, self.bottom_right)

    def resampled(self, mpp: float) -> GeoImage:
        """Isotropic resample to `mpp` metres per pixel; area-averaged when shrinking."""
        width_m, height_m = self.size_m
        size = (round(width_m / mpp), round(height_m / mpp))
        return self.with_pixels(cv2.resize(self.pixels, size, interpolation=cv2.INTER_AREA))

    def window(self, center: LatLon, half_size_m: float) -> GeoImage:
        """Axis-aligned sub-raster of about `2 * half_size_m` around `center`, clipped to the raster.

        A view, not a copy: warping a small patch out of a large many-channel raster is cheap from here.
        """
        h, w = self.pixels.shape[:2]
        cx, cy = self.to_px(center)
        mpp_x, mpp_y = self.mpp_xy
        x0, x1 = max(int(cx - half_size_m / mpp_x), 0), min(int(cx + half_size_m / mpp_x) + 1, w)
        y0, y1 = max(int(cy - half_size_m / mpp_y), 0), min(int(cy + half_size_m / mpp_y) + 1, h)
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"{center} is outside the raster")
        return GeoImage(self.pixels[y0:y1, x0:x1], self.to_latlon(x0, y0), self.to_latlon(x1, y1))

    def crop_rotated(self, center: LatLon, yaw_deg: float, width_m: float, height_m: float,
                     out_mpp: float | None = None, nearest: bool = False, allow_outside: bool = False) -> np.ndarray:
        """Heading-up patch centred on `center`, sampled at `out_mpp` (default: own mpp).

        `nearest` keeps values unmixed, for class-id rasters. Raises ValueError when any
        corner leaves the raster, unless `allow_outside`, which fills the outside with zeros.
        """
        out_mpp = out_mpp or self.mpp
        out_w, out_h = round(width_m / out_mpp), round(height_m / out_mpp)
        matrix = self._patch_to_source(center, yaw_deg, out_w, out_h, out_mpp)
        corners = np.array([[0, 0, 1], [out_w, 0, 1], [0, out_h, 1], [out_w, out_h, 1]], float) @ matrix.T
        h, w = self.pixels.shape[:2]
        outside = (corners[:, 0] < 0).any() or (corners[:, 0] > w).any() or (corners[:, 1] < 0).any() or (corners[:, 1] > h).any()
        if outside and not allow_outside:
            raise ValueError(f"patch at {center} leaves the raster")
        return _warp(self.pixels, matrix, (out_w, out_h), nearest)

    def _patch_to_source(self, center: LatLon, yaw_deg: float, out_w: int, out_h: int,
                         out_mpp: float) -> np.ndarray:
        """2x3 affine from patch pixels to source pixels."""
        cx, cy = self.to_px(center)
        mpp_x, mpp_y = self.mpp_xy
        (a, b), (c, d) = (rotate_to_north_up(1, 0, yaw_deg), rotate_to_north_up(0, 1, yaw_deg))
        # columns: source px per patch px along patch u (right) and v (down)
        m = np.array([[a * out_mpp / mpp_x, c * out_mpp / mpp_x],
                      [b * out_mpp / mpp_y, d * out_mpp / mpp_y]])
        t = np.array([cx, cy]) - m @ np.array([out_w / 2, out_h / 2])
        return np.c_[m, t]


def _warp(pixels: np.ndarray, patch_to_source: np.ndarray, size: tuple[int, int],
          nearest: bool = False) -> np.ndarray:
    flags = (cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR) | cv2.WARP_INVERSE_MAP
    if pixels.ndim == 2 or pixels.shape[2] <= CV2_MAX_WARP_CHANNELS:
        return cv2.warpAffine(pixels, patch_to_source, size, flags=flags)
    return np.stack([cv2.warpAffine(np.ascontiguousarray(pixels[..., i]), patch_to_source, size, flags=flags)
                     for i in range(pixels.shape[2])], axis=-1)
