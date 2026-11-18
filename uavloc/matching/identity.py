"""Identity check: feed a satellite crop at the true pose as the "drone" map, so correction must be exact."""

from __future__ import annotations

from typing import Callable

import torch
from PIL import Image

from uavloc.data.drone_frames import DroneFrame
from uavloc.geo import LatLon
from uavloc.matching.correlator import FeatureExtractor, PositionCorrelator

DroneFromSatellite = Callable[[FeatureExtractor, LatLon, float, float, float], torch.Tensor]


class SatelliteAsDrone:
    """Wraps a feature extractor; its drone map is cut from the satellite side at `truth`."""

    def __init__(self, inner: FeatureExtractor, truth: LatLon, yaw_deg: float,
                 drone_from_satellite: DroneFromSatellite) -> None:
        self.inner, self.truth, self.yaw_deg = inner, truth, yaw_deg
        self.drone_from_satellite = drone_from_satellite

    def drone(self, image: Image.Image, width_m: float, height_m: float) -> torch.Tensor:
        return self.drone_from_satellite(self.inner, self.truth, self.yaw_deg, width_m, height_m)

    def satellite_patch(self, center: LatLon, yaw_deg: float, width_m: float, height_m: float) -> torch.Tensor:
        return self.inner.satellite_patch(center, yaw_deg, width_m, height_m)


def satellite_patch_as_drone(features: FeatureExtractor, truth: LatLon, yaw_deg: float,
                             width_m: float, height_m: float) -> torch.Tensor:
    """For extractors whose satellite and drone maps live on the same grid (class maps)."""
    return features.satellite_patch(truth, yaw_deg, width_m, height_m)


class SatelliteAsDroneCorrelator(PositionCorrelator):
    """Correlator whose drone map is the satellite's own feature map cut at the frame's true pose.

    A self-match: the query is a rotated crop of the raster the search correlates against, so the
    only error left is the corrector's own geometry (window reach, resampling, match-grid
    quantisation, peak choice, gate). Works with `benchmark_draws` over many frames, unlike
    `SatelliteAsDrone`, which bakes one frame's pose into the extractor.
    """

    def drone_map(self, frame: DroneFrame) -> torch.Tensor:
        if frame.name not in self._drone_maps:
            self._drone_maps[frame.name] = self.features.satellite_patch(
                frame.position, frame.yaw_deg, *self.footprint_m(frame))
        return self._drone_maps[frame.name]
