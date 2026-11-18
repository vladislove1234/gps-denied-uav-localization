"""What the real-image thesis figures draw: simulated frames, class maps, and what one filter update saw.

Compute only: the pictures are drawn by `uavloc.plotting.thesis_images`.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Sequence, TypeVar

import numpy as np
import pandas as pd
from PIL import Image
from torch import nn

from uavloc.data.rasters import load_raster
from uavloc.data.satellite import GeoImage
from uavloc.features.classes import CLASSES
from uavloc.features.segmentation import class_probabilities
from uavloc.filtering.belief import observe
from uavloc.filtering.particles import ParticleFilter, ParticleFilterConfig
from uavloc.geo import offset_latlon
from uavloc.matching.correlator import PositionCorrelator
from uavloc.matching.window import SearchWindow, patch_size_m, to_heading_up
from uavloc.simulation.camera import CameraConfig, DroneCamera, RasterCamera
from uavloc.simulation.correctors import RenderedFrame
from uavloc.simulation.estimators import FixRecord, SimulatedFrames
from uavloc.simulation.ins import InsSimulator
from uavloc.simulation.route import Pose, sample_route
from uavloc.simulation.schedule import EveryInterval
from uavloc.simulation.sequence import FlightRun, FlightSetConfig, FlightSpec


@dataclass(frozen=True)
class FrameSample:
    """One simulated onboard view and where it was taken."""

    village: str
    route_index: int
    pose: Pose
    view: Image.Image


def sample_frames(set_config: FlightSetConfig, spec: FlightSpec, camera: DroneCamera, count: int) -> list[FrameSample]:
    """`count` views evenly spaced along the route, leaving out its first and last tenth."""
    poses = sample_route(spec.waypoints, set_config.speed_mps, set_config.step_s, set_config.altitude_m)
    picks = np.linspace(len(poses) * 0.1, len(poses) * 0.9, count).astype(int)
    return [FrameSample(spec.village, spec.route_index, poses[i], camera.view(poses[i])) for i in picks]


def view_class_ids(model: nn.Module, view: Image.Image, footprint_m: tuple[float, float], gsd_m: float,
                   device: str) -> np.ndarray:
    """`(H, W)` class ids of a view, segmented at the model's training scale `gsd_m`, as the onboard system does."""
    size = (round(footprint_m[0] / gsd_m), round(footprint_m[1] / gsd_m))
    rgb = np.asarray(view.resize(size, Image.BILINEAR))
    return class_probabilities(model, rgb, device).argmax(0).astype(np.uint8)


def class_shares(class_ids: np.ndarray) -> np.ndarray:
    """`(len(CLASSES),)` share of the view that each class covers."""
    return np.bincount(class_ids.ravel(), minlength=len(CLASSES))[:len(CLASSES)] / class_ids.size


@dataclass(frozen=True)
class FixSnapshot:
    """One particle-filter update: the heatmap it read and the cloud before and after.

    Offsets are `(right_m, down_m)` in the heading-up patch centred on `window.center`, as in the heatmap.
    """

    window: SearchWindow
    patch_rgb: np.ndarray                  # satellite patch of the heatmap's window, heading-up
    patch_size_m: tuple[float, float]      # (width, height) of `patch_rgb`
    heatmap: np.ndarray                    # z-scored score of every drone-centre offset in the window
    heatmap_extent_m: tuple[float, float, float, float]   # (left, right, bottom, top)
    cloud_before_m: np.ndarray             # (N, 2)
    weights_before: np.ndarray             # (N,)
    cloud_after_m: np.ndarray
    weights_after: np.ndarray
    step: int = -1
    t_s: float = float("nan")
    truth_m: tuple[float, float] = (float("nan"), float("nan"))


class SnapshotParticleFilter(ParticleFilter):
    """The 027 particle filter, keeping a `FixSnapshot` of every update that read a heatmap."""

    def __init__(self, correlator: PositionCorrelator, config: ParticleFilterConfig, seed: int,
                 map_raster: GeoImage) -> None:
        super().__init__(correlator, config, seed)
        self.map_raster = map_raster
        self.snapshots: list[FixSnapshot] = []

    def _offsets_m(self, window: SearchWindow) -> np.ndarray:
        return to_heading_up(self.origin, window.center, self.particles, window.yaw_deg)

    def update(self, frame: RenderedFrame) -> FixRecord:
        surface, window, decision = observe(self.correlator, frame, self.origin, self.particles, self._mean(),
                                            self.config.window)
        before, weights_before = self._offsets_m(window), np.exp(self.log_weights)
        record = super().update(frame)
        if decision == "update":
            size_m = patch_size_m(self.correlator, self.correlator.footprint_m(frame), window.half_m)
            patch = self.map_raster.crop_rotated(window.center, window.yaw_deg, *size_m)
            self.snapshots.append(FixSnapshot(window, patch, size_m, surface.heatmap, surface.extent_m(), before,
                                              weights_before, self._offsets_m(window), np.exp(self.log_weights)))
        return record


def fly_with_snapshots(set_config: FlightSetConfig, spec: FlightSpec, drone_raster: GeoImage,
                       estimator: SnapshotParticleFilter, sigma0_m: float) -> list[FixSnapshot]:
    """Fly one route with `estimator` and return its snapshots, each with the step, time and true offset."""
    poses = sample_route(spec.waypoints, set_config.speed_mps, set_config.step_s, set_config.altitude_m)
    camera = RasterCamera(drone_raster, CameraConfig(set_config.fov_w_deg))
    run = FlightRun(poses[0], InsSimulator(set_config.ins, spec.seed), estimator,
                    EveryInterval(set_config.correction_interval_s), sigma0_m)
    for step, pose in enumerate(poses):
        known = len(estimator.snapshots)
        run.step(pose, camera)
        if len(estimator.snapshots) > known:
            last = estimator.snapshots[-1]
            truth_m = PositionCorrelator.patch_offset_m(last.window.center, pose.position, last.window.yaw_deg)
            estimator.snapshots[-1] = replace(last, step=step, t_s=pose.t_s, truth_m=truth_m)
    return estimator.snapshots


def spread_m(cloud_m: np.ndarray, weights: np.ndarray) -> float:
    """Weighted standard deviation of a cloud around its weighted mean, root of the summed axis variances."""
    mean = weights @ cloud_m
    return float(np.sqrt((weights[:, None] * (cloud_m - mean) ** 2).sum()))


def snapshot_true_cell_inside(snapshot: FixSnapshot) -> bool:
    """The true position lies inside the heatmap's window."""
    left, right, bottom, top = snapshot.heatmap_extent_m
    x, y = snapshot.truth_m
    return bool(left <= x <= right and top <= y <= bottom)


T = TypeVar("T")


def cached_pickle(path: Path, build: Callable[[], T], refresh: bool) -> T:
    """`build()` once, kept at `path`; an existing file is read back unless `refresh`."""
    if path.exists() and not refresh:
        return pickle.loads(path.read_bytes())
    value = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(value))
    return value


@dataclass(frozen=True)
class PoolFrame:
    """A simulated view with its class map from the onboard segmenter and the share each class covers."""

    village: str
    route_index: int
    pose: Pose
    view: np.ndarray        # (H, W, 3) uint8
    class_ids: np.ndarray   # (H, W)
    shares: np.ndarray      # (len(CLASSES),)


def build_frame_pool(set_config: FlightSetConfig, specs: Sequence[FlightSpec], model: nn.Module, per_route: int,
                     gsd_m: float, device: str) -> list[PoolFrame]:
    """Views along the first route of every village in `specs`, each segmented by `model`."""
    pool, seen = [], set()
    for spec in specs:
        if spec.village in seen:
            continue
        seen.add(spec.village)
        camera = RasterCamera(load_raster({"kind": "multidate", "place": spec.village, "date": set_config.drone_date}),
                              CameraConfig(set_config.fov_w_deg))
        footprint = camera.footprint_m(set_config.altitude_m)
        for sample in sample_frames(set_config, spec, camera, per_route):
            ids = view_class_ids(model, sample.view, footprint, gsd_m, device)
            pool.append(PoolFrame(spec.village, spec.route_index, sample.pose, np.asarray(sample.view), ids,
                                  class_shares(ids)))
    return pool


def frames_with_most(pool: Sequence[PoolFrame], class_id: int, count: int) -> list[PoolFrame]:
    """The `count` frames covering most of `class_id`, at most one per village."""
    picked, villages = [], set()
    for frame in sorted(pool, key=lambda f: -f.shares[class_id]):
        if frame.village not in villages:
            villages.add(frame.village)
            picked.append(frame)
        if len(picked) == count:
            break
    return picked


def most_mixed_frame(pool: Sequence[PoolFrame], class_ids: Sequence[int], required_ids: Sequence[int],
                     min_share: float) -> PoolFrame:
    """The frame whose shares over `class_ids` are most even, among frames where each `required_ids` class covers `min_share`."""
    def entropy(frame: PoolFrame) -> float:
        share = frame.shares[list(class_ids)]
        share = share / share.sum()
        return float(-(share[share > 0] * np.log(share[share > 0])).sum())

    eligible = [f for f in pool if all(f.shares[c] >= min_share for c in required_ids)]
    return max(eligible, key=entropy)


def pick_typical_flights(table: pd.DataFrame, n_typical: int, good_quantile: float) -> list[tuple[str, int, str]]:
    """`(village, route_index, reason)`: the `n_typical` flights whose end error is nearest the median (one per
    village), then the flight nearest the `good_quantile` end error."""
    median = table["end_error_m"].median()
    good = table["end_error_m"].quantile(good_quantile)
    picks: list[tuple[str, int, str]] = []
    for target, count, reason in ((median, n_typical, "typical"), (good, 1, "good")):
        ranked = table.assign(gap=(table["end_error_m"] - target).abs()).sort_values("gap")
        villages = {village for village, _, _ in picks}
        for row in ranked.itertuples():
            if row.village not in villages:
                picks.append((row.village, int(row.route_index), reason))
                villages.add(row.village)
                count -= 1
            if count == 0:
                break
    return picks


@dataclass(frozen=True)
class SingleFix:
    """One correction of a frame whose prior is `prior_offset_en_m` away from the truth."""

    patch_rgb: np.ndarray
    patch_size_m: tuple[float, float]
    heatmap: np.ndarray
    heatmap_extent_m: tuple[float, float, float, float]
    true_m: tuple[float, float]
    found_m: tuple[float, float]

    @property
    def error_m(self) -> float:
        return float(np.hypot(self.true_m[0] - self.found_m[0], self.true_m[1] - self.found_m[1]))


def correct_once(correlator: PositionCorrelator, map_raster: GeoImage, frame: PoolFrame, altitude_m: float,
                 prior_offset_en_m: tuple[float, float]) -> SingleFix:
    """The correlator's heatmap and best offset for `frame`, searched around a prior shifted by `prior_offset_en_m`."""
    prior = offset_latlon(frame.pose.position, *prior_offset_en_m)
    rendered = SimulatedFrames().frame(Image.fromarray(frame.view), prior, frame.pose.yaw_deg, altitude_m)
    correction = correlator.correct(rendered, prior)
    true_m = PositionCorrelator.patch_offset_m(prior, frame.pose.position, frame.pose.yaw_deg)
    patch = map_raster.crop_rotated(prior, frame.pose.yaw_deg, *correction.patch_size_m)
    return SingleFix(patch, correction.patch_size_m, correction.heatmap, correction.heatmap_extent_m, true_m,
                     correction.offset_m)


def satellite_class_ids(features: object, pose: Pose, footprint_m: tuple[float, float]) -> np.ndarray:
    """`(H, W)` class ids of the satellite ground under the footprint at `pose`, from the satellite's class map."""
    probabilities = features.satellite_patch(pose.position, pose.yaw_deg, *footprint_m)[0]
    return probabilities.argmax(0).numpy().astype(np.uint8)
