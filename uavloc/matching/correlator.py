"""Correct a position prior by correlating a drone feature map against a satellite patch.

Port of the 2025 `PositionCorrelator`, with the feature extraction injected so
the same geometry serves any representation (CNN features, class maps).

A correction is two steps: `PositionCorrelator.surface` scores every candidate
offset, `decide` turns the scores into a move. The split lets a benchmark tune
the decision (temperature, gate) without recomputing the scores.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, Protocol

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from uavloc.data.drone_frames import DroneFrame
from uavloc.geo import LatLon, east_north_m, footprint_m, offset_latlon, rotate_to_heading_up, rotate_to_north_up

Z_SCORE_EPS = 1e-6
NCC_EPS = 1e-6
CAMERA_ASPECT_W_OVER_H = 16 / 9
MATCHING_MODES = ("cosine", "weighted-ncc", "learned")
MIN_PRESENT_SHARE = 0.01   # a class takes part only if it is the drone argmax on >= 1% of pixels
MIN_WINDOW_STD = 0.05      # a satellite window flatter than this in a class says nothing about position
WINDOWS = ("rect", "round")
GATES = ("none", "full", "accept-reject")
SECOND_PEAK_EXCLUSION_M = 5.0   # a rival peak must lie this far from the chosen cell
HALF_TURN_DEG = 180.0
POSE_SEARCH_MATCHING = ("cosine", "weighted-ncc")   # modes with a raw score comparable across poses
POSE_SCHEDULES = ("greedy", "joint")

FeaturesHook = Callable[[torch.Tensor, torch.Tensor], None]  # (drone map, satellite map), uncropped
# (drone, satellite, config, round-window mask) -> (z-scored heatmap, coverage); maps are cropped (1, C, H, W)
Matcher = Callable[[torch.Tensor, torch.Tensor, "CorrelatorConfig", "np.ndarray | None"], tuple[np.ndarray, float]]
_MATCHERS: dict[str, Matcher] = {}


def register_matching(name: str, matcher: Matcher) -> None:
    """Add a scoring mode beyond `MATCHING_MODES`; experiment modules call this at import, before building configs."""
    if name in MATCHING_MODES:
        raise ValueError(f"{name!r} is a built-in matching mode")
    _MATCHERS[name] = matcher


def matching_modes() -> tuple[str, ...]:
    return MATCHING_MODES + tuple(_MATCHERS)


# (cropped drone map, cropped satellite map, round-window mask or None) -> z-scored heatmap, NaN outside the mask
SurfaceScorer = Callable[[torch.Tensor, torch.Tensor, "np.ndarray | None"], np.ndarray]


class FeatureExtractor(Protocol):
    def drone(self, image: Image.Image, width_m: float, height_m: float) -> torch.Tensor: ...

    def satellite_patch(self, center: LatLon, yaw_deg: float, width_m: float, height_m: float) -> torch.Tensor: ...


@dataclass(frozen=True)
class CorrelatorConfig:
    search_scale: float
    fov_w_deg: float
    aspect_w_over_h: float = CAMERA_ASPECT_W_OVER_H
    sat_feature_crop_pct: float = 0.0
    drone_feature_crop_pct: float = 0.0
    matching: str = "cosine"
    class_weights: tuple[float, ...] | None = None  # per feature channel, for "weighted-ncc"
    window: str = "rect"                    # "rect": search_scale x footprint; "round": disc of search_radius_m
    search_radius_m: float | None = None    # "round" only: how far the drone centre may move in every direction
    prior_sigma_m: float | None = None      # INS uncertainty of the prior; None = flat prior (plain argmax)
    temperature: float = 1.0                # log p(cell) = z / temperature - |d|^2 / (2 sigma^2)
    gate: str = "none"                      # "full": accept / across-only / reject; "accept-reject": no across-only
    gate_s_max_m: float | None = None       # posterior std along an axis below which that axis is trusted
    gate_c_min: float | None = None         # min drone share of weighted classes
    # Pose search (003, `uavloc.matching.pose_search`): coarse heading offsets at scale 1, then the footprint
    # scales at the best coarse heading, then fine offsets around it. The defaults search translation only.
    heading_offsets_deg: tuple[float, ...] = (0.0,)        # added to the logged heading
    heading_fine_offsets_deg: tuple[float, ...] = (0.0,)   # added to the best coarse offset
    scales: tuple[float, ...] = (1.0,)                     # multiply the drone footprint from the logged altitude
    pose_schedule: str = "greedy"   # "joint": coarse offsets x every scale, then fine offsets at the best (003 amendment)
    fine_pass_top_k: int = 1        # "joint" only: fine offsets around this many best coarse pairs (003 amendment 2)

    @property
    def searches_pose(self) -> bool:
        return tuple(self.heading_offsets_deg) != (0.0,) or tuple(self.scales) != (1.0,)

    def __post_init__(self) -> None:
        if self.matching not in matching_modes():
            raise ValueError(f"matching must be one of {matching_modes()}, got {self.matching!r}")
        if self.matching == "weighted-ncc" and not self.class_weights:
            raise ValueError("weighted-ncc needs class_weights")
        if self.window not in WINDOWS:
            raise ValueError(f"window must be one of {WINDOWS}, got {self.window!r}")
        if self.window == "round" and not self.search_radius_m:
            raise ValueError("a round window needs search_radius_m")
        if self.gate not in GATES:
            raise ValueError(f"gate must be one of {GATES}, got {self.gate!r}")
        if self.gate != "none" and (self.gate_s_max_m is None or self.gate_c_min is None):
            raise ValueError("a gate needs gate_s_max_m and gate_c_min")
        # the logged pose is always one of the hypotheses, so a pose search can only add to translation-only
        if 0.0 not in self.heading_offsets_deg or 0.0 not in self.heading_fine_offsets_deg or 1.0 not in self.scales:
            raise ValueError("heading offsets must include 0 and scales must include 1.0")
        if min(self.scales) <= 0:
            raise ValueError(f"scales must be positive, got {self.scales}")
        if self.searches_pose and self.matching not in POSE_SEARCH_MATCHING:
            raise ValueError(f"pose search needs matching in {POSE_SEARCH_MATCHING}, got {self.matching!r}")
        if self.pose_schedule not in POSE_SCHEDULES:
            raise ValueError(f"pose_schedule must be one of {POSE_SCHEDULES}, got {self.pose_schedule!r}")
        if self.fine_pass_top_k < 1:
            raise ValueError(f"fine_pass_top_k must be >= 1, got {self.fine_pass_top_k}")


@dataclass(frozen=True)
class SearchSurface:
    """The z-scored score of every candidate drone-centre offset, before any decision.

    Offsets are `(right_m, down_m)` in the heading-up patch centred on the prior. Cells outside a
    round window are NaN.
    """

    prior: LatLon
    yaw_deg: float
    heatmap: np.ndarray
    coverage: float                           # drone share of classes with weight > 0; NaN for cosine matching
    sat_crop: tuple[int, int]                 # rows, cols dropped from each side of the satellite map
    drone_crop: tuple[int, int]
    drone_hw: tuple[int, int]                 # uncropped feature-map sizes
    sat_hw: tuple[int, int]
    patch_hw_m: tuple[float, float]
    heading_offset_deg: float = 0.0           # pose search: `yaw_deg` is the logged heading plus this
    scale: float = 1.0                        # pose search: drone footprint / footprint from the logged altitude
    hypotheses: int = 1                       # poses scored to find this surface
    skipped_hypotheses: int = 0               # poses the schedule asked for whose satellite patch left the raster

    def offset_m(self, cell: tuple[float, float]) -> tuple[float, float]:
        """Drone-centre offset from the patch centre for a (possibly fractional) `(row, col)`.

        Unlike 2025, adds back the cropped border cells, which the original dropped (a one-cell bias).
        """
        right_down = []
        for axis in (1, 0):
            centre_cells = self.sat_crop[axis] + cell[axis] - self.drone_crop[axis] + self.drone_hw[axis] / 2
            right_down.append((centre_cells - self.sat_hw[axis] / 2) * self.patch_hw_m[axis] / self.sat_hw[axis])
        return right_down[0], right_down[1]

    def cell_of_offset(self, offset_m: tuple[float, float]) -> tuple[float, float]:
        """Fractional `(row, col)` whose `offset_m` is `offset_m` (right, down): the inverse of `offset_m`."""
        cell = []
        for axis, value in ((0, offset_m[1]), (1, offset_m[0])):
            centre_cells = value * self.sat_hw[axis] / self.patch_hw_m[axis] + self.sat_hw[axis] / 2
            cell.append(centre_cells - self.sat_crop[axis] + self.drone_crop[axis] - self.drone_hw[axis] / 2)
        return cell[0], cell[1]

    def offset_grid_m(self) -> np.ndarray:
        """`(2, H, W)`: `offset_m` of every heatmap cell, right then down."""
        rows, cols = np.indices(self.heatmap.shape, dtype=np.float64)
        return np.stack(self.offset_m((rows, cols)))

    def extent_m(self) -> tuple[float, float, float, float]:
        """`(left, right, bottom, top)` of the heatmap, for imshow."""
        (left, top), (right, bottom) = (self.offset_m((-0.5, -0.5)),
                                        self.offset_m((self.heatmap.shape[0] - 0.5, self.heatmap.shape[1] - 0.5)))
        return left, right, bottom, top


@dataclass(frozen=True)
class Posterior:
    """`p(cell)` over the included cells, with the mean and covariance of the offset it implies (metres)."""

    probabilities: np.ndarray    # (H, W), 0 on excluded cells
    map_cell: tuple[int, int]
    mean_m: np.ndarray           # (2,) right, down
    cov_m2: np.ndarray           # (2, 2)

    @property
    def axes(self) -> tuple[float, float, np.ndarray, np.ndarray]:
        """`(s1, s2, e1, e2)`: std along the long and short axis (s1 >= s2, metres) and their unit vectors."""
        eigenvalues, eigenvectors = np.linalg.eigh(self.cov_m2)
        s2, s1 = np.sqrt(np.clip(eigenvalues, 0, None))
        return float(s1), float(s2), eigenvectors[:, 1], eigenvectors[:, 0]


@dataclass(frozen=True)
class Correction:
    """Everything is in the heading-up patch centred on the prior; offsets are `(right_m, down_m)`.

    `offset_m` is the move applied; `map_offset_m` is the posterior's best cell, which a gate may
    shorten (across-only) or drop (reject).
    """

    position: LatLon
    peak_z: float
    heatmap: np.ndarray
    offset_m: tuple[float, float]
    heatmap_extent_m: tuple[float, float, float, float]  # (left, right, bottom, top), for imshow
    patch_size_m: tuple[float, float]
    map_offset_m: tuple[float, float] = (0.0, 0.0)
    across_offset_m: tuple[float, float] = (0.0, 0.0)    # map_offset_m projected on the short axis
    decision: str = "accept"                             # "accept" | "across" | "reject"
    posterior: Posterior | None = None
    coverage: float = float("nan")
    second_peak_ratio: float = float("nan")
    heading_offset_deg: float = 0.0     # the winning pose; `position` is rotated back with logged heading + this
    scale: float = 1.0

    @property
    def uncertainty_m(self) -> tuple[float, float]:
        """`(s1, s2)`, the posterior std along its long and short axis."""
        s1, s2, _, _ = self.posterior.axes
        return s1, s2

    def heatmap_cell(self, offset_m: tuple[float, float]) -> tuple[int, int] | None:
        """`(row, col)` of the heatmap cell whose drone centre is nearest `offset_m`; None outside the search range."""
        left, right, bottom, top = self.heatmap_extent_m
        rows, cols = self.heatmap.shape
        col = int((offset_m[0] - left) / (right - left) * cols)
        row = int((offset_m[1] - top) / (bottom - top) * rows)
        return (row, col) if 0 <= row < rows and 0 <= col < cols else None


def crop_border(fmap: torch.Tensor, pct: float) -> tuple[torch.Tensor, int, int]:
    """Drop `pct` of rows/cols on each side of `(1, C, H, W)`; returns the map and the rows/cols dropped."""
    h, w = fmap.shape[2:]
    rows, cols = int(h * pct), int(w * pct)
    return fmap[:, :, rows:h - rows, cols:w - cols], rows, cols


def z_scored(corr: torch.Tensor, mask: np.ndarray | None = None) -> np.ndarray:
    """z-score over all cells, or over `mask` only, with NaN outside it."""
    if mask is None:
        return ((corr - corr.mean()) / (corr.std() + Z_SCORE_EPS)).numpy()
    inside = corr[torch.from_numpy(mask)]
    z = ((corr - inside.mean()) / (inside.std() + Z_SCORE_EPS)).numpy()
    return np.where(mask, z, np.nan).astype(z.dtype)


def correlation_heatmap(drone: torch.Tensor, satellite: torch.Tensor, mask: np.ndarray | None = None) -> np.ndarray:
    """z-scored sliding dot product of per-pixel unit feature vectors, `(H_s - H_d + 1, W_s - W_d + 1)`."""
    kernel = F.normalize(drone.float().cpu(), dim=1)
    corr = F.conv2d(F.normalize(satellite.float().cpu(), dim=1), kernel)[0, 0]
    return z_scored(corr, mask)


def class_ncc(drone_channel: torch.Tensor, satellite_channel: torch.Tensor) -> torch.Tensor:
    """Zero-mean normalised cross-correlation of one class channel, `(H_s - H_d + 1, W_s - W_d + 1)`.

    Windows whose satellite std is below `MIN_WINDOW_STD` score 0.
    """
    kernel = drone_channel - drone_channel.mean()
    n = kernel.numel()
    sat = satellite_channel[None, None]
    ones = torch.ones(1, 1, *kernel.shape)
    window_mean = F.conv2d(sat, ones)[0, 0] / n
    window_std = (F.conv2d(sat ** 2, ones)[0, 0] / n - window_mean ** 2).clamp(min=0).sqrt()
    ncc = F.conv2d(sat, kernel[None, None])[0, 0] / (kernel.norm() * window_std * n ** 0.5 + NCC_EPS)
    return torch.where(window_std >= MIN_WINDOW_STD, ncc, torch.zeros_like(ncc))


def drone_class_shares(drone: torch.Tensor) -> np.ndarray:
    """Share of drone pixels where each class is the argmax, `(C,)`."""
    drone = drone[0].float().cpu()
    return (torch.bincount(drone.argmax(0).flatten(), minlength=drone.shape[0]).float() / drone[0].numel()).numpy()


def class_ncc_maps(drone: torch.Tensor, satellite: torch.Tensor, class_weights: tuple[float, ...]) -> dict[int, np.ndarray]:
    """Per-class NCC surfaces for the classes that take part: weight > 0 and present in the drone map."""
    shares = drone_class_shares(drone)
    drone, satellite = drone[0].float().cpu(), satellite[0].float().cpu()
    return {c: class_ncc(drone[c], satellite[c]).numpy() for c, weight in enumerate(class_weights)
            if weight > 0 and shares[c] >= MIN_PRESENT_SHARE}


def weighted_ncc_heatmap(drone: torch.Tensor, satellite: torch.Tensor, class_weights: tuple[float, ...],
                         mask: np.ndarray | None = None) -> np.ndarray:
    """z-scored weighted mean of per-class NCC over the classes present in the drone map.

    All zeros when no weighted class is present: the caller then keeps the prior.
    """
    maps = class_ncc_maps(drone, satellite, class_weights)
    out_shape = (satellite.shape[2] - drone.shape[2] + 1, satellite.shape[3] - drone.shape[3] + 1)
    if not maps:
        zeros = np.zeros(out_shape, np.float32)
        return zeros if mask is None else np.where(mask, zeros, np.nan).astype(np.float32)
    weight_sum = sum(class_weights[c] for c in maps)
    return z_scored(torch.from_numpy(sum(class_weights[c] * m for c, m in maps.items()) / weight_sum), mask)


def weighted_coverage(drone: torch.Tensor, class_weights: tuple[float, ...]) -> float:
    """Share of drone pixels whose argmax class has weight > 0."""
    shares = drone_class_shares(drone)
    return float(sum(shares[c] for c, weight in enumerate(class_weights) if weight > 0))


def posterior(surface: SearchSurface, config: CorrelatorConfig) -> Posterior:
    """`log p(cell) = z / temperature - |d|^2 / (2 sigma^2)`, normalised over the included cells.

    With `prior_sigma_m=None` the best cell is the plain argmax of `z`, as before 007.
    """
    included = np.isfinite(surface.heatmap)
    z = np.where(included, surface.heatmap, -np.inf).astype(np.float64)
    offsets = surface.offset_grid_m()
    log_p = z / config.temperature
    if config.prior_sigma_m is not None:
        log_p = log_p - (offsets ** 2).sum(0) / (2 * config.prior_sigma_m ** 2)
    best = np.argmax(z if config.prior_sigma_m is None else log_p)
    p = np.exp(log_p - log_p.max())
    p /= p.sum()
    mean = (p * offsets).sum((1, 2))
    centred = offsets - mean[:, None, None]
    cov = np.einsum("ihw,jhw,hw->ij", centred, centred, p)
    return Posterior(p, np.unravel_index(best, z.shape), mean, cov)


def second_peak_ratio(surface: SearchSurface, cell: tuple[int, int]) -> float:
    """Best z more than `SECOND_PEAK_EXCLUSION_M` from `cell`, divided by z at `cell`; NaN if none or z <= 0."""
    offsets = surface.offset_grid_m()
    distance = np.hypot(*(offsets - offsets[:, cell[0], cell[1]][:, None, None]))
    rivals = surface.heatmap[(distance > SECOND_PEAK_EXCLUSION_M) & np.isfinite(surface.heatmap)]
    peak = surface.heatmap[cell]
    return float(rivals.max() / peak) if rivals.size and peak > 0 else float("nan")


def gate_decision(post: Posterior, coverage: float, config: CorrelatorConfig) -> str:
    """accept / across / reject, in the pre-registered order (007 proposal, Method 4)."""
    if config.gate == "none":
        return "accept"
    s1, s2, _, _ = post.axes
    if coverage < config.gate_c_min:
        return "reject"
    if s1 <= config.gate_s_max_m:
        return "accept"
    if s2 <= config.gate_s_max_m and config.gate == "full":
        return "across"
    return "reject"


def shifted(prior: LatLon, offset_m: tuple[float, float], yaw_deg: float) -> LatLon:
    """The position a heading-up `(right_m, down_m)` move from `prior` lands on."""
    dx_m, dy_m = rotate_to_north_up(*offset_m, yaw_deg)
    return offset_latlon(prior, dx_m, -dy_m)


def decide(surface: SearchSurface, config: CorrelatorConfig) -> Correction:
    """Turn a scored surface into a move. A flat surface (nothing to match on) keeps the prior."""
    heatmap = surface.heatmap
    flat = np.nanmax(heatmap) == np.nanmin(heatmap)
    post = posterior(surface, config)
    row, col = post.map_cell
    map_offset = surface.offset_m((row, col))
    _, _, _, e2 = post.axes
    across = float(np.dot(map_offset, e2))
    across_offset = (across * e2[0], across * e2[1])
    decision = "reject" if flat else gate_decision(post, surface.coverage, config)
    applied = {"accept": map_offset, "across": across_offset, "reject": (0.0, 0.0)}[decision]
    if flat and config.gate == "none":
        # kept prior as before 007: the centre cell, not a literal zero, so old runs reproduce exactly
        applied = surface.offset_m(((heatmap.shape[0] - 1) / 2, (heatmap.shape[1] - 1) / 2))
    return Correction(shifted(surface.prior, applied, surface.yaw_deg), 0.0 if flat else float(heatmap[row, col]), heatmap,
                      applied, surface.extent_m(), (surface.patch_hw_m[1], surface.patch_hw_m[0]),
                      map_offset, across_offset, decision, post, surface.coverage,
                      float("nan") if flat else second_peak_ratio(surface, (row, col)),
                      surface.heading_offset_deg, surface.scale)


def patch_size_for_footprint_m(config: CorrelatorConfig, footprint: tuple[float, float]) -> tuple[float, float]:
    """Satellite search patch `(width_m, height_m)` for a drone footprint.

    Round window: after both border crops the drone centre can move `search_radius_m` along each axis.
    """
    width_m, height_m = footprint
    if config.window == "rect":
        return width_m * config.search_scale, height_m * config.search_scale
    kept_sat, kept_drone = 1 - 2 * config.sat_feature_crop_pct, 1 - 2 * config.drone_feature_crop_pct
    return ((width_m * kept_drone + 2 * config.search_radius_m) / kept_sat,
            (height_m * kept_drone + 2 * config.search_radius_m) / kept_sat)


def half_range_for_footprint_m(config: CorrelatorConfig, footprint: tuple[float, float]) -> tuple[float, float]:
    """How far the truth may sit from the prior along the footprint's width and height and still be found."""
    if config.window == "round":
        return config.search_radius_m, config.search_radius_m
    span = config.search_scale * (1 - 2 * config.sat_feature_crop_pct) - (1 - 2 * config.drone_feature_crop_pct)
    width_m, height_m = footprint
    return span * width_m / 2, span * height_m / 2


def ridge_bearing_deg(correction: Correction, yaw_deg: float) -> float:
    """Compass bearing of the posterior's long axis, in [0, 180)."""
    _, _, e1, _ = correction.posterior.axes
    east, south = rotate_to_north_up(e1[0], e1[1], yaw_deg)
    return math.degrees(math.atan2(east, -south)) % HALF_TURN_DEG


def agreement_map(drone: torch.Tensor, satellite: torch.Tensor) -> np.ndarray:
    """Per-pixel cosine of unit feature vectors for two aligned `(1, C, H, W)` maps.

    Its sum is the raw (pre-z-score) correlation score at that alignment.
    """
    return (F.normalize(drone.float().cpu(), dim=1) * F.normalize(satellite.float().cpu(), dim=1)).sum(1)[0].numpy()


class PositionCorrelator:
    def __init__(self, features: FeatureExtractor, config: CorrelatorConfig,
                 features_hook: FeaturesHook | None = None, scorer: SurfaceScorer | None = None) -> None:
        if (config.matching == "learned") != (scorer is not None):
            raise ValueError("a scorer is given exactly when matching is 'learned'")
        self.features = features
        self.config = config
        self.features_hook = features_hook
        self.scorer = scorer
        self._drone_maps: dict[tuple[str, float], torch.Tensor] = {}

    def footprint_m(self, frame: DroneFrame) -> tuple[float, float]:
        return footprint_m(frame.altitude_m, self.config.fov_w_deg, self.config.aspect_w_over_h)

    def patch_size_m(self, frame: DroneFrame) -> tuple[float, float]:
        return patch_size_for_footprint_m(self.config, self.footprint_m(frame))

    def search_half_range_m(self, frame: DroneFrame) -> tuple[float, float]:
        """How far the truth may sit from the prior, along the frame's width and height, and still be found.

        For a round window this is the bounding box of the disc; `reachable` checks the disc itself.
        """
        return half_range_for_footprint_m(self.config, self.footprint_m(frame))

    @staticmethod
    def patch_offset_m(prior: LatLon, point: LatLon, yaw_deg: float) -> tuple[float, float]:
        """Where `point` falls in the heading-up patch centred on `prior`, `(right_m, down_m)`."""
        east_m, north_m = east_north_m(prior, point)
        return rotate_to_heading_up(east_m, -north_m, yaw_deg)

    def reachable(self, frame: DroneFrame, prior: LatLon) -> bool:
        return self.reach_shortfall_m(frame, prior) == 0.0

    def reach_shortfall_m(self, frame: DroneFrame, prior: LatLon) -> float:
        """Distance from the truth to the nearest position the search can return: the error floor, 0 when reachable."""
        du_m, dv_m = self.patch_offset_m(prior, frame.position, frame.yaw_deg)
        if self.config.window == "round":
            return max(float(np.hypot(du_m, dv_m)) - self.config.search_radius_m, 0.0)
        half_u_m, half_v_m = self.search_half_range_m(frame)
        return float(np.hypot(max(abs(du_m) - half_u_m, 0.0), max(abs(dv_m) - half_v_m, 0.0)))

    def drone_map(self, frame: DroneFrame, scale: float = 1.0) -> torch.Tensor:
        """Cached per (frame, scale). A scale other than 1 needs an extractor whose `drone` takes `scale=`."""
        key = (frame.name, scale)
        if key not in self._drone_maps:
            footprint = self.footprint_m(frame)
            self._drone_maps[key] = (self.features.drone(frame.image(), *footprint) if scale == 1.0
                                     else self.features.drone(frame.image(), *footprint, scale=scale))
        return self._drone_maps[key]

    def clear_cache(self) -> None:
        """Forget cached drone maps; a long simulated flight never sees the same frame twice."""
        self._drone_maps.clear()

    def prepare(self, drone_full: torch.Tensor, prior: LatLon, yaw_deg: float,
                patch_size_m: tuple[float, float]) -> tuple[torch.Tensor, torch.Tensor, SearchSurface, np.ndarray | None]:
        """Cropped drone map, cropped satellite patch at `prior`, an unscored surface and the window mask.

        `surface` scores these; a training-pair sampler calls it directly so pairs share the bench's geometry.
        """
        cfg = self.config
        patch_w_m, patch_h_m = patch_size_m
        sat_full = self.features.satellite_patch(prior, yaw_deg, patch_w_m, patch_h_m)
        if self.features_hook:
            self.features_hook(drone_full, sat_full)
        drone, drone_rows, drone_cols = crop_border(drone_full, cfg.drone_feature_crop_pct)
        sat, sat_rows, sat_cols = crop_border(sat_full, cfg.sat_feature_crop_pct)
        out_shape = (sat.shape[2] - drone.shape[2] + 1, sat.shape[3] - drone.shape[3] + 1)
        empty = SearchSurface(prior, yaw_deg, np.zeros(out_shape), float("nan"), (sat_rows, sat_cols),
                              (drone_rows, drone_cols), tuple(drone_full.shape[2:]), tuple(sat_full.shape[2:]),
                              (patch_h_m, patch_w_m))
        mask = None
        if cfg.window == "round":
            mask = np.hypot(*empty.offset_grid_m()) <= cfg.search_radius_m
        return drone, sat, empty, mask

    def surface(self, frame: DroneFrame, prior: LatLon) -> SearchSurface:
        if self.config.searches_pose:
            from uavloc.matching.pose_search import search_pose  # imported here: pose_search imports this module
            return search_pose(self, frame, prior)
        return self.surface_from_map(self.drone_map(frame), prior, frame.yaw_deg, self.patch_size_m(frame))

    def surface_from_map(self, drone_full: torch.Tensor, prior: LatLon, yaw_deg: float,
                         patch_size_m: tuple[float, float]) -> SearchSurface:
        """Translation-only surface for a ready drone feature map, e.g. a map patch scored against its surroundings."""
        cfg = self.config
        drone, sat, empty, mask = self.prepare(drone_full, prior, yaw_deg, patch_size_m)
        if cfg.matching == "weighted-ncc":
            heatmap = weighted_ncc_heatmap(drone, sat, cfg.class_weights, mask)
            coverage = weighted_coverage(drone, cfg.class_weights)
        elif cfg.matching == "learned":
            # hand-weight coverage, so a gate's c_min means the same "blind" as for weighted NCC
            heatmap, coverage = self.scorer(drone, sat, mask), weighted_coverage(drone, cfg.class_weights)
        elif cfg.matching in _MATCHERS:
            heatmap, coverage = _MATCHERS[cfg.matching](drone, sat, cfg, mask)
        else:
            heatmap, coverage = correlation_heatmap(drone, sat, mask), float("nan")
        return replace(empty, heatmap=heatmap, coverage=coverage)

    def correct(self, frame: DroneFrame, prior: LatLon) -> Correction:
        return decide(self.surface(frame, prior), self.config)
