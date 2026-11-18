"""A contrario acceptance of a correction (experiment 025): apply the argmax only when it is not a likely false alarm.

`NFA = N_tests * P_null(raw >= observed peak)`. The raw score is the weighted mean of per-class NCC before
z-scoring (as `weighted_ncc_heatmap` builds it), computed with the same drone map and window size on the actual
search surface and on every null window. The null pools every cell of M surfaces of the drone map against
satellite windows where the drone certainly is not; `N_tests` is the number of finite cells of the actual surface.
Nothing is fitted (Desolneux, Moisan & Morel; Moisan & Stival, IJCV 2004).

Two null samplers: `VillageNull` (windows on other villages' maps, one pool per frame) and `OwnMapNull` (windows
on the test place's own map, centres at least `OWN_MAP_EXCLUSION_M` from the draw's prior).
"""

from __future__ import annotations

import math
import time
import zlib
from dataclasses import dataclass, replace
from typing import Callable, Protocol

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from uavloc.data.drone_frames import DroneFrame, frame_place
from uavloc.data.match_pairs import random_pose
from uavloc.data.satellite import GeoImage
from uavloc.geo import LatLon, haversine_m
from uavloc.matching.benchmark import Draw, injected_priors, score_draws
from uavloc.matching.correlator import (MIN_PRESENT_SHARE, MIN_WINDOW_STD, CorrelatorConfig, FeatureExtractor,
                                        PositionCorrelator, crop_border, decide, drone_class_shares)
from uavloc.matching.learned import batched_ncc

NULL_WINDOWS = 200              # M, pre-registered
OWN_MAP_EXCLUSION_M = 60.0      # own-map null: a window centre this close to the prior may hold the truth
NFA_EPSILON = 1.0               # accept when NFA < epsilon; pre-registered, never tuned
WINDOW_MARGIN_PAD_M = 1.0       # keeps a rotated window's corners strictly inside the raster
NULL_SCORE_CHUNK = 50           # windows per FFT batch, bounds memory
OWN_MAP_CANDIDATE_CHUNK = 64    # own-map poses drawn at a time, as the exclusion discards some
PATCH_SIZE_TOLERANCE_M = 1e-6


class MapFeatures(FeatureExtractor, Protocol):
    """A feature extractor over one georeferenced map (`SegmentationFeatures`): poses are drawn on `probabilities`."""

    probabilities: GeoImage


@dataclass(frozen=True)
class WindowPose:
    place: str
    center: LatLon
    yaw_deg: float


def weighted_classes(drone: torch.Tensor, class_weights: tuple[float, ...]) -> list[int]:
    """Classes that take part, as in `class_ncc_maps`: weight > 0 and the drone argmax on >= `MIN_PRESENT_SHARE`."""
    shares = drone_class_shares(drone)
    return [c for c, weight in enumerate(class_weights) if weight > 0 and shares[c] >= MIN_PRESENT_SHARE]


def raw_scores(drone: torch.Tensor, satellites: torch.Tensor, class_weights: tuple[float, ...]) -> np.ndarray:
    """Raw weighted NCC of `drone` `(1, C, h, w)` against each of `satellites` `(B, C, H, W)`: `(B, H-h+1, W-w+1)`.

    The weighted mean of per-class NCC that `weighted_ncc_heatmap` z-scores, batched through `batched_ncc`
    (equal to `class_ncc` to ~1e-7). All zeros when no weighted class is present in the drone map.
    """
    drone, satellites = drone.float().cpu(), satellites.float().cpu()
    out_hw = (satellites.shape[2] - drone.shape[2] + 1, satellites.shape[3] - drone.shape[3] + 1)
    classes = weighted_classes(drone, class_weights)
    if not classes:
        return np.zeros((len(satellites), *out_hw), np.float32)
    weights = torch.tensor([class_weights[c] for c in classes], dtype=torch.float32)[None, :, None, None]
    query = drone[:, classes]
    parts = []
    for chunk in satellites.split(NULL_SCORE_CHUNK):
        ncc = batched_ncc(query.expand(len(chunk), -1, -1, -1), chunk[:, classes], MIN_WINDOW_STD)
        parts.append((ncc * weights).sum(1) / weights.sum())
    return torch.cat(parts).numpy()


def cropped_drone(correlator: PositionCorrelator, frame: DroneFrame) -> torch.Tensor:
    return crop_border(correlator.drone_map(frame), correlator.config.drone_feature_crop_pct)[0]


def raw_surface(correlator: PositionCorrelator, frame: DroneFrame, prior: LatLon) -> np.ndarray:
    """Raw weighted NCC of a draw's actual search surface, NaN outside a round window."""
    drone, satellite, _, mask = correlator.prepare(correlator.drone_map(frame), prior, frame.yaw_deg,
                                                   correlator.patch_size_m(frame))
    raw = raw_scores(drone, satellite, correlator.config.class_weights)[0]
    return raw if mask is None else np.where(mask, raw, np.nan).astype(np.float32)


def nfa(peak: float, n_tests: int, null_cells: np.ndarray) -> float:
    """`n_tests` times the share of pooled null cells scoring at least `peak`."""
    return n_tests * float(np.count_nonzero(null_cells >= peak)) / null_cells.size


def window_margin_m(patch_size_m: tuple[float, float]) -> float:
    """How far a window centre must stay from the raster edge for any heading."""
    return math.hypot(*patch_size_m) / 2 + WINDOW_MARGIN_PAD_M


def window_tensor(features: FeatureExtractor, pose: WindowPose, patch_size_m: tuple[float, float],
                  sat_crop_pct: float) -> torch.Tensor:
    """Satellite feature window at `pose`, border-cropped as the bench crops its search patch: `(1, C, H, W)` on CPU."""
    patch = features.satellite_patch(pose.center, pose.yaw_deg, *patch_size_m)
    return crop_border(patch, sat_crop_pct)[0].float().cpu()


def same_size(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return all(abs(x - y) <= PATCH_SIZE_TOLERANCE_M for x, y in zip(a, b))


class NullModel(Protocol):
    def pool(self, correlator: PositionCorrelator, frame: DroneFrame, prior: LatLon) -> np.ndarray:
        """Every cell of the M null surfaces for this draw, flattened."""
        ...

    def clear_cache(self) -> None: ...


class VillageNull:
    """Null (a): M windows at random poses on other villages' maps.

    The windows do not depend on the prior, so one pool serves every draw of a frame.
    """

    def __init__(self, windows: torch.Tensor, poses: list[WindowPose], patch_size_m: tuple[float, float],
                 sat_crop_pct: float) -> None:
        self.windows = windows
        self.poses = poses
        self.patch_size_m = patch_size_m
        self.sat_crop_pct = sat_crop_pct
        self._pools: dict[str, np.ndarray] = {}

    @classmethod
    def build(cls, places: list[str], features_for: Callable[[str], MapFeatures], patch_size_m: tuple[float, float],
              sat_crop_pct: float, n_windows: int, seed: int) -> VillageNull:
        """Each window's village uniform over `places`, then a uniform pose inside it; one map in memory at a time."""
        rng = np.random.default_rng(seed)
        village_of = rng.integers(len(places), size=n_windows)
        margin_m = window_margin_m(patch_size_m)
        windows: list[torch.Tensor | None] = [None] * n_windows
        poses: list[WindowPose | None] = [None] * n_windows
        for index in tqdm(sorted(set(village_of.tolist())), desc="null villages"):
            features = features_for(places[index])
            for k in np.flatnonzero(village_of == index):
                windows[k], poses[k] = sampled_window(rng, places[index], features, patch_size_m, sat_crop_pct,
                                                      margin_m)
            del features
        return cls(torch.cat(windows), poses, patch_size_m, sat_crop_pct)

    def pool(self, correlator: PositionCorrelator, frame: DroneFrame, prior: LatLon) -> np.ndarray:
        if frame.name not in self._pools:
            if not same_size(correlator.patch_size_m(frame), self.patch_size_m):
                raise ValueError(f"{frame.name}: patch {correlator.patch_size_m(frame)} m, null built for "
                                 f"{self.patch_size_m} m")
            if correlator.config.sat_feature_crop_pct != self.sat_crop_pct:
                raise ValueError("null windows were cropped with a different sat_feature_crop_pct")
            scores = raw_scores(cropped_drone(correlator, frame), self.windows, correlator.config.class_weights)
            self._pools[frame.name] = scores.ravel()
        return self._pools[frame.name]

    def clear_cache(self) -> None:
        self._pools.clear()


def sampled_window(rng: np.random.Generator, place: str, features: MapFeatures, patch_size_m: tuple[float, float],
                   sat_crop_pct: float, margin_m: float) -> tuple[torch.Tensor, WindowPose]:
    """A uniform pose on `features`' map whose window fits in the raster; redraws the rare one that does not."""
    while True:
        center, yaw_deg = random_pose(rng, features.probabilities, margin_m)
        pose = WindowPose(place, center, yaw_deg)
        try:
            return window_tensor(features, pose, patch_size_m, sat_crop_pct), pose
        except ValueError:
            continue


class OwnMapNull:
    """Null (b): M windows at random poses on the test place's own map, centres >= `exclusion_m` from the prior.

    Each place has one seeded sequence of candidate poses; a draw takes the first `n_windows` of them that are far
    enough from its prior. A frame's window scores are cached, so its draws share every window they all may use.
    """

    def __init__(self, n_windows: int, exclusion_m: float, seed: int) -> None:
        self.n_windows = n_windows
        self.exclusion_m = exclusion_m
        self.seed = seed
        self._rngs: dict[str, np.random.Generator] = {}
        self._poses: dict[str, list[WindowPose]] = {}
        self._windows: dict[str, list[torch.Tensor]] = {}   # one place at a time
        self._scores: dict[str, dict[int, np.ndarray]] = {}
        self._places: dict[str, str] = {}
        self._patch_size_m: dict[str, tuple[float, float]] = {}

    def place_of(self, frame: DroneFrame) -> str:
        if frame.name not in self._places:
            self._places[frame.name] = frame_place(frame)
        return self._places[frame.name]

    def ensure(self, place: str, features: MapFeatures, patch_size_m: tuple[float, float], sat_crop_pct: float,
               n_candidates: int) -> None:
        """At least `n_candidates` candidate poses for `place`, with their windows cropped."""
        if place not in self._rngs:
            self._rngs[place] = np.random.default_rng([self.seed, zlib.crc32(place.encode())])
            self._poses[place] = []
            self._patch_size_m[place] = patch_size_m
        if not same_size(patch_size_m, self._patch_size_m[place]):
            raise ValueError(f"{place}: candidates drawn for a {self._patch_size_m[place]} m patch, got {patch_size_m}")
        self.switch(place)
        poses, windows = self._poses[place], self._windows[place]
        while len(windows) < min(len(poses), n_candidates):   # poses whose crops went with another place's cache
            windows.append(window_tensor(features, poses[len(windows)], patch_size_m, sat_crop_pct))
        margin_m = window_margin_m(patch_size_m)
        while len(poses) < n_candidates:
            window, pose = sampled_window(self._rngs[place], place, features, patch_size_m, sat_crop_pct, margin_m)
            poses.append(pose)
            windows.append(window)

    def window_indices(self, correlator: PositionCorrelator, frame: DroneFrame, prior: LatLon) -> list[int]:
        """The first `n_windows` candidates of the frame's place whose centres are >= `exclusion_m` from `prior`."""
        place = self.place_of(frame)
        chosen, index = [], 0
        while len(chosen) < self.n_windows:
            if index >= len(self._poses.get(place, [])):
                self.ensure(place, correlator.features, correlator.patch_size_m(frame),
                            correlator.config.sat_feature_crop_pct, index + OWN_MAP_CANDIDATE_CHUNK)
            if haversine_m(self._poses[place][index].center, prior) >= self.exclusion_m:
                chosen.append(index)
            index += 1
        return chosen

    def switch(self, place: str) -> None:
        """Keep window crops and scores of one place only, to bound memory."""
        if place not in self._windows:
            self._windows = {place: []}
            self._scores = {}

    def pool(self, correlator: PositionCorrelator, frame: DroneFrame, prior: LatLon) -> np.ndarray:
        place = self.place_of(frame)
        self.switch(place)
        chosen = self.window_indices(correlator, frame, prior)
        scores = self._scores.setdefault(frame.name, {})
        missing = [i for i in chosen if i not in scores]
        if missing:
            self.ensure(place, correlator.features, correlator.patch_size_m(frame),
                        correlator.config.sat_feature_crop_pct, max(missing) + 1)
            windows = torch.cat([self._windows[place][i] for i in missing])
            for i, surface in zip(missing, raw_scores(cropped_drone(correlator, frame), windows,
                                                      correlator.config.class_weights)):
                scores[i] = surface
        return np.concatenate([scores[i].ravel() for i in chosen])

    def clear_cache(self) -> None:
        """Forget window crops and scores; the candidate pose sequences stay, so draws stay reproducible."""
        self._windows.clear()
        self._scores.clear()


def draw_nfa(correlator: PositionCorrelator, null: NullModel, draw: Draw) -> dict[str, float]:
    """`nfa`, the raw peak and `n_tests` of one draw; NaN when its patch left the raster."""
    if draw.surface is None:
        return {"nfa": float("nan"), "raw_peak": float("nan"), "n_tests": 0}
    raw = raw_surface(correlator, draw.frame, draw.prior)
    finite = raw[np.isfinite(raw)]
    peak = float(finite.max())
    return {"nfa": nfa(peak, finite.size, null.pool(correlator, draw.frame, draw.prior)), "raw_peak": peak,
            "n_tests": int(finite.size)}


def at_epsilon(records: pd.DataFrame, epsilon: float) -> pd.DataFrame:
    """Decision at `epsilon` on `score_nfa_draws` rows: the argmax move where NFA < epsilon, else the prior."""
    out = records.copy()
    accept = (out["status"] == "ok") & (out["nfa"] < epsilon)
    out["decision"] = np.where(accept, "accept", "reject")
    out["error_m"] = np.where(accept, out["argmax_error_m"], out["prior_error_m"])
    out["corrected_lat"] = np.where(accept, out["argmax_lat"], out["prior_lat"])
    out["corrected_lon"] = np.where(accept, out["argmax_lon"], out["prior_lon"])
    out["applied_offset_m"] = np.where(accept, out["map_offset_m"], 0.0)
    return out


def score_nfa_draws(draws: list[Draw], correlator: PositionCorrelator, null: NullModel,
                    epsilon: float = NFA_EPSILON) -> pd.DataFrame:
    """`score_draws` rows (ungated argmax) plus `nfa`, `raw_peak`, `n_tests`, decided at `epsilon` (no across-only)."""
    records = score_draws(draws, replace(correlator.config, gate="none"))
    records["argmax_error_m"] = records["error_m"]
    records["argmax_lat"], records["argmax_lon"] = records["corrected_lat"], records["corrected_lon"]
    stats = [draw_nfa(correlator, null, d) for d in tqdm(draws, desc="nfa", leave=False)]
    records = pd.concat([records, pd.DataFrame(stats)], axis=1)
    return at_epsilon(records, epsilon)


def median_ms_per_nfa_correction(correlator: PositionCorrelator, null: NullModel, frames: list[DroneFrame],
                                 noise_m: float, n_directions: int, repeats: int,
                                 epsilon: float = NFA_EPSILON) -> float:
    """As `median_ms_per_correction`, plus the raw surface, the frame's null and the NFA (caches cleared)."""
    priors = [injected_priors(f.position, noise_m, n_directions, 0.0)[0][1] for f in frames]
    config: CorrelatorConfig = replace(correlator.config, gate="none")
    times_ms = []
    for _ in range(repeats):
        correlator.clear_cache()
        null.clear_cache()
        start = time.perf_counter()
        for frame, prior in zip(frames, priors):
            surface = correlator.surface(frame, prior)
            decide(surface, config)
            raw = raw_surface(correlator, frame, prior)
            finite = raw[np.isfinite(raw)]
            nfa(float(finite.max()), finite.size, null.pool(correlator, frame, prior)) < epsilon
        times_ms.append((time.perf_counter() - start) * 1000 / len(frames))
    return float(np.median(times_ms))
