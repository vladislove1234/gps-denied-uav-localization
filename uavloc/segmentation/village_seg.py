"""Round-3 village segmentation: satellite and drone data, training samplers, per-domain BatchNorm and scoring.

Moved from `pocs/village-seg-baseline/village_seg.py` on its third use (the PoC file now re-exports from here), which
had taken it from `round2.ipynb`. Sampler and augmentation draws are unchanged, so a seed gives the same crops.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Mapping

import numpy as np
import pandas as pd
import torch
from PIL import Image
from shapely import affinity
from shapely.geometry import Polygon
from torch import nn

from uavloc.data.annotated import IGNORE_LABEL, VILLAGE_EXPORT, AnnotatedImage, load_annotated_images
from uavloc.data.drone_frames import DERIVED
from uavloc.features.imagenet import IMAGENET_MEAN, IMAGENET_STD, UINT8_MAX
from uavloc.features.segmentation import class_probabilities, recalibrate_batchnorm
from uavloc.geo import LatLon, east_north_m, footprint_m
from uavloc.segmentation.eval import (HOLDOUT_LOCATION, MEAN_KEY, STRIP_LOCATION, STRIP_ROWS, label_at_stride,
                                      model_predictions, score_sets)
from uavloc.segmentation.instances import building_table, merge_scores
from uavloc.segmentation.metrics import iou_scores
from uavloc.segmentation.topology import track_scores
from uavloc.track import REPO_ROOT

IGNORE = IGNORE_LABEL
SAT_RARE_CLASSES = (1, 2, 3, 5)      # building, track, woody, water
DRONE_RARE_CLASSES = (2, 3)          # track, woody
CENTRE_THIN = 7                      # neighbouring pixels give the same crop
DRONE_HALF_FOV_W_DEG, DRONE_HALF_FOV_H_DEG = 31.5, 17.5
DRONE_ASPECT_W_OVER_H = math.tan(math.radians(DRONE_HALF_FOV_W_DEG)) / math.tan(math.radians(DRONE_HALF_FOV_H_DEG))
BLOCK_GAP_MS = 5000
GSD_JITTER = (0.8, 1.25)
BRIGHTNESS_JITTER, TINT_JITTER, CONTRAST_JITTER = 0.15, 0.05, 0.15
QUARTER_TURNS = 4
FLIP_P = 0.5
RGB_CHANNELS = 3
DRONE_EXPORT = "ua-drone-annotated"
DRONE_VAL_NAME = "drone-val"
ROUND3_RESULTS = REPO_ROOT / "pocs" / "village-seg-baseline" / "results-round3.json"
ORACLE, MODEL = "stride-8 oracle, bilinear", "model"      # keys of `building_table`'s status columns
MERGE_KEYS = ("separable_merge_rate", "resolution_bound_merge_rate", "separate_share", "merged_share", "missed_share")


@dataclass
class SatImage:
    loc: str
    date: str
    img: np.ndarray
    lab: np.ndarray
    banned: np.ndarray | None = None


@dataclass
class DroneImage:
    name: str
    block: int
    mpp: float
    alt_m: float
    footprint: Polygon
    img: np.ndarray
    lab: np.ndarray
    split: str = "train"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_satellite(village_dir: Path, derived_dir: Path) -> list[SatImage]:
    """Now a wrapper: loading moved to `uavloc.data.annotated.load_annotated_images` (build seg-boundary-eval)."""
    return [SatImage(a.location, a.date, a.rgb, a.label)
            for a in load_annotated_images(village_dir.relative_to(derived_dir).as_posix(), derived_dir)]


def ban_holdouts(sat: list[SatImage], holdout_village: str, strip_village: str, strip_rows: int, gap_px: int) -> None:
    for s in sat:
        s.banned = np.zeros(s.lab.shape, bool)
        if s.loc == holdout_village:
            s.banned[:] = True
        elif s.loc == strip_village:
            s.banned[:strip_rows + gap_px] = True


def drone_footprint(origin: LatLon, lat: float, lon: float, alt_m: float, yaw_deg: float) -> Polygon:
    east, north = east_north_m(origin, LatLon(lat, lon))
    w, h = footprint_m(alt_m, 2 * DRONE_HALF_FOV_W_DEG, DRONE_ASPECT_W_OVER_H)
    box = Polygon([(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)])
    return affinity.translate(affinity.rotate(box, -yaw_deg, origin=(0, 0)), east, north)


def load_drone(drone_dir: Path, derived_dir: Path) -> list[DroneImage]:
    """Frames by time, grouped into blocks wherever the log has a gap longer than `BLOCK_GAP_MS`."""
    records = sorted(read_jsonl(drone_dir / "samples.jsonl"), key=lambda r: r["position"]["timestamp_ms"])
    origin = LatLon(records[0]["position"]["lat"], records[0]["position"]["lon"])
    out, block, prev_ms = [], 0, None
    for r in records:
        p = r["position"]
        if prev_ms is not None and p["timestamp_ms"] - prev_ms > BLOCK_GAP_MS:
            block += 1
        prev_ms = p["timestamp_ms"]
        out.append(DroneImage(r["location"], block, r["mpp"], p["alt_m"],
                              drone_footprint(origin, p["lat"], p["lon"], p["alt_m"], p["yaw_deg"]),
                              np.array(Image.open(derived_dir / r["image_file"]).convert("RGB")),
                              np.array(Image.open(drone_dir / r["label_file"]))))
    return out


def split_by_footprint(drone: list[DroneImage], val_blocks: list[int]) -> None:
    """Val = `val_blocks`; any other frame whose footprint touches a val footprint is dropped."""
    val_union = None
    for d in drone:
        if d.block in val_blocks:
            val_union = d.footprint if val_union is None else val_union.union(d.footprint)
    for d in drone:
        d.split = "val" if d.block in val_blocks else "dropped" if d.footprint.intersects(val_union) else "train"


def augment(img: np.ndarray, lab: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """rot90/flip on both; brightness, tint, contrast and ImageNet norm on the image. Image out is `(3, H, W)`."""
    k = rng.integers(QUARTER_TURNS)
    img, lab = np.rot90(img, k), np.rot90(lab, k)
    if rng.random() < FLIP_P:
        img, lab = img[:, ::-1], lab[:, ::-1]
    x = np.asarray(img, np.float32) / UINT8_MAX
    x = x * (1 + rng.uniform(-BRIGHTNESS_JITTER, BRIGHTNESS_JITTER) + rng.uniform(-TINT_JITTER, TINT_JITTER, RGB_CHANNELS))
    mean = x.mean()
    x = (x - mean) * (1 + rng.uniform(-CONTRAST_JITTER, CONTRAST_JITTER)) + mean
    return ((x - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1).copy(), lab.copy()


def valid_corners(banned: np.ndarray, crop_px: int) -> np.ndarray:
    """True where a `crop_px` square with that top-left corner touches no banned pixel."""
    ii = np.pad(banned.astype(np.int32).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    hits = ii[crop_px:, crop_px:] - ii[:-crop_px, crop_px:] - ii[crop_px:, :-crop_px] + ii[:-crop_px, :-crop_px]
    return hits == 0


class SatelliteSampler:
    """Random `crop_px` crops off the banned area; a share `rare_p` centred on a pixel of a rare class."""

    def __init__(self, images: list[SatImage], crop_px: int, rare_p: float) -> None:
        self.images, self.crop_px, self.rare_p = images, crop_px, rare_p
        self.ok = [valid_corners(s.banned, crop_px) for s in images]
        self.corners = [np.argwhere(ok) for ok in self.ok]
        self.centres = {c: self._centres(c) for c in SAT_RARE_CLASSES}
        self.centres = {c: pts for c, pts in self.centres.items() if len(pts)}

    def _corner(self, s: SatImage, y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = s.lab.shape
        return np.clip(y - self.crop_px // 2, 0, h - self.crop_px), np.clip(x - self.crop_px // 2, 0, w - self.crop_px)

    def _centres(self, cls: int) -> np.ndarray:
        """`(image idx, y, x)` rows, thinned, whose crop stays clear of banned pixels."""
        pts = []
        for k, s in enumerate(self.images):
            ys, xs = np.nonzero(s.lab == cls)
            sel = np.nonzero(self.ok[k][self._corner(s, ys, xs)])[0][::CENTRE_THIN]
            pts.append(np.c_[np.full(len(sel), k), ys[sel], xs[sel]])
        return np.concatenate(pts)

    def __call__(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        if rng.random() < self.rare_p:
            pts = self.centres[rng.choice(list(self.centres))]
            k, y, x = pts[rng.integers(len(pts))]
            top, left = self._corner(self.images[k], y, x)
        else:
            k = rng.integers(len(self.images))
            top, left = self.corners[k][rng.integers(len(self.corners[k]))]
        s, n = self.images[k], self.crop_px
        return augment(s.img[top:top + n, left:left + n], s.lab[top:top + n, left:left + n], rng)


def pad_crop(arr: np.ndarray, top: int, left: int, n: int, fill: int) -> np.ndarray:
    """`n`-px square at `(top, left)`; the part outside `arr` is `fill`."""
    h, w = arr.shape[:2]
    out = np.full((n, n, *arr.shape[2:]), fill, arr.dtype)
    rows, cols = slice(max(top, 0), min(top + n, h)), slice(max(left, 0), min(left + n, w))
    out[rows.start - top:rows.stop - top, cols.start - left:cols.stop - left] = arr[rows, cols]
    return out


class DroneSampler:
    """`crop_px` squares at ~`gsd_m` (jittered); a share `rare_p` centred on track/woody."""

    def __init__(self, frames: list[DroneImage], gsd_m: float, crop_px: int, rare_p: float) -> None:
        self.frames, self.gsd_m, self.crop_px, self.rare_p = frames, gsd_m, crop_px, rare_p

    def _centre(self, lab: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
        rare = [c for c in DRONE_RARE_CLASSES if (lab == c).any()]
        if rng.random() < self.rare_p and rare:
            ys, xs = np.nonzero(lab == rng.choice(rare))
            j = rng.integers(len(ys))
            return ys[j], xs[j]
        return rng.integers(lab.shape[0]), rng.integers(lab.shape[1])

    def __call__(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        d = self.frames[rng.integers(len(self.frames))]
        n = int(round(self.crop_px * self.gsd_m * rng.uniform(*GSD_JITTER) / d.mpp))
        (h, w), (cy, cx) = d.lab.shape, self._centre(d.lab, rng)
        # inside the frame where it fits; centred (padded both sides) where it does not
        top = int(np.clip(cy - n // 2, 0, h - n)) if n <= h else (h - n) // 2
        left = int(np.clip(cx - n // 2, 0, w - n)) if n <= w else (w - n) // 2
        size = (self.crop_px, self.crop_px)
        img = Image.fromarray(pad_crop(d.img, top, left, n, 0)).resize(size, Image.BILINEAR)
        lab = Image.fromarray(pad_crop(d.lab, top, left, n, IGNORE)).resize(size, Image.NEAREST)
        return augment(np.asarray(img), np.asarray(lab), rng)


Sampler = Callable[[np.random.Generator], tuple[np.ndarray, np.ndarray]]


def batch_of(sampler: Sampler, rng: np.random.Generator, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    xs, ys = zip(*(sampler(rng) for _ in range(n)))
    return torch.from_numpy(np.stack(xs)).float(), torch.from_numpy(np.stack(ys)).long()


def domain_bn(model: nn.Module, sampler: Sampler, batch: int, n_batches: int, seed: int,
              device: torch.device | str) -> nn.Module:
    """Copy of `model` with BatchNorm statistics from `n_batches` of one domain's training crops."""
    rng = np.random.default_rng(seed)
    batches: Iterator[torch.Tensor] = (batch_of(sampler, rng, batch)[0] for _ in range(n_batches))
    return recalibrate_batchnorm(model, batches, device)


def drone_at_gsd(d: DroneImage, gsd_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Frame and label resampled to `gsd_m`, the scale the model is trained to see."""
    s = d.mpp / gsd_m
    size = (round(d.img.shape[1] * s), round(d.img.shape[0] * s))
    return (np.asarray(Image.fromarray(d.img).resize(size, Image.LANCZOS)),
            np.asarray(Image.fromarray(d.lab).resize(size, Image.NEAREST)))


def predict(model: nn.Module, rgb: np.ndarray, device: torch.device | str) -> np.ndarray:
    """`(H, W)` uint8 class ids of one image."""
    return class_probabilities(model, rgb, device).argmax(0).astype(np.uint8)


def agreement(preds: list[np.ndarray]) -> float | None:
    """Mean pixel agreement over every pair of predictions of the same ground."""
    pairs = [(i, j) for i in range(len(preds)) for j in range(i + 1, len(preds))]
    return float(np.mean([(preds[i] == preds[j]).mean() for i, j in pairs])) if pairs else None


def score_images(model: nn.Module, items: list[tuple[np.ndarray, np.ndarray]],
                 device: torch.device | str) -> dict:
    """`iou_scores` (per-class IoU, mIoU, Dice) of `model` on `(rgb, label)` whole images; `model` must be in eval mode."""
    return iou_scores([predict(model, rgb, device) for rgb, _ in items], [lab for _, lab in items])


def drone_val_metrics(model: nn.Module, frames: list[DroneImage], gsd_m: float,
                      device: torch.device | str) -> dict[str, float]:
    """`miou_drone-val` and `<class>_iou_drone-val` of a drone-BN model on frames resampled to `gsd_m`."""
    scores = score_images(model, [drone_at_gsd(d, gsd_m) for d in frames], device)
    return {f"miou_{DRONE_VAL_NAME}": scores["miou"],
            **{f"{cls}_iou_{DRONE_VAL_NAME}": v for cls, v in scores["iou"].items()}}


def building_merge_scores(preds: Mapping[str, list[np.ndarray]],
                          val_sets: Mapping[str, list[AnnotatedImage]]) -> dict[str, dict[str, float]]:
    """021's building status shares and merge rates per val set (all its dates pooled), plus `MEAN_KEY` over sets."""
    scores = {}
    for name, images in val_sets.items():
        tables = [building_table(a.label, {ORACLE: label_at_stride(a.label, "bilinear"), MODEL: p})
                  for a, p in zip(images, preds[name])]
        scores[name] = merge_scores(pd.concat(tables, ignore_index=True), MODEL, ORACLE)
    scores[MEAN_KEY] = {k: float(np.mean([scores[n][k] for n in val_sets])) for k in MERGE_KEYS}
    return scores


def track_score_table(preds: Mapping[str, list[np.ndarray]],
                      labels: Mapping[str, list[np.ndarray]]) -> dict[str, dict[str, float]]:
    """`topology.track_scores` per val set, plus `MEAN_KEY` over sets."""
    scores = {name: track_scores(preds[name], labels[name]) for name in labels}
    scores[MEAN_KEY] = {k: float(np.mean([scores[n][k] for n in labels])) for k in scores[next(iter(labels))]}
    return scores


def flatten_val_metrics(results: dict[str, dict], merges: dict[str, dict], tracks: dict[str, dict],
                        sets: list[str]) -> dict[str, float]:
    """One flat dict keyed `<metric>_<set>` and `<metric>_mean`, with 021's names: `miou_mean`,
    `building_missed_share_mean`, `separable_merge_rate_mean`, `track_skeleton_recall_mean`, `building_iou_<set>`..."""
    flat = {}
    for name in [*sets, MEAN_KEY]:
        flat.update({f"{k}_{name}": v for k, v in results[name].items() if not k.endswith("_per_class")})
        flat.update({f"building_{k}_{name}": merges[name][k] for k in ("separate_share", "merged_share", "missed_share")})
        flat.update({f"{k}_{name}": merges[name][k] for k in ("separable_merge_rate", "resolution_bound_merge_rate")})
        flat.update({f"{k}_{name}": v for k, v in tracks[name].items()})
    for kind in ("iou", "dice", "biou"):
        per_set = {name: results[name][f"{kind}_per_class"] for name in sets}
        for cls in sorted({c for scores in per_set.values() for c in scores}):
            present = {name: scores[cls] for name, scores in per_set.items() if cls in scores}
            flat.update({f"{cls}_{kind}_{name}": v for name, v in present.items()})
            flat[f"{cls}_{kind}_{MEAN_KEY}"] = float(np.mean(list(present.values())))
    return flat


def val_metrics(model: nn.Module, val_sets: Mapping[str, list[AnnotatedImage]],
                device: torch.device | str) -> dict[str, float]:
    """The 021 satellite val metrics of a satellite-BN model on `val_sets`, flat (see `flatten_val_metrics`)."""
    preds = model_predictions(model, dict(val_sets), device)
    labels = {name: [a.label for a in images] for name, images in val_sets.items()}
    return flatten_val_metrics(score_sets(preds, labels), building_merge_scores(preds, val_sets),
                               track_score_table(preds, labels), list(val_sets))


@dataclass(frozen=True)
class SplitConfig:
    """Round 3's split: bushcha held out whole, the top `strip_rows` of birky (plus a gap) held out, drone blocks 0 and 4."""

    dataset: str = VILLAGE_EXPORT
    drone_dataset: str = DRONE_EXPORT
    holdout_location: str = HOLDOUT_LOCATION
    strip_location: str = STRIP_LOCATION
    strip_rows: int = STRIP_ROWS
    gap_px: int = 64
    drone_val_blocks: tuple[int, ...] = field(default=(0, 4))


@dataclass
class VillageData:
    """Satellite images (with the held-out ground banned) and the drone frames, split into train and val."""

    sat: list[SatImage]
    sat_train: list[SatImage]
    drone_train: list[DroneImage]
    drone_val: list[DroneImage]


def load_village_data(split: SplitConfig = SplitConfig(), derived_dir: Path = DERIVED) -> VillageData:
    sat = load_satellite(derived_dir / split.dataset, derived_dir)
    ban_holdouts(sat, split.holdout_location, split.strip_location, split.strip_rows, split.gap_px)
    drone = load_drone(derived_dir / split.drone_dataset, derived_dir)
    split_by_footprint(drone, list(split.drone_val_blocks))
    return VillageData(sat, [s for s in sat if not s.banned.all()],
                       [d for d in drone if d.split == "train"], [d for d in drone if d.split == "val"])


def check_round3_splits(data: VillageData, results_path: Path = ROUND3_RESULTS) -> None:
    """Raise unless the satellite train images and the drone train / val frames are exactly round 3's."""
    recorded = json.loads(results_path.read_text())["splits"]
    got = {"sat_train": [(s.loc, s.date) for s in data.sat_train], "drone_train": [d.name for d in data.drone_train],
           "drone_val": [d.name for d in data.drone_val]}
    for key, names in got.items():
        if names != [tuple(n) if isinstance(n, list) else n for n in recorded[key]]:
            raise ValueError(f"{key} differs from round 3: {len(names)} vs {len(recorded[key])} entries")
