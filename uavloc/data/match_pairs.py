"""Training pairs for a learned matcher (experiment 023): a `current` crop at a known pose against an older date.

`ua-village-multidate` dates are co-registered to 1.5 cm median, so the offset between the drone view and a map
patch around a shifted prior is exact without any label. Each pair goes through the bench's own code path: the
query is cut and JPEG-encoded as `village-crossdate-frames/build.ipynb` writes a frame, segmented by
`SegmentationFeatures.drone`, and the patch comes from `PositionCorrelator.prepare`.

A bank is cached under `data/derived/match-pairs/<key>/` as memory-mapped arrays, `key` hashing everything that
shapes it. It is a cache of model outputs, not a registered dataset.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from tqdm.auto import tqdm

from uavloc.data.drone_frames import DERIVED
from uavloc.data.multidate import MULTIDATE, esri_dates, load_date
from uavloc.data.satellite import GeoImage
from uavloc.features.segmentation import SegmentationFeatures
from uavloc.geo import FULL_TURN_DEG, LatLon, footprint_m
from uavloc.matching.correlator import (CAMERA_ASPECT_W_OVER_H, MIN_WINDOW_STD, CorrelatorConfig, PositionCorrelator,
                                        SearchSurface, shifted)
from uavloc.matching.learned import batched_ncc, class_presence

BANKS_ROOT = DERIVED / "match-pairs"
PROBABILITY_SCALE = 255     # probabilities are stored as uint8
ARRAYS = ("query", "satellite", "ncc", "presence", "cell")


@dataclass(frozen=True)
class PairBankConfig:
    drone_date: str
    altitude_m: float
    fov_w_deg: float
    margin_m: float
    search_scale: float
    match_gsd_m: float
    sat_feature_crop_pct: float
    seg_channels: str
    seg_checkpoint: str
    jpeg_quality: int = 95    # as `village-crossdate-frames` writes frame.jpg

    @property
    def footprint_m(self) -> tuple[float, float]:
        return footprint_m(self.altitude_m, self.fov_w_deg, CAMERA_ASPECT_W_OVER_H)

    @property
    def patch_size_m(self) -> tuple[float, float]:
        width_m, height_m = self.footprint_m
        return width_m * self.search_scale, height_m * self.search_scale

    def correlator_config(self) -> CorrelatorConfig:
        """Geometry only: the window and border crops the bench uses."""
        return CorrelatorConfig(self.search_scale, self.fov_w_deg, sat_feature_crop_pct=self.sat_feature_crop_pct)


def multidate_places() -> list[str]:
    return sorted(p.parent.name for p in (DERIVED / MULTIDATE).glob("*/meta.json"))


def village_split(exclude: list[str], n_val: int, seed: int) -> tuple[list[str], list[str]]:
    """Locations of `ua-village-multidate` minus `exclude`, split by location into (train, val)."""
    places = [p for p in multidate_places() if p not in exclude]
    order = np.random.default_rng(seed).permutation(len(places))
    val = sorted(places[i] for i in order[:n_val])
    return sorted(p for p in places if p not in val), val


def random_pose(rng: np.random.Generator, image: GeoImage, margin_m: float) -> tuple[LatLon, float]:
    """Uniform position at least `margin_m` inside the raster, uniform heading (as the frames' build notebook)."""
    mpp_x, mpp_y = image.mpp_xy
    x_m = rng.uniform(margin_m, image.size_m[0] - margin_m)
    y_m = rng.uniform(margin_m, image.size_m[1] - margin_m)
    return image.to_latlon(x_m / mpp_x, y_m / mpp_y), float(rng.uniform(0, FULL_TURN_DEG))


def random_prior(rng: np.random.Generator, truth: LatLon, yaw_deg: float, template: SearchSurface) -> LatLon:
    """A prior from which the truth sits at a cell drawn uniformly over the search surface."""
    rows, cols = template.heatmap.shape
    right_m, down_m = template.offset_m((rng.uniform(0, rows - 1), rng.uniform(0, cols - 1)))
    return shifted(truth, (-right_m, -down_m), yaw_deg)


def jpeg_roundtrip(rgb: np.ndarray, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="JPEG", quality=quality)
    return Image.open(buffer).convert("RGB")


def bank_key(places: list[str], n_pairs: int, config: PairBankConfig, seed: int) -> str:
    spec = json.dumps({"places": places, "n_pairs": n_pairs, "config": asdict(config), "seed": seed}, sort_keys=True)
    return hashlib.sha1(spec.encode()).hexdigest()[:12]


class PairBank:
    """Pairs on disk: `query` `(N, C, h, w)` and `satellite` `(N, C, H, W)` uint8 probabilities, the per-class NCC
    surfaces `ncc` `(N, C, H', W')`, class `presence` `(N, C)` and the true fractional `cell` `(N, 2)`."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.arrays = {name: np.load(root / f"{name}.npy", mmap_mode="r") for name in ARRAYS}
        self.meta = json.loads((root / "meta.json").read_text())
        self.pairs = pd.read_csv(root / "pairs.csv")
        self.cell_m = tuple(self.meta["cell_m"])

    def __len__(self) -> int:
        return len(self.pairs)

    def batch(self, indices: np.ndarray) -> dict[str, torch.Tensor]:
        """CPU float tensors (bool for `presence`) of the pairs at `indices`."""
        out = {name: torch.from_numpy(np.ascontiguousarray(self.arrays[name][indices])) for name in ARRAYS}
        out["query"] = out["query"].float() / PROBABILITY_SCALE
        out["satellite"] = out["satellite"].float() / PROBABILITY_SCALE
        out["ncc"] = out["ncc"].float()
        return out


def as_uint8(probabilities: torch.Tensor) -> np.ndarray:
    return (probabilities[0].float().cpu().numpy() * PROBABILITY_SCALE).round().astype(np.uint8)


def build_pair_bank(places: list[str], n_pairs: int, config: PairBankConfig, model: nn.Module,
                    device: torch.device | str, seed: int) -> PairBank:
    """Pairs spread evenly over `places`, map date uniform over each place's Wayback dates; cached by `bank_key`."""
    root = BANKS_ROOT / bank_key(places, n_pairs, config, seed)
    if (root / "meta.json").exists():
        return PairBank(root)
    tmp = root.with_suffix(".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    counts = [len(chunk) for chunk in np.array_split(np.arange(n_pairs), len(places))]
    arrays, rows, template, index = {}, [], None, 0
    bar = tqdm(total=n_pairs, desc="pairs", unit="pair")
    for place, count in zip(places, counts):
        current = load_date(place, config.drone_date)
        dates = rng.choice(esri_dates(place), size=count)
        for date in sorted(set(dates)):
            features = SegmentationFeatures(model, model, load_date(place, date), config.seg_channels,
                                            config.match_gsd_m, config.match_gsd_m, device)
            correlator = PositionCorrelator(features, config.correlator_config())
            for _ in range(int((dates == date).sum())):
                truth, yaw_deg = random_pose(rng, current, config.margin_m)
                image = jpeg_roundtrip(current.crop_rotated(truth, yaw_deg, *config.footprint_m), config.jpeg_quality)
                drone_full = features.drone(image, *config.footprint_m)
                if template is None:
                    template = correlator.prepare(drone_full, truth, yaw_deg, config.patch_size_m)[2]
                prior = random_prior(rng, truth, yaw_deg, template)
                drone, satellite, surface, _ = correlator.prepare(drone_full, prior, yaw_deg, config.patch_size_m)
                cell = surface.cell_of_offset(PositionCorrelator.patch_offset_m(prior, truth, yaw_deg))
                values = {"query": as_uint8(drone), "satellite": as_uint8(satellite),
                          "ncc": batched_ncc(drone, satellite, MIN_WINDOW_STD)[0].numpy().astype(np.float16),
                          "presence": class_presence(drone)[0].numpy(), "cell": np.array(cell, np.float32)}
                if not arrays:
                    arrays = {name: np.lib.format.open_memmap(tmp / f"{name}.npy", mode="w+", dtype=v.dtype,
                                                              shape=(n_pairs, *v.shape)) for name, v in values.items()}
                for name, value in values.items():
                    arrays[name][index] = value
                rows.append({"place": place, "map_date": date, "lat": truth.lat, "lon": truth.lon, "yaw_deg": yaw_deg,
                             "prior_lat": prior.lat, "prior_lon": prior.lon, "cell_row": cell[0], "cell_col": cell[1]})
                index += 1
                bar.update()
            del features, correlator
    bar.close()
    for array in arrays.values():
        array.flush()
    patch_h_m, patch_w_m = template.patch_hw_m
    cell_m = (patch_h_m / template.sat_hw[0], patch_w_m / template.sat_hw[1])
    pd.DataFrame(rows).to_csv(tmp / "pairs.csv", index=False)
    (tmp / "meta.json").write_text(json.dumps({"places": places, "n_pairs": n_pairs, "config": asdict(config),
                                               "seed": seed, "cell_m": cell_m,
                                               "surface_hw": list(template.heatmap.shape)}, indent=1))
    tmp.rename(root)
    return PairBank(root)
