"""Ground-truth class maps as an oracle: GT labels, the GT query crop, its damage, and channel substitution.

Promoted from `experiments/006-segmentation-ceiling/gt_query.py` on its second use
(`builds/corrector-identity-bench/matching-ab.ipynb`), which still re-exports from here.
"""

from __future__ import annotations

import json
import math
from typing import Callable

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from uavloc.data.annotated import load_label_map
from uavloc.data.drone_frames import DERIVED, DroneFrame
from uavloc.data.multidate import MULTIDATE
from uavloc.data.satellite import GeoImage
from uavloc.features.segmentation import CLASSES
from uavloc.geo import LatLon
from uavloc.matching.correlator import CorrelatorConfig, FeatureExtractor, PositionCorrelator

OSM_TO_MODEL = np.array([0, 1, 2, 5, 3], np.uint8)  # osm background, building, road, water, wood -> model class ids
MIN_OFFSET_PEAK_Z = 4.0   # below this the build's label offset is meaningless (align.offset docstring)
ANNOTATED_DATE = "current"   # the date the frames were cut from, on the reference grid
LABEL_SOURCES = ("osm", "annotated")
DAMAGES = ("none", "stride", "boundary")

QueryFn = Callable[[DroneFrame], np.ndarray]


def osm_labels(place: str) -> GeoImage:
    """OSM `label.png` in model class ids, shifted onto the imagery by the build's measured offset when it is reliable.

    Shifting by `+(dy, dx)` raises building IoU against the hand labels (birky 0.57 -> 0.63, bushcha 0.23 -> 0.83).
    """
    folder = DERIVED / MULTIDATE / place
    meta = json.loads((folder / "meta.json").read_text())
    ids = OSM_TO_MODEL[np.asarray(Image.open(folder / "label.png"))]
    offset = meta["label_offset"]
    if offset["peak_z"] >= MIN_OFFSET_PEAK_Z and not offset["at_search_limit"]:
        ids = np.roll(ids, (offset["dy_px"], offset["dx_px"]), axis=(0, 1))
    bounds = meta["bounds"]
    return GeoImage(ids, LatLon(bounds["top"], bounds["left"]), LatLon(bounds["bottom"], bounds["right"]))


def gt_labels(place: str, source: str) -> GeoImage:
    """Class-id raster `(H, W)` of `place`: `osm` (pre-registered) or `annotated` (hand-corrected, `current` date)."""
    if source == "osm":
        return osm_labels(place)
    if source == "annotated":
        return load_label_map(place, ANNOTATED_DATE)
    raise ValueError(f"source must be one of {LABEL_SOURCES}, got {source!r}")


def one_hot(ids: np.ndarray) -> np.ndarray:
    """`(H, W)` class ids → `(C, H, W)` float32; ids outside the class set belong to no class."""
    return (ids[None] == np.arange(len(CLASSES))[:, None, None]).astype(np.float32)


def gt_query(labels: GeoImage, frame: DroneFrame, footprint_m: tuple[float, float], mpp: float) -> np.ndarray:
    """`(C, H, W)` one-hot GT cut heading-up at the frame's true pose, on the grid the model query uses."""
    return one_hot(labels.crop_rotated(frame.position, frame.yaw_deg, *footprint_m, out_mpp=mpp, nearest=True))


def stride_damage(query: np.ndarray, stride: int) -> np.ndarray:
    """One-hot → area-average down by `stride` → bilinear back up → argmax: a head that predicts at 1/stride."""
    if stride == 1:
        return query
    h, w = query.shape[1:]
    coarse = F.interpolate(torch.from_numpy(query)[None], size=(math.ceil(h / stride), math.ceil(w / stride)), mode="area")
    fine = F.interpolate(coarse, size=(h, w), mode="bilinear", align_corners=False)
    return one_hot(fine[0].argmax(0).numpy())


def boundary_damage(query: np.ndarray, radius_px: int) -> np.ndarray:
    """Per class, opening then closing with a disk of `radius_px`: rounds corners, removes parts narrower than 2r.

    Channels are damaged independently, so they may overlap or leave gaps, as a soft prediction can.
    """
    if radius_px == 0:
        return query
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius_px + 1, 2 * radius_px + 1))
    return np.stack([cv2.morphologyEx(cv2.morphologyEx(channel, cv2.MORPH_OPEN, disk), cv2.MORPH_CLOSE, disk)
                     for channel in query])


def damaged(query: np.ndarray, damage: str, damage_px: int) -> np.ndarray:
    if damage == "none":
        return query
    if damage == "stride":
        return stride_damage(query, damage_px)
    if damage == "boundary":
        return boundary_damage(query, damage_px)
    raise ValueError(f"damage must be one of {DAMAGES}, got {damage!r}")


def substituted(model_query: np.ndarray, gt: np.ndarray, class_ids: list[int]) -> np.ndarray:
    """The model's `(C, H, W)` probabilities with the channels in `class_ids` replaced by GT one-hot channels."""
    out = model_query.copy()
    out[class_ids] = gt[class_ids]
    return out


class QueryCorrelator(PositionCorrelator):
    """A correlator whose drone map is `query_fn(frame)`, e.g. ground truth cut at the true pose, not the image's."""

    def __init__(self, features: FeatureExtractor, config: CorrelatorConfig, query_fn: QueryFn) -> None:
        super().__init__(features, config)
        self.query_fn = query_fn

    def drone_map(self, frame: DroneFrame) -> torch.Tensor:
        if frame.name not in self._drone_maps:
            self._drone_maps[frame.name] = torch.from_numpy(np.ascontiguousarray(self.query_fn(frame)))[None]
        return self._drone_maps[frame.name]
