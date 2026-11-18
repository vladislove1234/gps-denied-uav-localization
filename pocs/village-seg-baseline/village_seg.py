"""Shim: the code moved to `uavloc.segmentation.village_seg`; the notebooks that put this folder on `sys.path` import it from here."""

from uavloc.data.annotated import IGNORE_LABEL as IGNORE
from uavloc.segmentation.metrics import iou_scores
from uavloc.segmentation.village_seg import (
    BLOCK_GAP_MS,
    CENTRE_THIN,
    DRONE_ASPECT_W_OVER_H,
    DRONE_HALF_FOV_H_DEG,
    DRONE_HALF_FOV_W_DEG,
    DRONE_RARE_CLASSES,
    SAT_RARE_CLASSES,
    DroneImage,
    DroneSampler,
    SatelliteSampler,
    SatImage,
    Sampler,
    agreement,
    augment,
    ban_holdouts,
    batch_of,
    domain_bn,
    drone_at_gsd,
    drone_footprint,
    load_drone,
    load_satellite,
    pad_crop,
    predict,
    read_jsonl,
    split_by_footprint,
    valid_corners,
)

__all__ = [
    "IGNORE", "iou_scores", "BLOCK_GAP_MS", "CENTRE_THIN", "DRONE_ASPECT_W_OVER_H", "DRONE_HALF_FOV_H_DEG",
    "DRONE_HALF_FOV_W_DEG", "DRONE_RARE_CLASSES", "SAT_RARE_CLASSES", "DroneImage", "DroneSampler",
    "SatelliteSampler", "SatImage", "Sampler", "agreement", "augment", "ban_holdouts", "batch_of", "domain_bn",
    "drone_at_gsd", "drone_footprint", "load_drone", "load_satellite", "pad_crop", "predict", "read_jsonl",
    "split_by_footprint", "valid_corners",
]
