"""Build a feature extractor and correlator config from a run config, so what is logged is what ran."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from uavloc.data.annotated import VILLAGE_EXPORT, load_label_map
from uavloc.data.satellite import GeoImage
from uavloc.features.segmentation import SegmentationFeatures, label_probabilities, load_segmenter
from uavloc.features.vgg11 import Vgg11Features, load_vgg11
from uavloc.matching.correlator import CorrelatorConfig, FeatureExtractor
from uavloc.track import REPO_ROOT

METHODS = ("vgg11", "segmentation", "segmentation-labels")


def build_features(config: Mapping[str, Any], satellite: GeoImage, device: torch.device | str) -> FeatureExtractor:
    """`segmentation-labels`: drone side from the model, satellite side from a hand-labelled map."""
    method = config["method"]
    if method == "vgg11":
        return Vgg11Features(load_vgg11(REPO_ROOT / config["vgg11_checkpoint"], device), satellite,
                             config["drone_input_px"], config["search_scale"], device)
    architecture = config.get("seg_architecture", "unet")
    drone_model = load_segmenter(REPO_ROOT / config["seg_checkpoint_drone"], device, architecture)
    if method == "segmentation":
        sat_model = load_segmenter(REPO_ROOT / config["seg_checkpoint_sat"], device, architecture)
        return SegmentationFeatures(drone_model, sat_model, satellite,
                                    config["seg_channels"], config["drone_seg_gsd_m"], config["match_gsd_m"], device)
    if method == "segmentation-labels":
        labels = load_label_map(config["label_location"], config["label_date"],
                                config.get("label_dataset") or VILLAGE_EXPORT)
        return SegmentationFeatures.from_satellite_probabilities(
            drone_model, label_probabilities(labels, config["match_gsd_m"]), config["seg_channels"],
            config["drone_seg_gsd_m"], config["match_gsd_m"], device)
    raise ValueError(f"method must be one of {METHODS}, got {method!r}")


def correlator_config(config: Mapping[str, Any]) -> CorrelatorConfig:
    """Keys absent from older runs fall back to the pre-007 behaviour.

    Tuned decision values (`temperature`, `gate_s_max_m`, `gate_c_min`) are left at their defaults when the run
    logs them per fold (a dict); the caller applies each fold's values with `dataclasses.replace`.
    """
    weights = config.get("class_weights")
    tuned = {key: config[key] for key in ("temperature", "gate_s_max_m", "gate_c_min")
             if isinstance(config.get(key), (int, float))}
    return CorrelatorConfig(config["search_scale"], config["fov_w_deg"],
                            sat_feature_crop_pct=config["sat_feature_crop_pct"],
                            drone_feature_crop_pct=config["drone_feature_crop_pct"],
                            matching=config.get("matching", "cosine"),
                            class_weights=tuple(weights) if weights else None,
                            window=config.get("window", "rect"),
                            search_radius_m=config.get("search_radius_m"),
                            prior_sigma_m=config.get("prior_sigma_m"),
                            gate=config.get("gate", "none"),
                            heading_offsets_deg=tuple(config.get("heading_offsets_deg", (0.0,))),
                            heading_fine_offsets_deg=tuple(config.get("heading_fine_offsets_deg", (0.0,))),
                            scales=tuple(config.get("scales", (1.0,))),
                            pose_schedule=config.get("pose_schedule", "greedy"),
                            fine_pass_top_k=config.get("fine_pass_top_k", 1),
                            **tuned)
