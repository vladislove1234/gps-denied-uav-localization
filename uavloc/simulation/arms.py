"""Sequence-benchmark arms (027, 028): the correlator, and an estimator factory per arm name."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import torch

from uavloc.data.satellite import GeoImage
from uavloc.features.factory import build_features, correlator_config
from uavloc.filtering.belief import BlindRule, WindowConfig
from uavloc.filtering.empty_view import EmptyViewModel
from uavloc.filtering.hypotheses import HypothesisTracker, HypothesisTrackerConfig
from uavloc.filtering.ins_error import InsErrorFilterConfig, InsErrorParticleFilter, InsErrorPrior
from uavloc.filtering.particles import ParticleFilter, ParticleFilterConfig
from uavloc.matching.correlator import CAMERA_ASPECT_W_OVER_H, PositionCorrelator
from uavloc.matching.learned import LearnedScorer, build_head
from uavloc.simulation.estimators import InsOnlyEstimator, ReplaceEstimator
from uavloc.simulation.sequence import EstimatorFactory

ARMS = ("ins", "replace", "gated", "particles", "hypotheses", "ins-error")
FILTERS = ("particles", "hypotheses", "ins-error")


@dataclass(frozen=True)
class FilterParams:
    temperature: float
    q_pos_m_per_sqrt_s: float
    n_particles: int
    k_max: int
    peaks_per_hypothesis: int
    peak_nms_m: float
    merge_distance_m: float
    child_floor_sigma_m: float
    window_sigmas: float
    window_cap_half_m: float
    window_margin_m: float
    empty_view: EmptyViewModel | None = None   # particles only (032)
    empty_view_weight: float = 0.0
    blind_rule: BlindRule | None = None        # 034: structure test on the drone frame


@dataclass(frozen=True)
class HeadSpec:
    """A trained 023 correlation head: its checkpoint and the architecture it was trained with."""

    checkpoint: Path
    encoder_channels: tuple[int, ...]
    refine_channels: int


def build_correlator(map_raster: GeoImage, correlator: Mapping[str, Any], device: str,
                     head: HeadSpec | None = None) -> PositionCorrelator:
    """The 007.1 correlator (rect 1.7, flat prior, no gate) on `map_raster`: weighted NCC, or the head's scores."""
    features = build_features(correlator, map_raster, device)
    config = replace(correlator_config(correlator), aspect_w_over_h=CAMERA_ASPECT_W_OVER_H)
    if head is None:
        return PositionCorrelator(features, config)
    model = build_head("head", config.class_weights, head.encoder_channels, head.refine_channels)
    model.load_state_dict(torch.load(head.checkpoint, map_location="cpu"))
    return PositionCorrelator(features, replace(config, matching="learned"), scorer=LearnedScorer(model, device))


def gated(correlator: PositionCorrelator, gate: Mapping[str, float]) -> PositionCorrelator:
    """The same features under the 007 full gate."""
    config = replace(correlator.config, gate="full", temperature=gate["temperature"], gate_s_max_m=gate["s_max_m"],
                     gate_c_min=gate["c_min"])
    return PositionCorrelator(correlator.features, config)


def estimator_factory(arm: str, params: FilterParams | None = None, gate: Mapping[str, float] | None = None,
                      ins_prior: InsErrorPrior | None = None) -> EstimatorFactory:
    """`ins-error` needs `ins_prior`: the IMU-spec error the particles start from."""
    if arm == "ins":
        return lambda correlator, spec: InsOnlyEstimator()
    if arm == "replace":
        return lambda correlator, spec: ReplaceEstimator(correlator)
    if arm == "gated":
        return lambda correlator, spec: ReplaceEstimator(gated(correlator, gate), name="gated")
    window = WindowConfig(params.window_cap_half_m, params.window_margin_m, params.blind_rule)
    if arm == "particles":
        config = ParticleFilterConfig(params.n_particles, params.temperature, params.q_pos_m_per_sqrt_s, window=window,
                                      empty_view=params.empty_view, empty_view_weight=params.empty_view_weight)
        return lambda correlator, spec: ParticleFilter(correlator, config, spec.seed)
    if arm == "hypotheses":
        config = HypothesisTrackerConfig(params.k_max, params.peaks_per_hypothesis, params.peak_nms_m,
                                         params.merge_distance_m, params.child_floor_sigma_m, params.window_sigmas,
                                         params.temperature, params.q_pos_m_per_sqrt_s, window)
        return lambda correlator, spec: HypothesisTracker(correlator, config)
    if arm == "ins-error":
        config = InsErrorFilterConfig(ins_prior, params.n_particles, params.temperature, params.q_pos_m_per_sqrt_s,
                                      window=window)
        return lambda correlator, spec: InsErrorParticleFilter(correlator, config, spec.seed)
    raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
