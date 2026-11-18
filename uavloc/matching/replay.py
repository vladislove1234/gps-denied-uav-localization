"""Replay one logged benchmark draw and keep every intermediate, for step-by-step diagnosis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from uavloc.data.drone_frames import DroneFrame
from uavloc.geo import LatLon
from uavloc.matching.correlator import Correction, PositionCorrelator, crop_border
from uavloc.track import experiment_dir


@dataclass(frozen=True)
class Replay:
    """One correction with the feature maps it matched; offsets are `(right_m, down_m)` in the prior's patch."""

    frame: DroneFrame
    prior: LatLon
    correction: Correction
    drone_map: torch.Tensor       # (1, C, H, W), border-cropped as matched
    satellite_map: torch.Tensor   # (1, C, H_s, W_s), border-cropped as matched
    footprint_m: tuple[float, float]
    true_offset_m: tuple[float, float]
    shortfall_m: float            # truth's distance outside the search range; 0 when reachable

    def window(self, offset_m: tuple[float, float]) -> torch.Tensor | None:
        """The satellite window the drone map was compared with when centred at `offset_m`; None outside the range."""
        cell = self.correction.heatmap_cell(offset_m)
        if cell is None:
            return None
        h, w = self.drone_map.shape[2:]
        return self.satellite_map[:, :, cell[0]:cell[0] + h, cell[1]:cell[1] + w]


def replay(correlator: PositionCorrelator, frame: DroneFrame, prior: LatLon) -> Replay:
    captured = {}
    previous_hook = correlator.features_hook
    correlator.features_hook = lambda drone, satellite: captured.update(drone=drone, satellite=satellite)
    try:
        correction = correlator.correct(frame, prior)
    finally:
        correlator.features_hook = previous_hook
    cfg = correlator.config
    return Replay(frame, prior, correction,
                  crop_border(captured["drone"], cfg.drone_feature_crop_pct)[0].cpu(),
                  crop_border(captured["satellite"], cfg.sat_feature_crop_pct)[0].cpu(),
                  correlator.footprint_m(frame), correlator.patch_offset_m(prior, frame.position, frame.yaw_deg),
                  correlator.reach_shortfall_m(frame, prior))


def load_records(exp: str, run_id: str) -> pd.DataFrame:
    """A run's per-draw records (`artifacts/records-<run>.csv`)."""
    return pd.read_csv(experiment_dir(exp) / "artifacts" / f"records-{run_id}.csv")


def score_at(replay_: Replay, offset_m: tuple[float, float]) -> float:
    """The z-scored heatmap value at `offset_m`; NaN outside the search range."""
    cell = replay_.correction.heatmap_cell(offset_m)
    return float(replay_.correction.heatmap[cell]) if cell else np.nan
