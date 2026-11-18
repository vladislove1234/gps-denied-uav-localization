"""Dead-reckoning drift: where the flight controller believes it is once GNSS is gone.

The drift is correlated, unlike the fixed-magnitude random-direction error of the
single-frame benchmark: a constant velocity bias that random-walks, a constant heading
error that rotates every step's motion, and an odometry scale error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from uavloc.geo import LatLon, offset_latlon, rotate_clockwise

N_AXES = 2   # east, north


@dataclass(frozen=True)
class InsConfig:
    velocity_bias_mps: float = 0.2            # std per axis of a velocity error drawn once per flight
    velocity_random_walk_mps: float = 0.02    # std per axis of the bias change over one second (scales with sqrt(dt))
    heading_bias_deg: float = 1.0             # std of a constant heading error, drawn once per flight
    speed_scale_std: float = 0.01             # std of a constant odometry scale error (0.01 = 1%)
    reported_yaw_noise_deg: float = 0.0       # std of per-correction noise on the yaw given to the corrector
    reported_altitude_noise_m: float = 0.0    # std of per-correction noise on the altitude given to the corrector
    initial_error_m: float = 0.0              # position error at the moment GNSS is lost, random direction


class InsSimulator:
    """One flight's drift. Drift and reported-value noise use separate streams, so the INS
    track stays the same whatever the correction interval."""

    def __init__(self, config: InsConfig, seed: int) -> None:
        self.config = config
        drift_seed, report_seed = np.random.SeedSequence(seed).spawn(2)
        self._drift_rng, self._report_rng = np.random.default_rng(drift_seed), np.random.default_rng(report_seed)
        self.velocity_bias_mps = self._drift_rng.normal(0.0, config.velocity_bias_mps, N_AXES)
        self.heading_bias_deg = float(self._drift_rng.normal(0.0, config.heading_bias_deg))
        self.speed_scale = 1.0 + float(self._drift_rng.normal(0.0, config.speed_scale_std))

    def start(self, truth: LatLon) -> LatLon:
        direction = self._drift_rng.uniform(0.0, 2 * math.pi)
        error_m = self.config.initial_error_m
        return offset_latlon(truth, error_m * math.sin(direction), error_m * math.cos(direction))

    def displacement_m(self, true_east_m: float, true_north_m: float, dt_s: float) -> tuple[float, float]:
        """The INS estimate of one step's `(east, north)` motion."""
        self.velocity_bias_mps = self.velocity_bias_mps + self._drift_rng.normal(
            0.0, self.config.velocity_random_walk_mps * math.sqrt(dt_s), N_AXES)
        east_m, north_m = rotate_clockwise(true_east_m * self.speed_scale, true_north_m * self.speed_scale,
                                           self.heading_bias_deg)
        return east_m + self.velocity_bias_mps[0] * dt_s, north_m + self.velocity_bias_mps[1] * dt_s

    def reported_yaw_deg(self, true_yaw_deg: float) -> float:
        return true_yaw_deg + self.heading_bias_deg + float(self._report_rng.normal(0.0, self.config.reported_yaw_noise_deg))

    def reported_altitude_m(self, true_altitude_m: float) -> float:
        return true_altitude_m + float(self._report_rng.normal(0.0, self.config.reported_altitude_noise_m))
