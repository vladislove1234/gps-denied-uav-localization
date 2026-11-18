"""How far the drone may fly without a fix before the matcher can no longer find it."""

from __future__ import annotations

import math

from uavloc.simulation.ins import InsConfig

BISECTION_STEPS = 60
SEARCH_CEILING_M = 1e6


def drift_sigma_m(distance_m: float, speed_mps: float, ins: InsConfig) -> float:
    """1-sigma radial INS position error after `distance_m` without a fix, under `InsConfig`'s model.

    Velocity bias grows with time, its random walk with time^1.5, heading and scale errors with distance.
    """
    t_s = distance_m / speed_mps
    per_axis_var = (ins.velocity_bias_mps * t_s) ** 2 + ins.velocity_random_walk_mps ** 2 * t_s ** 3 / 3
    heading_m = distance_m * math.radians(ins.heading_bias_deg)
    scale_m = distance_m * ins.speed_scale_std
    return math.sqrt(2 * per_axis_var + heading_m ** 2 + scale_m ** 2)


def max_blind_distance_m(speed_mps: float, search_half_range_m: float, ins: InsConfig, sigmas: float,
                         ekf_coast_s: float | None = None) -> float:
    """Longest blind stretch: drift at `sigmas` must stay inside the search window, and within the EKF's coast time."""
    low, high = 0.0, SEARCH_CEILING_M
    for _ in range(BISECTION_STEPS):
        mid = (low + high) / 2
        if sigmas * drift_sigma_m(mid, speed_mps, ins) <= search_half_range_m:
            low = mid
        else:
            high = mid
    return low if ekf_coast_s is None else min(low, ekf_coast_s * speed_mps)
