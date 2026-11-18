"""Inject a known position error, let a correlator fix it, measure what is left."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from uavloc.data.drone_frames import DroneFrame
from uavloc.geo import LatLon, haversine_m, offset_latlon
from uavloc.matching.correlator import (CorrelatorConfig, PositionCorrelator, SearchSurface, decide, ridge_bearing_deg,
                                        shifted)

FULL_CIRCLE_DEG = 360.0
P90 = 90
HARM_MARGIN_M = 2.0   # a correction "harms" when it leaves the position worse than the prior by more than this
DECISIONS = ("accept", "across", "reject")


@dataclass(frozen=True)
class Draw:
    """One injected prior and its scored search surface; `surface` is None when the patch left the raster."""

    frame: DroneFrame
    noise_m: float
    bearing_deg: float
    prior: LatLon
    surface: SearchSurface | None
    reachable: bool


def injected_priors(truth: LatLon, noise_m: float, n_directions: int, phase_deg: float) -> list[tuple[float, LatLon]]:
    """`n_directions` priors `noise_m` from `truth`, evenly spaced bearings starting at `phase_deg`."""
    priors = []
    for k in range(n_directions):
        bearing_deg = phase_deg + k * FULL_CIRCLE_DEG / n_directions
        bearing = math.radians(bearing_deg)
        priors.append((bearing_deg, offset_latlon(truth, noise_m * math.sin(bearing), noise_m * math.cos(bearing))))
    return priors


def benchmark_draws(correlator: PositionCorrelator, frames: list[DroneFrame], noise_m: float,
                    n_directions: int, seed: int, progress: bool = False) -> list[Draw]:
    """Score every (frame, direction) once; `score_draws` then applies any decision config to them."""
    phases = np.random.default_rng(seed).uniform(0, FULL_CIRCLE_DEG / n_directions, len(frames))
    draws = []
    pairs = zip(frames, phases)
    if progress:
        pairs = tqdm(pairs, total=len(frames), desc=f"{noise_m:g} m", unit="frame")
    for frame, phase_deg in pairs:
        for bearing_deg, prior in injected_priors(frame.position, noise_m, n_directions, phase_deg):
            try:
                surface = correlator.surface(frame, prior)
            except ValueError:
                surface = None
            draws.append(Draw(frame, noise_m, bearing_deg, prior, surface, correlator.reachable(frame, prior)))
    return draws


def score_draws(draws: list[Draw], config: CorrelatorConfig) -> pd.DataFrame:
    """One row per draw under `config`'s decision settings. A patch that left the raster keeps the prior."""
    rows = []
    for draw in draws:
        truth = draw.frame.position
        row = {"frame": draw.frame.name, "noise_m": draw.noise_m, "bearing_deg": draw.bearing_deg,
               "prior_lat": draw.prior.lat, "prior_lon": draw.prior.lon,
               "prior_error_m": haversine_m(draw.prior, truth), "reachable": draw.reachable}
        if draw.surface is None:
            rows.append({**row, "corrected_lat": draw.prior.lat, "corrected_lon": draw.prior.lon,
                         "error_m": row["prior_error_m"], "peak_z": float("nan"), "status": "out-of-raster",
                         "decision": "reject"})
            continue
        correction = decide(draw.surface, config)
        s1_m, s2_m = correction.uncertainty_m
        yaw_deg = draw.surface.yaw_deg   # the logged heading, plus the winning offset under a pose search
        rows.append({
            **row, "corrected_lat": correction.position.lat, "corrected_lon": correction.position.lon,
            "error_m": haversine_m(correction.position, truth), "peak_z": correction.peak_z, "status": "ok",
            "decision": correction.decision, "s1_m": s1_m, "s2_m": s2_m,
            "ridge_bearing_deg": ridge_bearing_deg(correction, yaw_deg), "coverage": correction.coverage,
            "second_peak_ratio": correction.second_peak_ratio,
            "map_offset_m": float(np.hypot(*correction.map_offset_m)),
            "applied_offset_m": float(np.hypot(*correction.offset_m)),
            "map_error_m": haversine_m(shifted(draw.prior, correction.map_offset_m, yaw_deg), truth),
            "across_error_m": haversine_m(shifted(draw.prior, correction.across_offset_m, yaw_deg), truth),
            "heading_offset_deg": correction.heading_offset_deg, "scale": correction.scale,
            "pose_hypotheses": draw.surface.hypotheses, "pose_hypotheses_skipped": draw.surface.skipped_hypotheses,
        })
    return pd.DataFrame(rows)


def run_benchmark(correlator: PositionCorrelator, frames: list[DroneFrame], noise_m: float,
                  n_directions: int, seed: int, progress: bool = False) -> pd.DataFrame:
    """One row per (frame, direction). A patch that leaves the raster keeps the prior (`status` says so)."""
    return score_draws(benchmark_draws(correlator, frames, noise_m, n_directions, seed, progress), correlator.config)


def error_stats(records: pd.DataFrame) -> dict[str, float]:
    errors = records["error_m"]
    stats = {
        "mean_position_error_m": float(errors.mean()),
        "median_position_error_m": float(errors.median()),
        "p90_position_error_m": float(np.percentile(errors, P90)),
        "improved_share": float((errors < records["prior_error_m"]).mean()),
        "reachable_share": float(records["reachable"].mean()),
        "mean_error_reachable_m": float(errors[records["reachable"]].mean()),
        "out_of_raster": int((records["status"] != "ok").sum()),
        "draws": len(records),
        "harm_share": float((errors > records["prior_error_m"] + HARM_MARGIN_M).mean()),
    }
    if "decision" in records:
        stats.update({f"{d}_share": float((records["decision"] == d).mean()) for d in DECISIONS})
    return stats


def per_frame_error(records: pd.DataFrame, noise_m: float) -> pd.Series:
    """Mean error per frame over its directions at one injected level."""
    return records[records["noise_m"] == noise_m].groupby("frame")["error_m"].mean()


def paired_frame_bootstrap(records: pd.DataFrame, reference: pd.DataFrame, noise_m: float,
                           n_bootstrap: int, ci_pct: float, seed: int) -> dict[str, float]:
    """Mean of `records - reference` per frame, with a frame-bootstrap CI and the frames where `records` is better."""
    diff = (per_frame_error(records, noise_m) - per_frame_error(reference, noise_m)).to_numpy()
    rng = np.random.default_rng(seed)
    boot = diff[rng.integers(0, len(diff), (n_bootstrap, len(diff)))].mean(1)
    lo, hi = np.percentile(boot, [(100 - ci_pct) / 2, 100 - (100 - ci_pct) / 2])
    return {"diff_m": float(diff.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
            "frames_better": int((diff < 0).sum()), "frames": len(diff)}


def summarise(records: pd.DataFrame, subsets: dict[str, set[str]], primary_noise_m: float) -> dict[str, float]:
    """Flat metrics: the primary noise level unprefixed, every level and every subset suffixed."""
    primary = records[records["noise_m"] == primary_noise_m]
    metrics = error_stats(primary)
    for noise_m, group in records.groupby("noise_m"):
        for key, value in error_stats(group).items():
            metrics[f"{key}_at_{noise_m:g}m"] = value
    for name, members in subsets.items():
        subset_stats = error_stats(primary[primary["frame"].isin(members)])
        metrics[f"mean_position_error_m_{name}"] = subset_stats["mean_position_error_m"]
        metrics[f"draws_{name}"] = subset_stats["draws"]
    return metrics


def median_ms_per_correction(correlator: PositionCorrelator, frames: list[DroneFrame], noise_m: float,
                             n_directions: int, repeats: int) -> float:
    """Median over repeats of the mean wall time of one correction, drone inference included (cache cleared)."""
    priors = [injected_priors(f.position, noise_m, n_directions, 0.0)[0][1] for f in frames]
    times_ms = []
    for _ in range(repeats):
        correlator.clear_cache()
        start = time.perf_counter()
        for frame, prior in zip(frames, priors):
            correlator.correct(frame, prior)
        times_ms.append((time.perf_counter() - start) * 1000 / len(frames))
    return float(np.median(times_ms))
