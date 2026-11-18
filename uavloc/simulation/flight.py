"""Fly a route on dead reckoning, correct it every few seconds, and log every step."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from uavloc.geo import LatLon, east_north_m, haversine_m, offset_latlon
from uavloc.simulation.camera import DroneCamera
from uavloc.simulation.correctors import Corrector, Fix
from uavloc.simulation.ins import InsSimulator
from uavloc.simulation.route import Pose, route_length_m

TIME_EPS_S = 1e-9


@dataclass(frozen=True)
class FlightConfig:
    correction_interval_s: float = 5.0
    first_correction_s: float | None = None   # None: one interval after GNSS loss


@dataclass(frozen=True)
class FlightResult:
    track: pd.DataFrame          # one row per pose: truth, INS-only and corrected estimate, errors, fix columns
    fixes: dict[int, Fix]        # pose index -> fix, for the poses that attempted a correction
    poses: list[Pose]
    corrector_name: str

    @property
    def fix_steps(self) -> list[int]:
        return sorted(self.fixes)


def simulate_flight(poses: Sequence[Pose], camera: DroneCamera, ins: InsSimulator, corrector: Corrector,
                    config: FlightConfig) -> FlightResult:
    """The estimate moves with the INS increments; an applied fix replaces it. The INS-only track is never corrected."""
    ins_position = estimate = ins.start(poses[0].position)
    next_fix_s = config.correction_interval_s if config.first_correction_s is None else config.first_correction_s
    rows, fixes = [], {}
    for step, pose in enumerate(poses):
        if step:
            previous = poses[step - 1]
            move_m = ins.displacement_m(*east_north_m(previous.position, pose.position), pose.t_s - previous.t_s)
            ins_position, estimate = offset_latlon(ins_position, *move_m), offset_latlon(estimate, *move_m)
        fix = None
        if pose.t_s + TIME_EPS_S >= next_fix_s:
            fix = corrector.correct(camera.view(pose), estimate, ins.reported_yaw_deg(pose.yaw_deg),
                                    ins.reported_altitude_m(pose.altitude_m))
            fixes[step] = fix
            next_fix_s += config.correction_interval_s
            estimate = fix.position if fix.applied else estimate
        rows.append(track_row(pose, ins_position, estimate, fix))
    return FlightResult(pd.DataFrame(rows), fixes, list(poses), corrector.name)


def track_row(pose: Pose, ins_position: LatLon, estimate: LatLon, fix: Fix | None) -> dict:
    s1_m, s2_m = fix.uncertainty_m if fix is not None and fix.uncertainty_m else (np.nan, np.nan)
    return {
        "t_s": pose.t_s, "true_lat": pose.position.lat, "true_lon": pose.position.lon,
        "yaw_deg": pose.yaw_deg, "altitude_m": pose.altitude_m,
        "ins_lat": ins_position.lat, "ins_lon": ins_position.lon,
        "est_lat": estimate.lat, "est_lon": estimate.lon,
        "ins_error_m": haversine_m(ins_position, pose.position), "est_error_m": haversine_m(estimate, pose.position),
        "fix_attempted": fix is not None, "decision": fix.decision if fix else "",
        "peak_z": fix.peak_z if fix else np.nan, "s1_m": s1_m, "s2_m": s2_m,
        "fix_ms": fix.elapsed_ms if fix else np.nan,
    }


def flight_summary(result: FlightResult) -> dict[str, float | int | str]:
    """End-point errors (the headline), plus along-track errors and what the corrector did."""
    track, attempts = result.track, result.track[result.track["fix_attempted"]]
    distance_m = route_length_m(result.poses)
    summary = {
        "corrector": result.corrector_name, "duration_s": float(track["t_s"].iloc[-1]), "distance_m": distance_m,
        "end_error_ins_m": float(track["ins_error_m"].iloc[-1]), "end_error_corrected_m": float(track["est_error_m"].iloc[-1]),
        "mean_error_ins_m": float(track["ins_error_m"].mean()), "mean_error_corrected_m": float(track["est_error_m"].mean()),
        "max_error_ins_m": float(track["ins_error_m"].max()), "max_error_corrected_m": float(track["est_error_m"].max()),
        "ins_drift_share_of_distance": float(track["ins_error_m"].iloc[-1] / distance_m) if distance_m else np.nan,
        "fix_attempts": len(attempts), "fixes_applied": int(sum(f.applied for f in result.fixes.values())),
        "mean_fix_ms": float(attempts["fix_ms"].mean()) if len(attempts) else np.nan,
    }
    summary.update({f"decision_{d}": int(n) for d, n in attempts["decision"].value_counts().items()})
    return summary
