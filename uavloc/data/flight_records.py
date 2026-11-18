"""Per-flight and per-fix error tables of a sequence-filter run, and the track of one long flight."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from uavloc.geo import LatLon, haversine_m
from uavloc.track import experiment_dir, load_runs

__all__ = ["find_run_id", "load_flight_table", "load_fix_table", "load_track", "distance_along_track_km"]

FLIGHT_KEY = ["village", "route_index"]
FIX_TIME_COLUMN = "t_s"
M_PER_KM = 1000.0


def _matches(config: Mapping[str, Any], wanted: Mapping[str, Any]) -> bool:
    return all(config.get(key) == value for key, value in wanted.items())


def find_run_id(exp: str, **config: Any) -> str:
    """Id of the one run of ``exp`` whose config has all the given values; raises if none or several."""
    ids = [run["run"] for run in load_runs(exp) if _matches(run["config"], config)]
    if len(ids) != 1:
        raise LookupError(f"{len(ids)} runs of {exp} match {config}: {ids}")
    return ids[0]


def load_flight_table(exp: str, run_id: str) -> pd.DataFrame:
    """One row per evaluated flight: distance, duration, end error."""
    return pd.read_csv(experiment_dir(exp, create=False) / "artifacts" / f"flights-{run_id}.csv")


def load_fix_table(exp: str, run_id: str) -> pd.DataFrame:
    """One row per fix time of every flight, with ``distance_m`` flown at that time (constant speed)."""
    artifacts = experiment_dir(exp, create=False) / "artifacts"
    fixes = pd.read_csv(artifacts / f"fixes-{run_id}.csv")
    flights = pd.read_csv(artifacts / f"flights-{run_id}.csv")[FLIGHT_KEY + ["distance_m", "duration_s"]]
    fixes = fixes.merge(flights, on=FLIGHT_KEY, how="left", validate="many_to_one")
    fixes["distance_m"] = fixes[FIX_TIME_COLUMN] / fixes["duration_s"] * fixes["distance_m"]
    return fixes.drop(columns="duration_s")


def load_track(path: Path) -> pd.DataFrame:
    """Track of one flown route (truth, INS, estimate and their errors per sample)."""
    return pd.read_csv(path)


def distance_along_track_km(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """Cumulative path length in km of a lat/lon polyline."""
    points = [LatLon(lat, lon) for lat, lon in zip(lat_deg, lon_deg)]
    steps_m = [haversine_m(a, b) for a, b in zip(points[:-1], points[1:])]
    return np.concatenate([[0.0], np.cumsum(steps_m)]) / M_PER_KM
