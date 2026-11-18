"""Many simulated flights with stateful estimators, and their village-level statistics (027).

`fly` is `simulate_flight` for an `Estimator`; it also scores, per fix, whether the truth
was inside the searched window and inside the estimator's 99 % ellipse. `run_flights`
flies a route set village by village, segmenting each map once.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Callable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from uavloc.data.rasters import load_raster
from uavloc.data.satellite import GeoImage
from uavloc.datasets import data_root, load_manifest
from uavloc.geo import LatLon, east_north_m, haversine_m, offset_latlon
from uavloc.matching.correlator import PositionCorrelator
from uavloc.simulation.camera import CameraConfig, DroneCamera, RasterCamera
from uavloc.simulation.correctors import Fix, RenderedFrame
from uavloc.simulation.estimators import Estimator, FixRecord, SimulatedFrames
from uavloc.simulation.flight import TIME_EPS_S, FlightResult
from uavloc.simulation.ins import InsConfig, InsSimulator
from uavloc.simulation.route import Pose, route_length_m, sample_route
from uavloc.simulation.schedule import EveryInterval, FixSchedule

CHI2_2D_99 = 9.21          # squared Mahalanobis radius of the 99 % ellipse in 2-D
RECOVERED_M = 10.0         # cold start: a flight ending this close counts as recovered
SKIPPED_DECISIONS = ("blind", "out-of-raster", "reject")
ROUTES_FILE = "routes.json"


@dataclass(frozen=True)
class FlightSpec:
    village: str
    route_index: int
    seed: int
    split: str
    waypoints: tuple[LatLon, ...]


@dataclass(frozen=True)
class FlightSetConfig:
    altitude_m: float
    speed_mps: float
    step_s: float
    correction_interval_s: float
    fov_w_deg: float
    ins: InsConfig
    drone_date: str = "current"
    map_date: str = "newest"


@dataclass(frozen=True)
class FlightOutcome:
    spec: FlightSpec
    track: pd.DataFrame      # one row per pose
    fixes: pd.DataFrame      # one row per attempted fix
    distance_m: float


def load_flight_set(name: str, split: str | None = None) -> tuple[FlightSetConfig, list[FlightSpec]]:
    """The route set's flight settings and its routes, optionally one split only."""
    config = load_manifest(name)["config"]
    set_config = FlightSetConfig(config["altitude_m"], config["speed_mps"], config["step_s"],
                                 config["correction_interval_s"], config["fov_w_deg"], InsConfig(**config["ins"]),
                                 config["drone_date"], config["map_date"])
    routes = json.loads((data_root(name) / ROUTES_FILE).read_text())
    specs = [FlightSpec(r["village"], r["route_index"], r["seed"], r["split"], tuple(LatLon(*w) for w in r["waypoints"]))
             for r in routes]
    return set_config, [s for s in specs if split is None or s.split == split]


def truth_in_window(record: FixRecord, truth: LatLon) -> bool | float:
    if record.window is None or record.extent_m is None:
        return np.nan
    right, down = PositionCorrelator.patch_offset_m(record.window.center, truth, record.window.yaw_deg)
    left, right_edge, bottom, top = record.extent_m
    return bool(left <= right <= right_edge and top <= down <= bottom)


def truth_in_ellipse(estimate: LatLon, cov_m2: np.ndarray, truth: LatLon) -> bool | float:
    if not np.isfinite(cov_m2).all():
        return np.nan
    error = np.asarray(east_north_m(estimate, truth))
    return bool(error @ np.linalg.pinv(cov_m2) @ error <= CHI2_2D_99)


class FlightRun:
    """One flight stepped pose by pose, so a long route can swap the camera's raster and the estimator's map
    between steps. The estimator gets the INS increments, and the reported yaw and altitude at each fix."""

    def __init__(self, first_pose: Pose, ins: InsSimulator, estimator: Estimator, schedule: FixSchedule,
                 sigma0_m: float) -> None:
        self.ins, self.estimator, self.schedule = ins, estimator, schedule
        self.ins_position = ins.start(first_pose.position)
        estimator.start(self.ins_position, sigma0_m)
        self.frames = SimulatedFrames()
        self.previous: Pose | None = None
        self.rows: list[dict] = []
        self.fix_rows: list[dict] = []

    def step(self, pose: Pose, camera: DroneCamera) -> None:
        ins, estimator = self.ins, self.estimator
        if self.previous is not None:
            dt_s = pose.t_s - self.previous.t_s
            move_m = ins.displacement_m(*east_north_m(self.previous.position, pose.position), dt_s)
            self.ins_position = offset_latlon(self.ins_position, *move_m)
            estimator.predict(move_m, dt_s)
        self.previous = pose
        record = None
        prior, _ = estimator.estimate()
        if self.schedule.due(pose, prior):
            frame = self.frames.frame(camera.view(pose), prior, ins.reported_yaw_deg(pose.yaw_deg),
                                      ins.reported_altitude_m(pose.altitude_m))
            record = estimator.update(frame)
        estimate, cov = estimator.estimate()
        self.rows.append({"t_s": pose.t_s, "true_lat": pose.position.lat, "true_lon": pose.position.lon,
                          "ins_lat": self.ins_position.lat, "ins_lon": self.ins_position.lon,
                          "est_lat": estimate.lat, "est_lon": estimate.lon,
                          "ins_error_m": haversine_m(self.ins_position, pose.position),
                          "est_error_m": haversine_m(estimate, pose.position),
                          "cov_ee": cov[0, 0], "cov_en": cov[0, 1], "cov_nn": cov[1, 1]})
        if record is not None:
            self.fix_rows.append({
                "step": len(self.rows) - 1, "t_s": pose.t_s, "decision": record.decision, "fix_ms": record.elapsed_ms,
                "truth_in_window": truth_in_window(record, pose.position),
                "truth_in_ellipse": truth_in_ellipse(estimate, cov, pose.position),
                "error_before_m": haversine_m(prior, pose.position), "error_after_m": haversine_m(estimate, pose.position),
                "peak_z": record.peak_z, "coverage": record.coverage, "peak_at_edge": record.peak_at_edge,
                "window_half_right_m": record.window.half_m[0] if record.window else np.nan,
                "window_half_down_m": record.window.half_m[1] if record.window else np.nan,
                "window_capped": record.window.capped if record.window else False,
                "effective_size": record.effective_size, "live_hypotheses": record.live_hypotheses,
                "dominance": record.dominance,
            })

    def tables(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """`(track, fixes)`: one row per pose, one row per attempted fix."""
        return pd.DataFrame(self.rows), pd.DataFrame(self.fix_rows)


def fly(poses: Sequence[Pose], camera: DroneCamera, ins: InsSimulator, estimator: Estimator,
        correction_interval_s: float, sigma0_m: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`(track, fixes)` with a fix every `correction_interval_s`."""
    run = FlightRun(poses[0], ins, estimator, EveryInterval(correction_interval_s), sigma0_m)
    for pose in poses:
        run.step(pose, camera)
    return run.tables()


def fix_frames(poses: Sequence[Pose], camera: DroneCamera, ins: InsSimulator,
               correction_interval_s: float) -> Iterator[tuple[Pose, RenderedFrame]]:
    """The views a flight corrects with, framed at the true position (no estimator): for calibration (032)."""
    frames = SimulatedFrames()
    next_fix_s = correction_interval_s
    for pose in poses:
        if pose.t_s + TIME_EPS_S >= next_fix_s:
            yield pose, frames.frame(camera.view(pose), pose.position, ins.reported_yaw_deg(pose.yaw_deg),
                                     ins.reported_altitude_m(pose.altitude_m))
            next_fix_s += correction_interval_s


EstimatorFactory = Callable[[PositionCorrelator, FlightSpec], Estimator]


def run_flights(set_config: FlightSetConfig, specs: Sequence[FlightSpec], initial_error_m: float, sigma0_m: float,
                build_correlator: Callable[[GeoImage], PositionCorrelator],
                estimators: Mapping[str, EstimatorFactory]) -> dict[str, list[FlightOutcome]]:
    """Every route under every estimator, village by village; each village's map is segmented once.

    All estimators fly the same INS realisation per route (seeded by the route).
    """
    by_village: dict[str, list[FlightSpec]] = defaultdict(list)
    for spec in specs:
        by_village[spec.village].append(spec)
    ins_config = replace(set_config.ins, initial_error_m=initial_error_m)
    camera_config = CameraConfig(set_config.fov_w_deg)
    outcomes: dict[str, list[FlightOutcome]] = {name: [] for name in estimators}
    bar = tqdm(total=len(specs) * len(estimators), desc="flights")
    for village, village_specs in by_village.items():
        drone_raster = load_raster({"kind": "multidate", "place": village, "date": set_config.drone_date})
        correlator = build_correlator(load_raster({"kind": "multidate", "place": village, "date": set_config.map_date}))
        camera = RasterCamera(drone_raster, camera_config)
        for spec in village_specs:
            poses = sample_route(spec.waypoints, set_config.speed_mps, set_config.step_s, set_config.altitude_m)
            for name, make_estimator in estimators.items():
                track, fixes = fly(poses, camera, InsSimulator(ins_config, spec.seed), make_estimator(correlator, spec),
                                   set_config.correction_interval_s, sigma0_m)
                outcomes[name].append(FlightOutcome(spec, track, fixes, route_length_m(poses)))
                bar.update()
                bar.set_postfix(village=village[:12], arm=name[:12], end_m=f"{track['est_error_m'].iloc[-1]:.1f}")
    bar.close()
    return outcomes


def per_flight_table(outcomes: Sequence[FlightOutcome]) -> pd.DataFrame:
    rows = []
    for o in outcomes:
        track, fixes = o.track, o.fixes
        rows.append({
            "village": o.spec.village, "route_index": o.spec.route_index, "split": o.spec.split,
            "distance_m": o.distance_m, "duration_s": float(track["t_s"].iloc[-1]),
            "end_error_m": float(track["est_error_m"].iloc[-1]), "end_error_ins_m": float(track["ins_error_m"].iloc[-1]),
            "mean_error_m": float(track["est_error_m"].mean()),
            "p90_error_m": float(track["est_error_m"].quantile(0.9)), "max_error_m": float(track["est_error_m"].max()),
            "fixes": len(fixes),
            "skipped_share": float(fixes["decision"].isin(SKIPPED_DECISIONS).mean()) if len(fixes) else np.nan,
            "median_fix_ms": float(fixes["fix_ms"].median()) if len(fixes) else np.nan,
            "truth_in_window_share": float(fixes["truth_in_window"].astype(float).mean()) if len(fixes) else np.nan,
            "truth_in_ellipse_share": float(fixes["truth_in_ellipse"].astype(float).mean()) if len(fixes) else np.nan,
            "mean_coverage": float(fixes["coverage"].mean()) if len(fixes) else np.nan,
        })
    table = pd.DataFrame(rows)
    table["diverged"] = table["end_error_m"] > table["end_error_ins_m"]
    return table


def all_fixes(outcomes: Sequence[FlightOutcome]) -> pd.DataFrame:
    return pd.concat([o.fixes.assign(village=o.spec.village, route_index=o.spec.route_index) for o in outcomes],
                     ignore_index=True)


def flight_metrics(outcomes: Sequence[FlightOutcome]) -> dict[str, float]:
    """The scalars a run logs: the bar's metrics first."""
    table, fixes = per_flight_table(outcomes), all_fixes(outcomes)
    window_known = fixes["truth_in_window"].dropna().astype(float)
    ellipse_known = fixes["truth_in_ellipse"].dropna().astype(float)
    return {
        "mean_end_error_m": float(table["end_error_m"].mean()),
        "divergence_share": float(table["diverged"].mean()),
        "median_fix_ms": float(fixes["fix_ms"].median()),
        "truth_in_window_share": float(window_known.mean()) if len(window_known) else float("nan"),
        "median_end_error_m": float(table["end_error_m"].median()),
        "recovered_share": float((table["end_error_m"] <= RECOVERED_M).mean()),
        "mean_error_m": float(table["mean_error_m"].mean()),
        "mean_p90_error_m": float(table["p90_error_m"].mean()),
        "mean_max_error_m": float(table["max_error_m"].mean()),
        "mean_end_error_ins_m": float(table["end_error_ins_m"].mean()),
        "truth_in_ellipse_share": float(ellipse_known.mean()) if len(ellipse_known) else float("nan"),
        "skipped_share": float(fixes["decision"].isin(SKIPPED_DECISIONS).mean()),
        "window_capped_share": float(fixes["window_capped"].mean()),
        "peak_at_edge_share": float(fixes["peak_at_edge"].mean()),
        "flights": len(table),
        "villages": int(table["village"].nunique()),
        "fixes": len(fixes),
    }


def village_bootstrap(a: pd.DataFrame, b: pd.DataFrame, column: str, n_resamples: int,
                      seed: int) -> tuple[float, float, float]:
    """Mean of `a[column] - b[column]` paired by flight, with a 95 % CI resampling villages (both routes together)."""
    keys = ["village", "route_index"]
    paired = a[keys + [column]].merge(b[keys + [column]], on=keys, suffixes=("_a", "_b"))
    paired["diff"] = paired[f"{column}_a"].astype(float) - paired[f"{column}_b"].astype(float)
    by_village = paired.groupby("village")["diff"].agg(["sum", "count"])
    sums, counts = by_village["sum"].to_numpy(), by_village["count"].to_numpy()
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(sums), (n_resamples, len(sums)))
    means = sums[picks].sum(1) / counts[picks].sum(1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(paired["diff"].mean()), float(low), float(high)


def as_flight_result(outcome: FlightOutcome, poses: Sequence[Pose], name: str) -> FlightResult:
    """A filter flight in the single-corrector form `FlightVideo` renders; each fix sits at the estimate after it."""
    track = outcome.track
    fixes = {int(row.step): Fix(LatLon(track["est_lat"].iloc[int(row.step)], track["est_lon"].iloc[int(row.step)]),
                                row.decision, row.fix_ms, row.peak_z)
             for row in outcome.fixes.itertuples()}
    return FlightResult(track, fixes, list(poses), name)
