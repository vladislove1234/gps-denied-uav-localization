"""Fit the risk gate on one place and choose its `p_max` by frame-grouped cross-validation, for 022.

Glue imported by `train.ipynb`; moves to `uavloc/` only if a second experiment needs it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from uavloc.matching.benchmark import Draw, error_stats
from uavloc.matching.correlator import CorrelatorConfig
from uavloc.matching.risk import RiskGate, apply_gate, feature_table, harm_labels


def fit_gate_on(draws: list[Draw], config: CorrelatorConfig, taus: tuple[float, ...], ridge_c: float,
                margin_m: float, c_min: float) -> RiskGate:
    return RiskGate(ridge_c).fit(feature_table(draws, config, taus, c_min), harm_labels(draws, config, margin_m))


def out_of_fold_probability(draws: list[Draw], config: CorrelatorConfig, taus: tuple[float, ...], ridge_c: float,
                            margin_m: float, c_min: float, n_folds: int) -> np.ndarray:
    """Harm probability of every draw from a gate that never saw its frame."""
    table = feature_table(draws, config, taus, c_min)
    harm = harm_labels(draws, config, margin_m)
    groups = np.array([d.frame.name for d in draws])
    probability = np.ones(len(draws))
    for fit_idx, test_idx in GroupKFold(n_splits=n_folds).split(table, groups=groups):
        gate = RiskGate(ridge_c).fit(table.iloc[fit_idx].reset_index(drop=True), harm[fit_idx])
        probability[test_idx] = gate.harm_probability(table.iloc[test_idx].reset_index(drop=True))
    return probability


def choose_p_max(draws: list[Draw], config: CorrelatorConfig, taus: tuple[float, ...], ridge_c: float,
                 grid: list[float], n_folds: int, margin_m: float, noise_m: float, max_harm_share: float,
                 min_applied_share: float, c_min: float) -> tuple[float, pd.DataFrame]:
    """Lowest mean error among the `p_max` that meet the harm and applied bars; else the most applied."""
    probability = out_of_fold_probability(draws, config, taus, ridge_c, margin_m, c_min, n_folds)
    at_level = [i for i, d in enumerate(draws) if d.noise_m == noise_m]
    level_draws = [draws[i] for i in at_level]
    rows = []
    for p_max in grid:
        stats = error_stats(apply_gate(level_draws, config, probability[at_level], p_max))
        rows.append({"p_max": p_max, "mean_position_error_m": stats["mean_position_error_m"],
                     "harm_share": stats["harm_share"], "applied_share": stats["accept_share"]})
    table = pd.DataFrame(rows)
    feasible = table[(table["harm_share"] <= max_harm_share) & (table["applied_share"] >= min_applied_share)]
    if len(feasible):
        best = feasible.sort_values(["mean_position_error_m", "p_max"]).iloc[0]
    else:
        best = table.sort_values(["applied_share", "harm_share"], ascending=[False, True]).iloc[0]
    return float(best["p_max"]), table.assign(feasible=table.index.isin(feasible.index))
