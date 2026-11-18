"""Leave-one-place-out tuning of the decision values (temperature, gate), from 007.

Each held-out place gets values chosen on the other place's draws only. Moved here from
`experiments/007-search-prior-and-gate/lopo.py` when 022 needed it a second time.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product

import pandas as pd

from uavloc.matching.benchmark import Draw, error_stats, score_draws
from uavloc.matching.correlator import CorrelatorConfig


@dataclass(frozen=True)
class TuningGrid:
    temperatures: tuple[float, ...]
    s_max_m: tuple[float, ...]
    c_min: tuple[float, ...]
    noise_m: float              # tuning level
    min_applied_share: float    # accepted + across-only, gate objective constraint
    temperature_by: str = "mean-error"   # "mean-error" (pre-registered) | "gate": chosen jointly with the gate


def stats_at(draws: list[Draw], config: CorrelatorConfig, noise_m: float) -> dict[str, float]:
    return error_stats(score_draws([d for d in draws if d.noise_m == noise_m], config))


def gate_key(stats: dict[str, float], min_applied_share: float) -> tuple:
    """Feasible first, then lowest harm, then lowest mean error; infeasible ones by highest applied share."""
    applied = stats["accept_share"] + stats["across_share"]
    if applied >= min_applied_share:
        return (0, stats["harm_share"], stats["mean_position_error_m"])
    return (1, -applied, stats["harm_share"], stats["mean_position_error_m"])


def tuned_config(draws: list[Draw], base: CorrelatorConfig, gate: str,
                 grid: TuningGrid) -> tuple[CorrelatorConfig, pd.DataFrame]:
    """The fold's config (`base`, ungated, plus `gate` with tuned values) and every candidate's tuning-place stats.

    Temperature is chosen on mean error without the gate, so a gated run shares its prior-only twin's
    temperature. With a flat prior temperature cannot move the argmax, so it is chosen with the gate instead;
    `temperature_by="gate"` does that for every gated run (not pre-registered).
    """
    if base.prior_sigma_m is None and gate == "none":
        return base, pd.DataFrame()
    temperatures = grid.temperatures
    rows = []
    if base.prior_sigma_m is not None and (gate == "none" or grid.temperature_by == "mean-error"):
        for t in temperatures:
            rows.append({"stage": "temperature", "temperature": t,
                         **stats_at(draws, replace(base, temperature=t), grid.noise_m)})
        best = min(rows, key=lambda r: (r["mean_position_error_m"], r["temperature"]))
        temperatures = (best["temperature"],)
        base = replace(base, temperature=best["temperature"])
    if gate != "none":
        gate_rows = []
        for t, s_max, c_min in product(temperatures, grid.s_max_m, grid.c_min):
            candidate = replace(base, temperature=t, gate=gate, gate_s_max_m=s_max, gate_c_min=c_min)
            gate_rows.append({"stage": "gate", "temperature": t, "gate_s_max_m": s_max, "gate_c_min": c_min,
                              **stats_at(draws, candidate, grid.noise_m)})
        best = min(gate_rows, key=lambda r: gate_key(r, grid.min_applied_share))
        base = replace(base, temperature=best["temperature"], gate=gate, gate_s_max_m=best["gate_s_max_m"],
                       gate_c_min=best["gate_c_min"])
        rows += gate_rows
    return base, pd.DataFrame(rows)
