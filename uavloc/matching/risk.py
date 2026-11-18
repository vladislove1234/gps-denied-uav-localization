"""A gate that reads the heatmap's shape and predicts the probability that applying the move harms.

Features come from the `SearchSurface` and the config only, never from the truth, the place or the
injected error. `score_risk` then applies the argmax move where the predicted harm is at most `p_max`
and keeps the prior elsewhere.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from uavloc.matching.benchmark import HARM_MARGIN_M, Draw, score_draws
from uavloc.matching.correlator import CorrelatorConfig, SearchSurface, posterior, second_peak_ratio

LOG_EPS = 1e-6
NO_RIVAL_RATIO = 1.0            # worst case, used when the surface has no rival peak to compare with
LOGISTIC_MAX_ITER = 1000


def gate_features(surface: SearchSurface, config: CorrelatorConfig, taus: tuple[float, ...]) -> dict[str, float]:
    """Shape of one heatmap: posterior spread at each temperature, peak height, rival peak and coverage.

    The posterior uses a flat prior whatever `config` says, so the features describe the score alone.
    """
    features = {}
    for tau in taus:
        s1, s2, _, _ = posterior(surface, replace(config, temperature=tau, prior_sigma_m=None)).axes
        features[f"log_s1_t{tau:g}"] = float(np.log(s1 + LOG_EPS))
        features[f"log_s2_t{tau:g}"] = float(np.log(s2 + LOG_EPS))
    cell = posterior(surface, replace(config, prior_sigma_m=None)).map_cell
    ratio = second_peak_ratio(surface, cell)
    features["peak_z"] = float(surface.heatmap[cell])
    features["second_peak_ratio"] = NO_RIVAL_RATIO if np.isnan(ratio) else ratio
    features["coverage"] = surface.coverage
    return features


def is_eligible(surface: SearchSurface | None, c_min: float) -> bool:
    """A draw the gate may accept: inside the raster, a non-flat heatmap and enough weighted classes."""
    if surface is None or np.nanmax(surface.heatmap) == np.nanmin(surface.heatmap):
        return False
    return surface.coverage >= c_min


def feature_table(draws: list[Draw], config: CorrelatorConfig, taus: tuple[float, ...], c_min: float) -> pd.DataFrame:
    """One row per draw, in order; `eligible` is False and the features NaN where the gate cannot accept."""
    rows = []
    for draw in draws:
        eligible = is_eligible(draw.surface, c_min)
        rows.append({"eligible": eligible, **(gate_features(draw.surface, config, taus) if eligible else {})})
    return pd.DataFrame(rows).astype({"eligible": bool})


def harm_labels(draws: list[Draw], config: CorrelatorConfig, margin_m: float = HARM_MARGIN_M) -> np.ndarray:
    """True where the plain argmax move leaves the position worse than the prior by more than `margin_m`."""
    ungated = score_draws(draws, replace(config, gate="none"))
    return (ungated["error_m"] > ungated["prior_error_m"] + margin_m).to_numpy()


class RiskGate:
    """Logistic regression from gate features to the probability that the argmax move harms."""

    def __init__(self, ridge_c: float) -> None:
        self.model: Pipeline = make_pipeline(StandardScaler(), LogisticRegression(C=ridge_c, max_iter=LOGISTIC_MAX_ITER))
        self.columns: list[str] = []

    def fit(self, table: pd.DataFrame, harm: np.ndarray) -> RiskGate:
        eligible = table["eligible"].to_numpy()
        self.columns = [c for c in table.columns if c != "eligible"]
        self.model.fit(table.loc[eligible, self.columns].to_numpy(), harm[eligible])
        return self

    def harm_probability(self, table: pd.DataFrame) -> np.ndarray:
        """Predicted harm per row; 1.0 on rows the gate cannot accept."""
        probability = np.ones(len(table))
        eligible = table["eligible"].to_numpy()
        if eligible.any():
            probability[eligible] = self.model.predict_proba(table.loc[eligible, self.columns].to_numpy())[:, 1]
        return probability

    def weights(self) -> dict[str, float]:
        """Coefficients on standardised features, for the record."""
        return dict(zip(self.columns, self.model[-1].coef_[0].round(4).tolist()))


def apply_gate(draws: list[Draw], config: CorrelatorConfig, harm_probability: np.ndarray, p_max: float) -> pd.DataFrame:
    """`score_draws`' rows with the argmax move kept where `harm_probability <= p_max`, the prior elsewhere."""
    scored = score_draws(draws, replace(config, gate="none"))
    rejected = harm_probability > p_max
    scored.loc[rejected, "corrected_lat"] = scored.loc[rejected, "prior_lat"]
    scored.loc[rejected, "corrected_lon"] = scored.loc[rejected, "prior_lon"]
    scored.loc[rejected, "error_m"] = scored.loc[rejected, "prior_error_m"]
    scored.loc[rejected, "applied_offset_m"] = 0.0
    scored["decision"] = np.where(rejected, "reject", "accept")
    scored["harm_probability"] = harm_probability
    return scored


def score_risk(draws: list[Draw], config: CorrelatorConfig, gate: RiskGate, p_max: float,
               taus: tuple[float, ...], c_min: float) -> pd.DataFrame:
    return apply_gate(draws, config, gate.harm_probability(feature_table(draws, config, taus, c_min)), p_max)
