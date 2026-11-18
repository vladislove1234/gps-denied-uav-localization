"""B: multi-hypothesis tracker over the correlator heatmap (027).

A handful of Gaussian hypotheses, no random particles. Each fix multiplies a
hypothesis's Gaussian by exp(z / T) over one shared heatmap; the top local maxima
of that product, at least `peak_nms_m` apart, become its children, weighted by the
product's mass near them. Children that land together merge; the heaviest
`k_max` survive.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from uavloc.filtering.belief import WindowConfig, east_north_to_heading_up, observe
from uavloc.geo import LatLon, east_north_m, offset_latlon
from uavloc.matching.correlator import PositionCorrelator, SearchSurface
from uavloc.matching.window import SearchWindow, peak_at_edge, to_heading_up
from uavloc.simulation.correctors import MS_PER_S, RenderedFrame
from uavloc.simulation.estimators import FixRecord

ELLIPSE_POINTS = 32   # boundary points per hypothesis when sizing the window


@dataclass(frozen=True)
class HypothesisTrackerConfig:
    k_max: int = 5
    peaks_per_hypothesis: int = 3
    peak_nms_m: float = 10.0          # children of one parent are at least this far apart; mass is summed within it
    merge_distance_m: float = 5.0
    child_floor_sigma_m: float = 1.0  # added to every child's covariance
    window_sigmas: float = 3.0        # the window covers each hypothesis's ellipse at this many sigma
    temperature: float = 1.0
    q_pos_m_per_sqrt_s: float = 0.3
    window: WindowConfig = WindowConfig()


def gaussian_log_density(points: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """`(M,)` log N(points; mean, cov) for `(M, 2)` points."""
    centred = points - mean
    inverse = np.linalg.inv(cov)
    mahalanobis2 = np.einsum("mi,ij,mj->m", centred, inverse, centred)
    return -0.5 * mahalanobis2 - math.log(2 * math.pi) - 0.5 * math.log(np.linalg.det(cov))


def weighted_moments(points: np.ndarray, log_p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weights = np.exp(log_p - logsumexp(log_p))
    mean = weights @ points
    centred = points - mean
    return mean, (weights[:, None] * centred).T @ centred


class HypothesisTracker:
    name = "hypotheses"

    def __init__(self, correlator: PositionCorrelator, config: HypothesisTrackerConfig) -> None:
        self.correlator = correlator
        self.config = config

    def start(self, position: LatLon, sigma0_m: float) -> None:
        self.origin = position
        self.means = np.zeros((1, 2))
        self.covs = np.array([np.eye(2) * sigma0_m ** 2])
        self.log_weights = np.zeros(1)

    def predict(self, move_en_m: tuple[float, float], dt_s: float) -> None:
        self.means = self.means + np.asarray(move_en_m)
        self.covs = self.covs + np.eye(2) * self.config.q_pos_m_per_sqrt_s ** 2 * dt_s

    def update(self, frame: RenderedFrame) -> FixRecord:
        start = time.perf_counter()
        surface, window, decision = observe(self.correlator, frame, self.origin, self._ellipse_points(),
                                            self.means[np.argmax(self.log_weights)], self.config.window)
        if decision == "update":
            self._update(surface, window)
        return FixRecord(decision, (time.perf_counter() - start) * MS_PER_S, window,
                         surface.extent_m() if surface else None,
                         float(np.nanmax(surface.heatmap)) if decision == "update" else math.nan,
                         surface.coverage if surface else math.nan,
                         peak_at_edge(surface) if decision == "update" else False,
                         live_hypotheses=len(self.means), dominance=self.dominance())

    def _ellipse_points(self) -> np.ndarray:
        angles = np.linspace(0.0, 2 * math.pi, ELLIPSE_POINTS, endpoint=False)
        circle = np.stack([np.cos(angles), np.sin(angles)], axis=1) * self.config.window_sigmas
        return np.concatenate([mean + circle @ np.linalg.cholesky(cov).T for mean, cov in zip(self.means, self.covs)])

    def _update(self, surface: SearchSurface, window: SearchWindow) -> None:
        cfg = self.config
        cells_rd = surface.offset_grid_m().reshape(2, -1).T
        log_likelihood = np.nan_to_num(surface.heatmap.reshape(-1).astype(np.float64), nan=0.0) / cfg.temperature
        rotation = east_north_to_heading_up(window.yaw_deg)
        center_en = np.asarray(east_north_m(self.origin, window.center))
        means_rd = to_heading_up(self.origin, window.center, self.means, window.yaw_deg)
        children = []
        for mean_rd, cov, log_weight in zip(means_rd, self.covs, self.log_weights):
            log_p = log_likelihood + gaussian_log_density(cells_rd, mean_rd, rotation @ cov @ rotation.T)
            for mean_child_rd, cov_child_rd, log_mass in self._peaks(cells_rd, log_p):
                mean_en = center_en + rotation @ mean_child_rd
                cov_en = rotation @ cov_child_rd @ rotation.T + np.eye(2) * cfg.child_floor_sigma_m ** 2
                children.append((log_weight + log_mass, mean_en, cov_en))
        self._keep(self._merged(children))

    def _peaks(self, cells_rd: np.ndarray, log_p: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, float]]:
        """Up to `peaks_per_hypothesis` local maxima, each with the moments and log mass of the disc around it."""
        remaining = np.ones(len(log_p), bool)
        peaks = []
        for _ in range(self.config.peaks_per_hypothesis):
            if not remaining.any():
                break
            best = np.flatnonzero(remaining)[np.argmax(log_p[remaining])]
            disc = remaining & (np.hypot(*(cells_rd - cells_rd[best]).T) <= self.config.peak_nms_m)
            mean, cov = weighted_moments(cells_rd[disc], log_p[disc])
            peaks.append((mean, cov, float(logsumexp(log_p[disc]))))
            remaining &= ~disc
        return peaks

    def _merged(self, children: list[tuple[float, np.ndarray, np.ndarray]]) -> list[tuple[float, np.ndarray, np.ndarray]]:
        """Greedy, heaviest first: a child within `merge_distance_m` of a kept one is moment-matched into it."""
        groups: list[list[tuple[float, np.ndarray, np.ndarray]]] = []
        for child in sorted(children, key=lambda c: -c[0]):
            for group in groups:
                if np.hypot(*(child[1] - group[0][1])) <= self.config.merge_distance_m:
                    group.append(child)
                    break
            else:
                groups.append([child])
        merged = []
        for group in groups:
            log_weights = np.array([c[0] for c in group])
            weights = np.exp(log_weights - logsumexp(log_weights))
            mean = sum(w * c[1] for w, c in zip(weights, group))
            cov = sum(w * (c[2] + np.outer(c[1] - mean, c[1] - mean)) for w, c in zip(weights, group))
            merged.append((float(logsumexp(log_weights)), mean, cov))
        return merged

    def _keep(self, hypotheses: list[tuple[float, np.ndarray, np.ndarray]]) -> None:
        kept = sorted(hypotheses, key=lambda h: -h[0])[:self.config.k_max]
        log_weights = np.array([h[0] for h in kept])
        self.log_weights = log_weights - logsumexp(log_weights)
        self.means = np.array([h[1] for h in kept])
        self.covs = np.array([h[2] for h in kept])

    def dominance(self) -> float:
        return float(np.exp(self.log_weights.max()))

    def estimate(self) -> tuple[LatLon, np.ndarray]:
        best = int(np.argmax(self.log_weights))
        return offset_latlon(self.origin, *self.means[best]), self.covs[best]
