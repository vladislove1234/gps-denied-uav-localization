"""A: dense particle filter over the correlator heatmap (027).

Particles are east/north offsets from the start. Each fix multiplies a particle's
weight by exp(z / T), with z read from one heatmap that covers the whole cloud.
With an empty-view model (032), a blind frame multiplies it by P(blind | map coverage)^β
instead of being skipped.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from uavloc.filtering.belief import MIN_WEIGHTED_COVERAGE, WindowConfig, observe
from uavloc.filtering.empty_view import EmptyViewModel, coverage_surface
from uavloc.geo import LatLon, offset_latlon
from uavloc.matching.correlator import PositionCorrelator
from uavloc.matching.window import SearchWindow, cells_of, peak_at_edge, to_heading_up, z_at
from uavloc.simulation.correctors import MS_PER_S, RenderedFrame
from uavloc.simulation.estimators import FixRecord


@dataclass(frozen=True)
class ParticleFilterConfig:
    n_particles: int = 2000
    temperature: float = 1.0
    q_pos_m_per_sqrt_s: float = 0.3
    resample_ess_share: float = 0.5
    window: WindowConfig = WindowConfig()
    empty_view: EmptyViewModel | None = None   # None or weight 0: blind frames are skipped, as in 027
    empty_view_weight: float = 0.0             # β, the exponent on P(blind | map coverage)


class ParticleFilter:
    name = "particles"

    def __init__(self, correlator: PositionCorrelator, config: ParticleFilterConfig, seed: int) -> None:
        self.correlator = correlator
        self.config = config
        self.rng = np.random.default_rng(seed)

    def start(self, position: LatLon, sigma0_m: float) -> None:
        self.origin = position
        self.particles = self.rng.normal(0.0, sigma0_m, (self.config.n_particles, 2))
        self.log_weights = np.full(self.config.n_particles, -math.log(self.config.n_particles))

    def predict(self, move_en_m: tuple[float, float], dt_s: float) -> None:
        noise_m = self.config.q_pos_m_per_sqrt_s * math.sqrt(dt_s)
        self.particles += np.asarray(move_en_m) + self.rng.normal(0.0, noise_m, self.particles.shape)

    def update(self, frame: RenderedFrame) -> FixRecord:
        start = time.perf_counter()
        surface, window, decision = observe(self.correlator, frame, self.origin, self.particles, self._mean(),
                                            self.config.window)
        if decision == "blind" and self._uses_empty_view() and surface.coverage < MIN_WEIGHTED_COVERAGE:
            return self._empty_view_update(frame, window, start)
        if decision != "update":
            return FixRecord(decision, (time.perf_counter() - start) * MS_PER_S, window,
                             surface.extent_m() if surface else None)
        z = z_at(surface, to_heading_up(self.origin, window.center, self.particles, window.yaw_deg))
        effective_size = self._reweight(z / self.config.temperature)
        return FixRecord("update", (time.perf_counter() - start) * MS_PER_S, window, surface.extent_m(),
                         float(np.nanmax(surface.heatmap)), surface.coverage, peak_at_edge(surface),
                         effective_size=effective_size)

    def _uses_empty_view(self) -> bool:
        return self.config.empty_view is not None and self.config.empty_view_weight > 0

    def _empty_view_update(self, frame: RenderedFrame, window: SearchWindow, start: float) -> FixRecord:
        try:
            grid = coverage_surface(self.correlator, frame, window)
        except ValueError:   # the patch leaves the map raster
            return FixRecord("out-of-raster", (time.perf_counter() - start) * MS_PER_S, window)
        log_prob = self.config.empty_view.log_prob_blind(grid.heatmap)
        rows, cols, inside = cells_of(grid, to_heading_up(self.origin, window.center, self.particles, window.yaw_deg))
        log_likelihood = np.full(len(self.particles), log_prob.mean())   # outside the window: no map read, neutral
        log_likelihood[inside] = log_prob[rows[inside], cols[inside]]
        effective_size = self._reweight(self.config.empty_view_weight * log_likelihood)
        return FixRecord("empty-view", (time.perf_counter() - start) * MS_PER_S, window, grid.extent_m(),
                         coverage=grid.coverage, effective_size=effective_size)

    def _reweight(self, log_likelihood: np.ndarray) -> float:
        """Fold one update into the weights, resample at low ESS, and return the ESS before resampling."""
        self.log_weights = self.log_weights + log_likelihood
        self.log_weights -= logsumexp(self.log_weights)
        effective_size = float(1.0 / np.exp(2 * self.log_weights).sum())
        if effective_size < self.config.resample_ess_share * self.config.n_particles:
            self._resample()
        return effective_size

    def _resample(self) -> None:
        """Systematic resampling."""
        n = self.config.n_particles
        positions = (self.rng.random() + np.arange(n)) / n
        cumulative = np.cumsum(np.exp(self.log_weights))
        cumulative[-1] = 1.0
        self.particles = self.particles[np.searchsorted(cumulative, positions)]
        self.log_weights = np.full(n, -math.log(n))

    def _mean(self) -> np.ndarray:
        return np.exp(self.log_weights) @ self.particles

    def estimate(self) -> tuple[LatLon, np.ndarray]:
        weights = np.exp(self.log_weights)
        mean = weights @ self.particles
        centred = self.particles - mean
        return offset_latlon(self.origin, *mean), (weights[:, None] * centred).T @ centred
