"""Particle filter whose particles also carry the INS error (028).

027's filter treats INS drift as white noise on position, but the drift is a constant
velocity bias, a heading bias and a scale error held for the whole flight. Here each
particle carries its own guess of those three, so a particle undoes the INS increment
the way the INS got it wrong:

    true move = rotate(-heading_bias) · (INS move - velocity_bias · dt) / scale

The heatmap update is 027's (`exp(z / T)` on one window covering the cloud); the heatmap
now also tells particles apart by their bias, through where that bias moved them.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from uavloc.filtering.belief import WindowConfig, observe
from uavloc.geo import LatLon, offset_latlon
from uavloc.matching.correlator import PositionCorrelator
from uavloc.matching.window import peak_at_edge, to_heading_up, z_at
from uavloc.simulation.correctors import MS_PER_S, RenderedFrame
from uavloc.simulation.estimators import FixRecord


@dataclass(frozen=True)
class InsErrorPrior:
    """What the IMU spec says about the error at GNSS loss, and how fast the bias wanders."""

    velocity_bias_mps: float
    velocity_random_walk_mps: float   # per sqrt(s)
    heading_bias_deg: float
    speed_scale_std: float


@dataclass(frozen=True)
class InsErrorFilterConfig:
    prior: InsErrorPrior
    n_particles: int = 2000
    temperature: float = 1.0
    q_pos_m_per_sqrt_s: float = 0.3           # what the error model still misses
    heading_jitter_deg_per_sqrt_s: float = 0.01   # keeps resampled heading biases from collapsing to copies
    resample_ess_share: float = 0.5
    window: WindowConfig = WindowConfig()


class InsErrorParticleFilter:
    name = "ins-error-particles"

    def __init__(self, correlator: PositionCorrelator, config: InsErrorFilterConfig, seed: int) -> None:
        self.correlator = correlator
        self.config = config
        self.rng = np.random.default_rng(seed)

    def start(self, position: LatLon, sigma0_m: float) -> None:
        n, prior = self.config.n_particles, self.config.prior
        self.origin = position
        self.particles = self.rng.normal(0.0, sigma0_m, (n, 2))
        self.velocity_bias = self.rng.normal(0.0, prior.velocity_bias_mps, (n, 2))
        self.heading_bias = np.radians(self.rng.normal(0.0, prior.heading_bias_deg, n))
        self.scale = 1.0 + self.rng.normal(0.0, prior.speed_scale_std, n)
        self.log_weights = np.full(n, -math.log(n))

    def predict(self, move_en_m: tuple[float, float], dt_s: float) -> None:
        cfg, n = self.config, self.config.n_particles
        east = move_en_m[0] - self.velocity_bias[:, 0] * dt_s
        north = move_en_m[1] - self.velocity_bias[:, 1] * dt_s
        cos, sin = np.cos(-self.heading_bias), np.sin(-self.heading_bias)   # undo a clockwise turn
        true_move = np.stack([east * cos + north * sin, -east * sin + north * cos], axis=1) / self.scale[:, None]
        self.particles += true_move + self.rng.normal(0.0, cfg.q_pos_m_per_sqrt_s * math.sqrt(dt_s), (n, 2))
        self.velocity_bias += self.rng.normal(0.0, cfg.prior.velocity_random_walk_mps * math.sqrt(dt_s), (n, 2))
        self.heading_bias += np.radians(self.rng.normal(0.0, cfg.heading_jitter_deg_per_sqrt_s * math.sqrt(dt_s), n))

    def update(self, frame: RenderedFrame) -> FixRecord:
        start = time.perf_counter()
        surface, window, decision = observe(self.correlator, frame, self.origin, self.particles, self._mean(),
                                            self.config.window)
        if decision != "update":
            return FixRecord(decision, (time.perf_counter() - start) * MS_PER_S, window,
                             surface.extent_m() if surface else None)
        z = z_at(surface, to_heading_up(self.origin, window.center, self.particles, window.yaw_deg))
        self.log_weights = self.log_weights + z / self.config.temperature
        self.log_weights -= logsumexp(self.log_weights)
        effective_size = float(1.0 / np.exp(2 * self.log_weights).sum())
        if effective_size < self.config.resample_ess_share * self.config.n_particles:
            self._resample()
        return FixRecord("update", (time.perf_counter() - start) * MS_PER_S, window, surface.extent_m(),
                         float(np.nanmax(surface.heatmap)), surface.coverage, peak_at_edge(surface),
                         effective_size=effective_size)

    def _resample(self) -> None:
        """Systematic resampling; every state component follows its particle."""
        n = self.config.n_particles
        positions = (self.rng.random() + np.arange(n)) / n
        cumulative = np.cumsum(np.exp(self.log_weights))
        cumulative[-1] = 1.0
        keep = np.searchsorted(cumulative, positions)
        self.particles, self.velocity_bias = self.particles[keep], self.velocity_bias[keep]
        self.heading_bias, self.scale = self.heading_bias[keep], self.scale[keep]
        self.log_weights = np.full(n, -math.log(n))

    def _mean(self) -> np.ndarray:
        return np.exp(self.log_weights) @ self.particles

    def estimate(self) -> tuple[LatLon, np.ndarray]:
        weights = np.exp(self.log_weights)
        mean = weights @ self.particles
        centred = self.particles - mean
        return offset_latlon(self.origin, *mean), (weights[:, None] * centred).T @ centred

    def bias_estimate(self) -> dict[str, float]:
        """Weighted mean INS error the cloud believes in."""
        weights = np.exp(self.log_weights)
        return {"velocity_bias_east_mps": float(weights @ self.velocity_bias[:, 0]),
                "velocity_bias_north_mps": float(weights @ self.velocity_bias[:, 1]),
                "heading_bias_deg": float(np.degrees(weights @ self.heading_bias)),
                "scale": float(weights @ self.scale)}
