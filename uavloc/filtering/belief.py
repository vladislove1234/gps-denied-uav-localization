"""What both 027 filters share: score one window that covers the belief, or say why there is nothing to read."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uavloc.features.classes import CLASSES
from uavloc.geo import LatLon
from uavloc.matching.correlator import PositionCorrelator, SearchSurface, drone_class_shares
from uavloc.matching.window import SearchWindow, is_flat, score_window, window_around
from uavloc.simulation.correctors import RenderedFrame

MIN_WEIGHTED_COVERAGE = 0.01   # below this share of weighted-class drone pixels the frame is blind (as in 007)
ANCHOR_CLASSES = tuple(CLASSES.index(name) for name in ("building", "track", "water"))
BLIND_RULES = ("one-class", "no-anchors")

# heading-up (right, down) from north-up (east, north): right = cos·e − sin·n, down = −sin·e − cos·n.
# The matrix is symmetric and orthogonal, so it is its own inverse.


def east_north_to_heading_up(yaw_deg: float) -> np.ndarray:
    yaw = np.radians(yaw_deg)
    return np.array([[np.cos(yaw), -np.sin(yaw)], [-np.sin(yaw), -np.cos(yaw)]])


@dataclass(frozen=True)
class BlindRule:
    """A frame with no structure is blind (034). `one-class`: the commonest class covers >= `threshold` of the frame.
    `no-anchors`: building + track + water cover < `threshold`."""

    kind: str
    threshold: float

    def __post_init__(self) -> None:
        if self.kind not in BLIND_RULES:
            raise ValueError(f"kind must be one of {BLIND_RULES}, got {self.kind!r}")

    def is_blind(self, class_shares: np.ndarray) -> bool:
        if self.kind == "one-class":
            return bool(class_shares.max() >= self.threshold)
        return bool(class_shares[list(ANCHOR_CLASSES)].sum() < self.threshold)


@dataclass(frozen=True)
class WindowConfig:
    cap_half_m: float = 100.0   # the window is at most 200 m a side
    margin_m: float = 2.0       # beyond the outermost belief point
    blind_rule: BlindRule | None = None   # None: blind only by coverage and flatness, as in 027


def observe(correlator: PositionCorrelator, frame: RenderedFrame, origin: LatLon, points_en_m: np.ndarray,
            anchor_en_m: np.ndarray, config: WindowConfig) -> tuple[SearchSurface | None, SearchWindow, str]:
    """`(surface, window, decision)`: decision is "update", "blind" or "out-of-raster" (surface None)."""
    footprint = correlator.footprint_m(frame)
    window = window_around(origin, points_en_m, frame.yaw_deg, correlator.search_half_range_m(frame),
                           config.cap_half_m, config.margin_m, anchor_en_m)
    try:
        drone = correlator.drone_map(frame)
        surface = score_window(correlator, drone, footprint, window)
    except ValueError:   # the patch leaves the map raster
        return None, window, "out-of-raster"
    finally:
        correlator.clear_cache()
    if surface.coverage < MIN_WEIGHTED_COVERAGE or is_flat(surface):
        return surface, window, "blind"
    if config.blind_rule is not None and config.blind_rule.is_blind(drone_class_shares(drone)):
        return surface, window, "blind"
    return surface, window, "update"
