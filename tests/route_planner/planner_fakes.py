"""Test doubles and the Birky case coordinates."""

from __future__ import annotations

import numpy as np

from uavloc.geo import LatLon

START = LatLon(49.905, 23.885)
END = LatLon(49.915, 23.925)


class FlatGround:
    """Elevation source with no network: the same height everywhere."""

    def elevations_m(self, points) -> np.ndarray:
        return np.full(len(points), 250.0)


class BrokenNetwork:
    def elevations_m(self, points) -> np.ndarray:
        raise OSError("no network")
