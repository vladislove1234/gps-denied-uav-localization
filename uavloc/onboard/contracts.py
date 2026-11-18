"""Messages and interfaces of the onboard loop. The real localizer will plug into these later."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from uavloc.data.satellite import GeoImage
from uavloc.geo import LatLon

CELL_CENTRE = 0.5
BLIND_THRESHOLD = 0.5   # same cut as `uavloc.planning.routes.BLIND_THRESHOLD`


def bilinear_at(grid: np.ndarray, x: float, y: float) -> float:
    """Value of a `(rows, cols)` grid at fractional `(x, y)` index, edge values held beyond the grid."""
    rows, cols = grid.shape
    x, y = min(max(x, 0.0), cols - 1.0), min(max(y, 0.0), rows - 1.0)
    x0, y0 = int(x), int(y)
    x1, y1 = min(x0 + 1, cols - 1), min(y0 + 1, rows - 1)
    fx, fy = x - x0, y - y0
    top = grid[y0, x0] * (1 - fx) + grid[y0, x1] * fx
    bottom = grid[y1, x0] * (1 - fx) + grid[y1, x1] * fx
    return float(top * (1 - fy) + bottom * fy)


@dataclass(frozen=True)
class Frame:
    time_s: float             # capture time, flight-controller clock
    image: np.ndarray         # `(H, W, 3)` uint8 RGB


@dataclass(frozen=True)
class FlightState:
    time_s: float                       # flight-controller clock (seconds since boot)
    position: LatLon | None             # ArduPilot's own estimate; None while it has none
    altitude_m: float                   # relative to home; the runner swaps in the `HeightSource` value
    yaw_deg: float
    roll_deg: float
    pitch_deg: float
    velocity_north_mps: float
    velocity_east_mps: float


@dataclass(frozen=True)
class Fix:
    time_s: float                       # capture time of the frame the fix describes, flight-controller clock
    position: LatLon | None             # None = no fix; `reason` says why
    uncertainty_m: float                # 1-sigma horizontal
    quality: float                      # 0..1
    reason: str


@dataclass(frozen=True)
class ReferenceMap:
    """What the package carries to localize against."""

    classes: GeoImage                   # `(H, W, 6)` uint8 probabilities x 255, zero outside the corridor
    blind: GeoImage                     # `(rows, cols, 1)` float32, 1 = a fix is unlikely, 1 outside the corridor
    blind_threshold: float = BLIND_THRESHOLD

    def contains(self, point: LatLon) -> bool:
        x, y = self.blind.to_px(point)
        rows, cols = self.blind.pixels.shape[:2]
        return 0 <= x < cols and 0 <= y < rows

    def is_blind(self, point: LatLon) -> bool:
        """True where a fix is unlikely, and everywhere outside the reference.

        The blind share is interpolated between cell centres, as the planner's cost grid does, so a lone blind
        cell that the route's blind stretches ignore does not drop fixes here either.
        """
        if not self.contains(point):
            return True
        x, y = self.blind.to_px(point)
        return bilinear_at(self.blind.pixels[..., 0], x - CELL_CENTRE, y - CELL_CENTRE) >= self.blind_threshold


class CameraSource(Protocol):
    def next_frame(self) -> Frame | None: ...


class FlightStateSource(Protocol):
    def state_at(self, time_s: float) -> FlightState | None: ...

    def latest(self) -> FlightState | None: ...


class HeightSource(Protocol):
    def height_m(self, state: FlightState) -> float:
        """Height above the ground the camera sees. v1 takes it from ArduPilot; a rangefinder can replace it."""
        ...


class Localizer(Protocol):
    def locate(self, frame: Frame | None, state: FlightState, reference: ReferenceMap) -> Fix: ...


class PositionSink(Protocol):
    def send(self, fix: Fix) -> None: ...


class MotionSource(Protocol):
    """Short-term motion and the clocks, for carrying a fix forward from capture to send time."""

    def fc_time_s(self) -> float:
        """Now, on the flight-controller clock."""
        ...

    def unix_time_s(self) -> float:
        """Now, as Unix time according to the flight controller."""
        ...

    def velocity_displacement_m(self, from_s: float, to_s: float) -> tuple[float, float]:
        """`(east_m, north_m)` moved between two flight-controller times; raises LookupError when not covered."""
        ...


class AltitudeSource(Protocol):
    """Height above mean sea level, for the `GPS_INPUT` altitude: ArduPilot compares GPS 1 and GPS 2 in 3D."""

    def altitude_amsl_m(self) -> float:
        """Latest altitude above mean sea level; raises LookupError before the first sample."""
        ...


class TruthSource(Protocol):
    """Simulator ground truth, used only by the stub localizer."""

    def position_at(self, time_s: float) -> LatLon | None: ...


class FlightLogger(Protocol):
    def log_fix(self, fix: Fix) -> None: ...

    def log_state(self, state: FlightState) -> None: ...

    def log_frame(self, frame: Frame) -> None: ...

    def close(self) -> None: ...

