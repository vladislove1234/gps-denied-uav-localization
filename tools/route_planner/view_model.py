"""What the UI shows for planned routes: table rows, map polylines, colours. No Qt, so it can be tested."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from uavloc.geo import LatLon, haversine_m
from uavloc.onboard.corridor import point_at_distance
from uavloc.planning.elevation import ElevationCheck
from uavloc.planning.routes import Route
from uavloc.plotting.planning import BLIND_COLOUR, ROUTE_COLOURS

SECONDS_PER_MINUTE = 60.0
M_PER_KM = 1000.0
FLAG_BLIND_LIMIT = "blind limit"
FLAG_FLIGHT_TIME = "flight time"
FLAG_SEPARATOR = ", "
TABLE_HEADERS = ("", "Weight", "Length, km", "Flight, min", "Blind, %", "Longest blind, m", "Over limit")
ELEVATION_UNAVAILABLE = "elevation check unavailable"


@dataclass(frozen=True)
class RouteRow:
    index: int
    colour: str
    weight: float
    length_km: float
    flight_min: float
    blind_percent: float
    longest_blind_m: float
    flags: tuple[str, ...]
    elevation_warnings: tuple[str, ...]

    def cells(self) -> list[str]:
        """Table text per column of `TABLE_HEADERS`; the first column is the colour swatch, drawn by the table."""
        return ["", f"{self.weight:g}", f"{self.length_km:.2f}", f"{self.flight_min:.1f}",
                f"{self.blind_percent:.0f}", f"{self.longest_blind_m:.0f}", FLAG_SEPARATOR.join(self.flags)]

    def details(self) -> str:
        """Everything that did not fit a cell: the elevation warnings, or why there are none."""
        return "\n".join(self.elevation_warnings) if self.elevation_warnings else "no elevation warnings"


@dataclass(frozen=True)
class RoutePaths:
    """One route on the map: the whole line and its blind stretches, each as `[lat, lon]` lists."""

    colour: str
    line: list[list[float]]
    blind: list[list[list[float]]]


def route_colour(index: int) -> str:
    return ROUTE_COLOURS[index % len(ROUTE_COLOURS)]


def route_flags(route: Route) -> tuple[str, ...]:
    flags = []
    if route.exceeds_blind_limit:
        flags.append(FLAG_BLIND_LIMIT)
    if route.exceeds_flight_time:
        flags.append(FLAG_FLIGHT_TIME)
    return tuple(flags)


def route_rows(routes: Sequence[Route], checks: Sequence[ElevationCheck | None]) -> list[RouteRow]:
    """One row per route; a route without an elevation check says so as a warning instead of passing silently."""
    if len(routes) != len(checks):
        raise ValueError(f"{len(routes)} routes but {len(checks)} elevation checks")
    return [RouteRow(i, route_colour(i), route.weight, route.length_m / M_PER_KM,
                     route.flight_time_s / SECONDS_PER_MINUTE, route.blind_share * 100, route.longest_blind_m,
                     route_flags(route), check.warnings if check else (ELEVATION_UNAVAILABLE,))
            for i, (route, check) in enumerate(zip(routes, checks))]


def lat_lon_pair(point: LatLon) -> list[float]:
    return [point.lat, point.lon]


def sub_polyline(points: Sequence[LatLon], start_m: float, end_m: float) -> list[LatLon]:
    """The part of the polyline between two distances along it, with the corners in between."""
    out, travelled = [point_at_distance(tuple(points), start_m)], 0.0
    for a, b in zip(points, points[1:]):
        travelled += haversine_m(a, b)
        if start_m < travelled < end_m:
            out.append(b)
    out.append(point_at_distance(tuple(points), end_m))
    return out


def route_paths(route: Route, index: int) -> RoutePaths:
    return RoutePaths(route_colour(index), [lat_lon_pair(p) for p in route.waypoints],
                      [[lat_lon_pair(p) for p in sub_polyline(route.waypoints, s.start_m, s.end_m)]
                       for s in route.blind_stretches])


def legend_text(route: Route) -> str:
    flag = " (!)" if route_flags(route) else ""
    return (f"w={route.weight:g}: {route.length_m / M_PER_KM:.2f} km, blind {route.blind_share:.0%}, "
            f"longest {route.longest_blind_m:.0f} m{flag}")


def routes_payload(routes: Sequence[Route]) -> dict:
    """JSON-ready message for the map page."""
    entries = []
    for index, route in enumerate(routes):
        paths = route_paths(route, index)
        entries.append({"index": index, "colour": paths.colour, "line": paths.line, "blind": paths.blind,
                        "label": legend_text(route)})
    return {"blindColour": BLIND_COLOUR, "routes": entries}
