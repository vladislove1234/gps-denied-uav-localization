from __future__ import annotations

import json

import pytest

from tools.route_planner.planning_job import PlanOutcome
from tools.route_planner.view_model import (ELEVATION_UNAVAILABLE, FLAG_BLIND_LIMIT, TABLE_HEADERS, route_paths,
                                            route_rows, routes_payload, sub_polyline)
from uavloc.geo import LatLon, haversine_m
from uavloc.planning.routes import BlindStretch, Route

A, B, C = LatLon(49.9, 23.9), LatLon(49.9, 23.91), LatLon(49.91, 23.91)


def make_route(**changes) -> Route:
    values = dict(weight=2.0, waypoints=(A, B, C), length_m=1800.0, blind_share=0.25,
                  blind_stretches=(BlindStretch(600.0, 900.0),), flight_time_s=900.0, blind_limit_m=200.0,
                  flight_time_limit_s=960.0)
    return Route(**{**values, **changes})


def test_sub_polyline_ends_on_the_requested_distances_and_keeps_corners() -> None:
    length_ab = haversine_m(A, B)
    points = sub_polyline((A, B, C), length_ab - 50, length_ab + 80)
    assert B in points
    assert haversine_m(points[0], B) == pytest.approx(50, abs=0.5)
    assert haversine_m(points[-1], B) == pytest.approx(80, abs=0.5)


def test_row_has_one_cell_per_header_and_flags_blind_limit() -> None:
    route = make_route()
    row = route_rows([route], [None])[0]
    assert len(row.cells()) == len(TABLE_HEADERS)
    assert row.flags == (FLAG_BLIND_LIMIT,)
    assert row.cells()[1:6] == ["2", "1.80", "15.0", "25", "300"]
    assert row.elevation_warnings == (ELEVATION_UNAVAILABLE,)


def test_route_within_limits_has_no_flags() -> None:
    row = route_rows([make_route(blind_stretches=(), blind_share=0.0)], [None])[0]
    assert row.flags == ()
    assert row.cells()[-1] == ""


def test_rows_and_checks_must_pair_up() -> None:
    with pytest.raises(ValueError):
        route_rows([make_route()], [])


def test_payload_is_json_and_colours_differ_per_route() -> None:
    payload = routes_payload([make_route(), make_route(weight=5.0)])
    decoded = json.loads(json.dumps(payload))
    colours = [r["colour"] for r in decoded["routes"]]
    assert len(set(colours)) == 2
    assert decoded["routes"][0]["line"][0] == [A.lat, A.lon]
    assert len(decoded["routes"][0]["blind"]) == 1


def test_real_plan_rows_and_paths(outcome: PlanOutcome) -> None:
    routes = outcome.result.routes
    rows = route_rows(routes, outcome.elevation)
    assert len(rows) == len(routes) >= 1
    for route in routes:
        paths = route_paths(route, 0)
        assert paths.line[0] == [route.waypoints[0].lat, route.waypoints[0].lon]
        assert len(paths.blind) == len(route.blind_stretches)
