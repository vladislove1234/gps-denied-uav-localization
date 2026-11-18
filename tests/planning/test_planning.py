from __future__ import annotations

import numpy as np
import pytest

from uavloc.data.satellite import GeoImage
from uavloc.data.tiles import TILE_PX, TileRange, global_px, latlon_of_global_px, mercator_to_geoimage
from uavloc.features.tiling import tiled_logits, tiled_predict
from uavloc.geo import METERS_PER_DEG, GeoBounds, LatLon, haversine_m, meters_per_deg_lon
from uavloc.planning.config import MissionInputs, PlannerConfig
from uavloc.planning.drift import drift_sigma_m, max_blind_distance_m
from uavloc.planning.elevation import check_elevation
from uavloc.planning.grid import CostGrid, astar
from uavloc.planning.routes import blind_stretches, plan_routes, straighten
from uavloc.simulation.ins import InsConfig

ORIGIN = LatLon(49.9, 23.9)
CELL_M = 10.0


def make_grid(blind: np.ndarray, blocked: np.ndarray | None = None) -> CostGrid:
    rows, cols = blind.shape
    bottom_right = LatLon(ORIGIN.lat - rows * CELL_M / METERS_PER_DEG,
                          ORIGIN.lon + cols * CELL_M / meters_per_deg_lon(ORIGIN.lat))
    raster = GeoImage(blind.astype(np.float32)[..., None], ORIGIN, bottom_right)
    return CostGrid(raster, np.zeros(blind.shape, bool) if blocked is None else blocked)


def wall_with_gap() -> np.ndarray:
    """A blind band across the middle column range, open only at the top rows."""
    blind = np.zeros((40, 60))
    blind[5:, 28:32] = 1.0
    return blind


def test_global_px_round_trip() -> None:
    x, y = global_px(ORIGIN, 18)
    back = latlon_of_global_px(x, y, 18)
    assert haversine_m(ORIGIN, back) < 1e-6


def test_mercator_rows_become_linear_in_latitude() -> None:
    tiles = TileRange.covering(GeoBounds.around([ORIGIN], 3000), 18)
    h = tiles.shape_px[0]
    rows = np.repeat(np.arange(h, dtype=np.float32)[:, None, None], 2, axis=1)   # value = mercator row index
    image = mercator_to_geoimage(rows, tiles)
    r = h // 3
    lat = image.to_latlon(0.5, r + 0.5).lat
    expected_row = global_px(LatLon(lat, ORIGIN.lon), 18)[1] - tiles.y0 * TILE_PX - 0.5
    assert image.pixels[r, 0, 0] == pytest.approx(expected_row, abs=1e-3)


def test_bounds_around_keeps_margin() -> None:
    a, b = ORIGIN, LatLon(49.91, 23.92)
    bounds = GeoBounds.around([a, b], 500)
    assert bounds.contains(a) and bounds.contains(b)
    assert haversine_m(LatLon(bounds.north, a.lon), LatLon(b.lat, a.lon)) == pytest.approx(500, rel=1e-3)


def test_tiled_predict_matches_torch_wrapper() -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Conv2d(3, 4, 3, padding=1).eval()
    x = np.random.default_rng(0).normal(size=(70, 90, 3)).astype(np.float32)

    def predict(tiles: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return model(torch.from_numpy(tiles)).numpy()

    np.testing.assert_allclose(tiled_predict(predict, x, 32, 24, 4, batch=3),
                               tiled_logits(model, x, 32, 24, 4, "cpu", batch=3), atol=1e-6)


def test_astar_crosses_blind_wall_when_cheap_and_detours_when_not() -> None:
    grid = make_grid(wall_with_gap())
    start, goal = (30, 5), (30, 55)
    direct = astar(grid, start, goal, weight=0.0)
    detour = astar(grid, start, goal, weight=50.0)
    assert all(r == 30 for r, _ in direct)
    assert min(r for r, _ in detour) < 5


def test_astar_respects_blocked_cells() -> None:
    blocked = np.zeros((40, 60), bool)
    blocked[:35, 30] = True
    path = astar(make_grid(np.zeros((40, 60)), blocked), (5, 5), (5, 55), weight=0.0)
    assert not any(blocked[c] for c in path)
    assert max(r for r, _ in path) >= 35


def test_straighten_open_ground_is_one_leg() -> None:
    grid = make_grid(np.zeros((40, 60)))
    cells = astar(grid, (3, 3), (35, 50), weight=0.0)
    assert len(straighten(grid, cells, 0.0, 0.05)) == 2


def test_straighten_keeps_detour_around_wall() -> None:
    grid = make_grid(wall_with_gap())
    cells = astar(grid, (30, 5), (30, 55), weight=50.0)
    legs = straighten(grid, cells, 50.0, 0.05)
    assert 2 < len(legs) < len(cells)
    assert min(r for r, _ in legs) < 6


def test_blind_stretches() -> None:
    distances = np.arange(10, dtype=float) * 5
    blind = np.array([0, 0, 1, 1, 1, 0, 0, 1, 0, 0], float)
    stretches = blind_stretches(distances, blind)
    assert [(s.start_m, s.end_m) for s in stretches] == [(5.0, 25.0), (30.0, 40.0)]


def test_drift_grows_and_limit_is_consistent() -> None:
    ins = InsConfig()
    assert drift_sigma_m(200, 5, ins) > drift_sigma_m(100, 5, ins) > 0
    limit = max_blind_distance_m(5, 10.0, ins, 2.0)
    assert 2 * drift_sigma_m(limit, 5, ins) == pytest.approx(10.0, rel=1e-6)
    assert max_blind_distance_m(5, 10.0, ins, 2.0, ekf_coast_s=1.0) == pytest.approx(5.0)


def test_plan_routes_offers_shortest_and_detour() -> None:
    grid = make_grid(wall_with_gap())
    inputs = MissionInputs(grid.latlon_of(30.5, 5.5), grid.latlon_of(30.5, 55.5), max_flight_time_s=600)
    config = PlannerConfig(route_weights=(0.0, 50.0))
    routes = plan_routes(grid, inputs, config, blind_limit_m=20.0)
    assert len(routes) == 2
    shortest, detour = routes
    assert shortest.length_m < detour.length_m
    assert shortest.exceeds_blind_limit and not detour.exceeds_blind_limit
    assert shortest.waypoints[0] == inputs.start and shortest.waypoints[-1] == inputs.end


def test_plan_routes_drops_duplicates() -> None:
    grid = make_grid(np.zeros((40, 60)))
    inputs = MissionInputs(grid.latlon_of(5.5, 5.5), grid.latlon_of(30.5, 50.5))
    routes = plan_routes(grid, inputs, PlannerConfig(route_weights=(0.0, 5.0, 20.0)), blind_limit_m=100.0)
    assert len(routes) == 1


class FakeElevation:
    def elevations_m(self, points: list[LatLon]) -> np.ndarray:
        return np.linspace(100, 130, len(points))    # ground rises 30 m along the route


def test_elevation_check_warns_on_rising_ground() -> None:
    waypoints = [ORIGIN, LatLon(ORIGIN.lat, ORIGIN.lon + 0.03)]
    check = check_elevation(waypoints, 40.0, FakeElevation())
    assert check.height_above_ground_m.min() == pytest.approx(10.0)
    assert len(check.warnings) == 2
