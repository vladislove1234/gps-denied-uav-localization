"""Probe 1 -- can Delineate Anything label the `field` class over a satellite tile?

`pocs/drone-frame-boundaries/` marked DA **failed**, but that verdict is about a
*drone frame*: a 50x26 m footprint is 11% of a median 1.14 ha parcel, so there is
no parcel to find. A location tile here is 1284x988 m and holds ~100 parcels, so
the geometry that killed the earlier PoC does not apply.

What is actually being asked: does DA produce filled parcel regions that agree
with the imagery well enough to be a label source for class `field`.

Two rules inherited from the handoff and obeyed here:

* Compare at native GSD or by ground area -- never by matching pixel counts
  across domains. DA is resolution-agnostic over ~0.25-10 m/px, so the tile is
  resampled to `TARGET_MPP` before inference and polygons are scaled back.
* Measure the distribution before setting a threshold. Stage `scan` writes raw
  polygons and their statistics; filters are chosen from those numbers, not
  guessed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "map-derived-labels"))

import boundaries  # noqa: E402
import osm  # noqa: E402
from tiles import TileGrid  # noqa: E402

Image.MAX_IMAGE_PIXELS = None

ROOT = Path("data/derived/ua-village-multidate")
OUT = Path(__file__).resolve().parent
POLY_DIR = Path("data/derived/class-label-sources/field-polygons")

# OSM tags that mean "agricultural surface". `osm.classify()` has no branch for
# any of these -- farmland is exactly the class the built masks are missing.
FARM_LANDUSE = ("farmland", "meadow", "orchard", "vineyard", "greenhouse_horticulture")


def locations() -> list[str]:
    return json.loads((ROOT / "index.json").read_text())["built"]


def load_location(name: str, date: str = "current"):
    """Return `(image, grid, meta)` for one location at full resolution.

    The grid rather than a `geo.Bounds`: the dataset build rasterises OSM
    through `TileGrid.to_pixel`, which is true Web Mercator. `Bounds.to_pixel`
    is linear in latitude and drifts from it across the mosaic. Using the grid
    is what makes these labels land where the dataset's own labels land.
    """
    meta = json.loads((ROOT / name / "meta.json").read_text())
    image = Image.open(ROOT / name / f"{date}.jpg").convert("RGB")
    return image, TileGrid(**meta["grid"]), meta


def burn(polygons, shape, fill: int = 1, outline: bool = False, width_px: int = 2):
    """Burn polygons into a mask of `(height, width)`, filled or as outlines."""
    height, width = shape[:2]
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    for polygon in polygons:
        points = [tuple(map(float, p)) for p in polygon]
        if len(points) < 3:
            continue
        if outline:
            draw.line(points + [points[0]], fill=fill, width=width_px, joint="curve")
        else:
            draw.polygon(points, fill=fill)
    return np.array(canvas)


def osm_masks(grid: TileGrid, shape):
    """`(farmland, building)` masks from the cached Overpass response.

    Both are yardsticks, not labels. Farmland says where DA *should* find
    parcels; buildings say where it should not, since the handoff records DA
    over-segmenting built-up areas into nested near-duplicate rectangles.
    """
    elements = osm.fetch(grid.image_bounds())
    farm, built = [], []
    for element in elements:
        geometry = element.get("geometry")
        tags = element.get("tags", {})
        if not geometry or len(geometry) < 3:
            continue
        points = [grid.to_pixel(p["lat"], p["lon"], shape) for p in geometry]
        if tags.get("landuse") in FARM_LANDUSE:
            farm.append(points)
        elif "building" in tags:
            built.append(points)
    return burn(farm, shape), burn(built, shape)


def polygon_area_px(polygon: np.ndarray) -> float:
    """Shoelace area in pixels."""
    x, y = polygon[:, 0], polygon[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def describe(polygon: np.ndarray, mpp: float) -> dict:
    """Per-polygon statistics used to choose filters after the fact."""
    area_px = polygon_area_px(polygon)
    x0, y0 = polygon.min(axis=0)
    x1, y1 = polygon.max(axis=0)
    bbox_px = max(1.0, (x1 - x0) * (y1 - y0))
    side = max(1.0, max(x1 - x0, y1 - y0))
    return {
        "area_m2": area_px * mpp * mpp,
        "fill": area_px / bbox_px,          # 1.0 = perfect rectangle
        "aspect": float(min(x1 - x0, y1 - y0) / side),
        "side_m": float(side * mpp),
        "vertices": int(len(polygon)),
    }


def scan(names: list[str], device: str, conf: float, tile: int, variant: str) -> dict:
    """Run DA over each location, save polygons, return per-location statistics."""
    POLY_DIR.mkdir(parents=True, exist_ok=True)
    model = boundaries.load(variant)
    results = {}

    for index, name in enumerate(names, 1):
        started = time.time()
        image, grid, meta = load_location(name)
        shape = tuple(meta["shape"])
        mpp = meta["mpp"]

        small = boundaries.scale_for_model(image, grid.bounds(), boundaries.TARGET_MPP)
        scale = image.size[0] / small.size[0]
        polygons = boundaries.predict(
            model, small, tile=tile, conf=conf, device=device, min_valid=0.9
        )
        polygons = [p * scale for p in polygons]

        stats = [describe(p, mpp) for p in polygons]
        np.savez_compressed(
            POLY_DIR / f"{name}.npz",
            **{f"p{i}": p.astype(np.float32) for i, p in enumerate(polygons)},
        )

        farm, built = osm_masks(grid, shape)
        field = burn(polygons, shape)
        results[name] = {
            "polygons": len(polygons),
            "mpp": mpp,
            "shape": list(shape),
            "seconds": round(time.time() - started, 1),
            "coverage": {
                "da_field": float(field.mean()),
                "osm_farmland": float(farm.mean()),
                "osm_building": float(built.mean()),
            },
            "agreement": {
                # of the ground OSM calls farmland, how much did DA cover
                "recall_on_farmland": float(field[farm > 0].mean()) if farm.any() else None,
                # of what DA called field, how much sits on a building
                "leak_onto_buildings": float(built[field > 0].mean()) if field.any() else None,
            },
            "stats": stats,
        }
        print(
            f"[{index}/{len(names)}] {name}: {len(polygons)} polys, "
            f"DA {field.mean()*100:.1f}% / OSM-farm {farm.mean()*100:.1f}%, "
            f"{results[name]['seconds']}s",
            flush=True,
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=6, help="how many locations")
    parser.add_argument("--names", nargs="*", default=None)
    parser.add_argument("--device", default="cpu", help="CPU beats MPS for DA")
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--tile", type=int, default=1024)
    parser.add_argument("--variant", default="v2", choices=["v2", "small"])
    parser.add_argument("--out", default=str(OUT / "results-field.json"))
    args = parser.parse_args()

    names = args.names or locations()[: args.n]
    results = scan(names, args.device, args.conf, args.tile, args.variant)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
