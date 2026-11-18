"""Probe 2 -- can ESA WorldCover supply the `woody` class where OSM cannot?

OSM's answer for woody over these 59 locations is thin: median coverage 0.12%,
nothing at all at 24 of them, and 32 `tree_row` ways in total. The question is
whether a 10 m land-cover raster fills that gap well enough to label with.

The comparison is deliberately one-sided. OSM woody is *sparse but sharp*, so
where OSM says wood there really is wood -- it is a precision yardstick, not a
recall one. WorldCover covering OSM's woods is therefore the minimum bar, not
evidence of correctness; only looking at the imagery settles that, which is why
this probe ends at a viewer rather than at a number.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "map-derived-labels"))

import worldcover as wc  # noqa: E402
from tiles import TileGrid  # noqa: E402

ROOT = Path("data/derived/ua-village-multidate")
OSM_CACHE = Path("data/derived/osm-cache")
OUT = Path("data/derived/class-label-sources/worldcover")


def osm_woody(grid: TileGrid, shape: tuple[int, int], mpp: float,
              treeline_m: float = 10.0) -> np.ndarray:
    """OSM's woody mask: wood/scrub/forest polygons plus treelines.

    Treelines are drawn at `treeline_m` rather than the 2 m `bug-013` fall-through.
    That bug is real but costs nothing measurable here -- across all 59 locations
    there are 32 tree_row ways totalling 6.1 km, and widening them 5x leaves
    median coverage at 0.000%.
    """
    height, width = shape[:2]
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)

    key = hashlib.sha1(grid.image_bounds().as_overpass().encode()).hexdigest()[:12]
    elements = json.loads((OSM_CACHE / f"{key}.json").read_text())

    for element in elements:
        geometry = element.get("geometry")
        tags = element.get("tags", {})
        if not geometry:
            continue
        points = [grid.to_pixel(p["lat"], p["lon"], shape) for p in geometry]
        natural, landuse = tags.get("natural"), tags.get("landuse")
        if (natural in ("wood", "scrub") or landuse == "forest") and len(points) >= 3:
            draw.polygon(points, fill=1)
        elif (natural == "tree_row" or tags.get("barrier") == "hedge") and len(points) >= 2:
            draw.line(points, fill=1, width=max(1, round(treeline_m / mpp)), joint="curve")

    return np.array(canvas) > 0


def run(names: list[str]) -> dict:
    results = {}
    for index, name in enumerate(names, 1):
        started = time.time()
        meta = json.loads((ROOT / name / "meta.json").read_text())
        grid, shape, mpp = TileGrid(**meta["grid"]), tuple(meta["shape"]), meta["mpp"]

        cover = wc.sample(grid, shape)
        masks = wc.class_masks(cover)
        woody, osm = masks["woody"], osm_woody(grid, shape, mpp)

        np.save(OUT / f"{name}.npy", cover)
        histogram = {
            wc.LEGEND.get(int(v), str(int(v))): round(float(n) / cover.size * 100, 2)
            for v, n in zip(*np.unique(cover, return_counts=True))
        }
        results[name] = {
            "seconds": round(time.time() - started, 2),
            "histogram": histogram,
            "coverage": {
                "wc_woody": float(woody.mean()),
                "osm_woody": float(osm.mean()),
                "wc_water": float(masks["water"].mean()),
                "wc_built": float(masks["built"].mean()),
            },
            # Does WorldCover cover the wood OSM is confident about?
            "recall_on_osm_woody": float(woody[osm].mean()) if osm.any() else None,
        }
        recall = results[name]["recall_on_osm_woody"]
        print(
            f"[{index}/{len(names)}] {name.rsplit('-', 2)[0]:22s} "
            f"WC {woody.mean()*100:5.1f}%  OSM {osm.mean()*100:5.2f}%  "
            f"recall {'--' if recall is None else f'{recall*100:5.1f}%'}",
            flush=True,
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", nargs="*", default=None)
    parser.add_argument("--out", default="pocs/class-label-sources/results-woody.json")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    names = args.names or json.loads((ROOT / "index.json").read_text())["built"]
    results = run(names)
    Path(args.out).write_text(json.dumps(results, indent=2))

    woody = np.array([r["coverage"]["wc_woody"] for r in results.values()])
    osm = np.array([r["coverage"]["osm_woody"] for r in results.values()])
    recall = np.array([r["recall_on_osm_woody"] for r in results.values()
                       if r["recall_on_osm_woody"] is not None])
    print(f"\nWorldCover woody: median {np.median(woody)*100:.1f}%  "
          f"zero at {(woody == 0).sum()}/{len(woody)}  >=1% at {(woody >= 0.01).sum()}/{len(woody)}")
    print(f"OSM woody:        median {np.median(osm)*100:.2f}%  "
          f"zero at {(osm == 0).sum()}/{len(osm)}  >=1% at {(osm >= 0.01).sum()}/{len(osm)}")
    print(f"recall on OSM woody ({len(recall)} locations): "
          f"median {np.median(recall)*100:.1f}%  p10 {np.percentile(recall, 10)*100:.1f}%")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
