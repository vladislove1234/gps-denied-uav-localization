"""Fetch the Esri z18 mosaic over the 2025 flight area.

The imagery is not redistributable, so it lands in ``satelites/`` which is
gitignored. Run this before ``run.py`` / ``analyse.py`` / ``figure.py``.
"""
import json, sys
from pathlib import Path
from PIL import Image

SRC = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC / "pocs/map-derived-labels"))
from tiles import TileGrid, ESRI_CURRENT, mosaic   # noqa: E402

CENTRE = (49.76118, 24.17090)      # centroid of the 2025-04-05 flight track
SPAN_M = 700.0
OUT = HERE / "satelites/gonchary-esri-z18.jpg"


def main():
    OUT.parent.mkdir(exist_ok=True)
    grid = TileGrid.covering(*CENTRE, SPAN_M, SPAN_M, z=18)
    img = mosaic(ESRI_CURRENT, grid, SRC / "data/derived/tile-cache")
    Image.fromarray(img).save(OUT, quality=92)
    json.dump({"z": grid.z, "x0": grid.x0, "y0": grid.y0, "nx": grid.nx, "ny": grid.ny},
              open(HERE / "gonchary-grid.json", "w"), indent=1)
    print(f"{OUT}  {grid.shape}  {grid.mpp()[0]:.4f} m/px")


if __name__ == "__main__":
    main()
