"""Build `gonchary-2025-ortho`: the 2025 test-flight satellite image, as annotator input.

    python datasets/gonchary-2025-ortho/build.py        (from src/, any env with PIL)

Source is the raw Google Earth frame the position-correction harness reads
(``archive/uav_localization_2025/data/satelites/satellite_gonchary.jpg``,
8192x6026, ~0.067 m/px, 0% black -- not the ``_corrected``/``_cutted``
exports, which mask the village out). It has no exact georeference, so it is
annotated in pixels, like the drone frames.

Downsampled 3x to ~0.2 m/px: at full size one annotation layer is 49 Mpx and
the browser would need over a gigabyte for five classes plus undo. 0.2 m is
still 2x finer than the 0.385 m/px the models and the harness match at.
``scale_to_source`` maps label pixels back to the raw frame
(``x_raw = x * sx``); resize a label with NEAREST to ``source_shape``.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from PIL import Image

SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC))
Image.MAX_IMAGE_PIXELS = None

NAME = "gonchary-2025-ortho"
SLUG = "gonchary-2025"
RAW = SRC.parent / "archive/uav_localization_2025/data/satelites/satellite_gonchary.jpg"
FACTOR = 3
# Corners as recorded in the 2025 repo (gps_to_satellite_point.ipynb), via
# pocs/map-derived-labels/geo.py. A linear fit to a Google Earth frame --
# approximate, kept only for orientation and scale.
BOUNDS = {"top": 49.762965, "left": 24.166472, "bottom": 49.759291, "right": 24.174046}
OUT = Path(SRC / "data/derived" / NAME)


def main() -> None:
    im = Image.open(RAW).convert("RGB")
    W, H = im.size
    w, h = round(W / FACTOR), round(H / FACTOR)
    lat = (BOUNDS["top"] + BOUNDS["bottom"]) / 2
    width_m = (BOUNDS["right"] - BOUNDS["left"]) * 111320 * math.cos(math.radians(lat))
    d = OUT / SLUG
    d.mkdir(parents=True, exist_ok=True)
    im.resize((w, h), Image.LANCZOS).save(d / "ortho.jpg", quality=92)
    meta = {
        "name": "Gonchary 2025 ortho",
        "kind": "ortho", "crs": "pixel",
        "shape": [h, w], "mpp": width_m / w,
        "mpp_note": "from the recorded corner coordinates (approximate linear fit)",
        "source_file": str(RAW.relative_to(SRC.parent)), "source_shape": [H, W],
        "scale_to_source": [W / w, H / h],
        "bounds_approx": BOUNDS, "lat": lat, "lon": (BOUNDS["left"] + BOUNDS["right"]) / 2,
        "flight": "2025-04-05",
    }
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"{W}x{H} -> {w}x{h}, {meta['mpp']:.3f} m/px -> {d}")

    from uavloc.datasets import write_manifest
    write_manifest(NAME, version="1",
                   description="The 2025 Gonchary Google Earth frame (the harness satellite image), "
                               "downsampled 3x to ~0.2 m/px, pixel-only, as annotator input.",
                   source=str(RAW.relative_to(SRC.parent)),
                   config={"factor": FACTOR, "bounds_approx": BOUNDS},
                   samples=1, checks={"shape": [h, w], "mpp": round(meta["mpp"], 4)})


if __name__ == "__main__":
    main()
