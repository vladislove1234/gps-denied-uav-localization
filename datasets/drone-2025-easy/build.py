"""Build `drone-2025-easy`: the 44 hand-picked 2025 drone frames, as annotator input.

    python datasets/drone-2025-easy/build.py        (from src/, any env with PIL)

The frame set is the union of the two "easy" folders from the 2025 work
(``images_2_v3_renamed_small_easy`` 25 + ``images_v6_renamed_easy`` 26, 7
shared) -- the frames with roads used for validation. 11 of those files are
0 bytes, so every frame is copied from the complete parent folders
(``images_2_v3_renamed`` / ``images_v6_renamed``) by timestamp instead.

One frame = one location for the annotator, with a single image ``frame.jpg``
and no georeference: labels are drawn in frame pixels. Position, altitude and
yaw come from the filenames, which the 2025 pipeline wrote from the MAVLink log
(``gps_data.csv``, join gap <= 100 ms).

``mpp`` is an approximate ground sample distance for sizing brushes and road
widths in metres: the camera's 63 deg horizontal FOV at the logged altitude,
assuming a nadir view. Tilt and terrain are ignored, so treat it as +-20%.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import sys
from pathlib import Path

from PIL import Image

SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC))

NAME = "drone-2025-easy"
DRONE = SRC.parent / "archive/uav_localization_2025/data/drone"
PICKS = ["images_2_v3_renamed_small_easy", "images_v6_renamed_easy"]
PARENTS = ["images_2_v3_renamed", "images_v6_renamed"]
FOV_W_DEG = 63.0          # dataset_constructor_test.ipynb, 2025 repo
OUT = SRC / "data/derived" / NAME
PAT = re.compile(r"^(\d{13})_lat([\d.]+)_lon([\d.]+)_alt([\d.]+)_yaw([\d.]+)\.jpg$")


def main() -> None:
    picks = sorted({f.name.split("_")[0] for d in PICKS for f in (DRONE / d).iterdir() if f.suffix == ".jpg"})
    originals = {}
    for d in PARENTS:
        for f in (DRONE / d).iterdir():
            m = PAT.match(f.name)
            if m and f.stat().st_size > 0:
                originals.setdefault(m.group(1), f)
    missing = [t for t in picks if t not in originals]
    if missing:
        raise SystemExit(f"no intact original for {missing}")

    OUT.mkdir(parents=True, exist_ok=True)
    for ts in picks:
        src = originals[ts]
        _, lat, lon, alt, yaw = PAT.match(src.name).groups()
        lat, lon, alt, yaw = float(lat), float(lon), float(alt), float(yaw)
        w, h = Image.open(src).size
        slug = f"drone-{ts}"
        d = OUT / slug
        d.mkdir(exist_ok=True)
        shutil.copy2(src, d / "frame.jpg")
        mpp = 2 * alt * math.tan(math.radians(FOV_W_DEG / 2)) / w
        from datetime import datetime, timezone
        hhmmss = datetime.fromtimestamp(int(ts) / 1000, timezone.utc).strftime("%H:%M:%S")
        meta = {
            "name": f"{hhmmss} UTC · {alt:.0f} m",
            "kind": "drone", "crs": "pixel",
            "shape": [h, w], "mpp": mpp, "mpp_note": "nadir approximation from altitude and 63 deg FOV, +-20%",
            "timestamp_ms": int(ts), "lat": lat, "lon": lon, "alt_m": alt, "yaw_deg": yaw,
            "flight": "2025-04-05", "source_file": str(src.relative_to(SRC.parent)),
            "picked_from": [p for p in PICKS if any((DRONE / p).glob(f"{ts}_*"))],
        }
        (d / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"{len(picks)} frames -> {OUT}")

    from uavloc.datasets import write_manifest
    write_manifest(NAME, version="1",
                   description="44 hand-picked 2025 drone frames with roads (the 'easy' validation frames), "
                               "one folder per frame with pixel-only metadata, as annotator input.",
                   source=f"../archive/uav_localization_2025/data/drone/{{{','.join(PARENTS)}}} selected by {PICKS}",
                   config={"picks": PICKS, "parents": PARENTS, "fov_w_deg": FOV_W_DEG},
                   samples=len(picks),
                   checks={"frames": len(picks), "zero_byte_in_picks_replaced_from_parent": 11})


if __name__ == "__main__":
    main()
