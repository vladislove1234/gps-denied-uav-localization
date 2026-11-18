"""Export full-resolution woody layers for the inspector.

Same reasoning as `export_viewer.py`: probe 1's verdict was wrong because it was
written from aggregates, so probe 2 goes in front of the user at native
0.385 m/px before any verdict is written.

WorldCover cells are 10 m -- 26 px here -- so the layer is deliberately drawn
with hard edges rather than smoothed. Blockiness is the property being judged,
and softening it in the export would hide exactly what needs looking at.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "map-derived-labels"))

import worldcover as wc  # noqa: E402
from tiles import TileGrid  # noqa: E402
from woody_wc import ROOT, osm_woody  # noqa: E402

OUT = Path("data/derived/class-label-sources/viewer")
Image.MAX_IMAGE_PIXELS = None


def rgba(mask: np.ndarray, colour: tuple[int, int, int], alpha: int) -> Image.Image:
    h, w = mask.shape
    out = np.zeros((h, w, 4), np.uint8)
    hit = mask > 0
    out[hit, 0], out[hit, 1], out[hit, 2] = colour
    out[hit, 3] = alpha
    return Image.fromarray(out)


def outline_of(mask: np.ndarray, width: int = 3) -> np.ndarray:
    """Boundary pixels of a binary mask, by dilation minus erosion."""
    import cv2
    m = (mask > 0).astype(np.uint8)
    k = np.ones((width * 2 + 1, width * 2 + 1), np.uint8)
    return cv2.dilate(m, k) - cv2.erode(m, k)


def export(name: str, quality: int) -> dict:
    meta = json.loads((ROOT / name / "meta.json").read_text())
    grid, shape, mpp = TileGrid(**meta["grid"]), tuple(meta["shape"]), meta["mpp"]

    cover = wc.sample(grid, shape)
    masks = wc.class_masks(cover)
    osm = osm_woody(grid, shape, mpp)

    files = {}

    def save(img: Image.Image, suffix: str, **kw):
        path = OUT / f"{name}-{suffix}"
        img.save(path, **kw)
        files[path.name] = path.stat().st_size

    photo = OUT / f"{name}-photo.jpg"
    if not photo.exists():
        save(Image.open(ROOT / name / "current.jpg").convert("RGB"),
             "photo.jpg", quality=quality, optimize=True)
    else:
        files[photo.name] = photo.stat().st_size

    save(rgba(masks["woody"], (44, 190, 120), 96), "wcwoody.png", optimize=True)
    save(rgba(outline_of(osm), (90, 150, 255), 235), "osmwoody.png", optimize=True)
    save(rgba(masks["water"], (60, 140, 240), 120), "wcwater.png", optimize=True)

    recall = float(masks["woody"][osm].mean()) if osm.any() else None
    return {
        "id": name, "label": name.rsplit("-", 2)[0],
        "w": shape[1], "h": shape[0], "mpp": round(mpp, 4), "grid": meta["grid"],
        "wc": round(float(masks["woody"].mean()) * 100, 1),
        "osm": round(float(osm.mean()) * 100, 2),
        "recall": None if recall is None else round(recall * 100, 1),
        "water": round(float(masks["water"].mean()) * 100, 2),
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--quality", type=int, default=82)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = [export(n, args.quality) for n in args.names]
    for e in manifest:
        print(f"{e['label']:16s} WC {e['wc']:5.1f}%  OSM {e['osm']:5.2f}%  "
              f"recall {e['recall']}  {sum(e['files'].values())/1e6:.2f} MB", flush=True)
    (OUT / "woody-manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"total {sum(sum(e['files'].values()) for e in manifest)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
