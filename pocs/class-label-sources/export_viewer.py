"""Export full-resolution layers for the zoomable inspection viewer.

The overview figure is 760 px wide per panel, which is fine for "did DA find
anything" and useless for "is that parcel sitting on a house". This writes the
layers at native 0.385 m/px so they can be zoomed to the pixel.

Layers are RGBA PNGs with transparent background so the viewer can stack them
over the imagery and toggle each independently. Sparse content, so they
compress to a fraction of the JPEG.
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

from field_da import POLY_DIR, burn, load_location, osm_masks  # noqa: E402

OUT = Path("data/derived/class-label-sources/viewer")


def rgba(mask: np.ndarray, colour: tuple[int, int, int], alpha: int) -> Image.Image:
    """A mask as a flat-coloured RGBA layer, transparent where the mask is 0."""
    h, w = mask.shape
    out = np.zeros((h, w, 4), np.uint8)
    hit = mask > 0
    out[hit, 0], out[hit, 1], out[hit, 2] = colour
    out[hit, 3] = alpha
    return Image.fromarray(out, "RGBA")


def export(name: str, quality: int, max_width: int) -> dict:
    image, grid, meta = load_location(name)
    shape = tuple(meta["shape"])
    data = np.load(POLY_DIR / f"{name}.npz")
    polygons = [data[k] for k in data.files]

    _, building = osm_masks(grid, shape)
    fill = burn(polygons, shape)
    outline = burn(polygons, shape, outline=True, width_px=3)

    scale = min(1.0, max_width / shape[1])
    size = (round(shape[1] * scale), round(shape[0] * scale))

    def save(img: Image.Image, suffix: str, **kw):
        if scale < 1.0:
            img = img.resize(size, Image.LANCZOS if img.mode == "RGB" else Image.NEAREST)
        path = OUT / f"{name}-{suffix}"
        img.save(path, **kw)
        return path.name, path.stat().st_size

    files = dict(
        [
            save(image, "photo.jpg", quality=quality, optimize=True),
            save(rgba(outline, (255, 232, 40), 255), "outline.png", optimize=True),
            save(rgba(fill, (60, 210, 100), 90), "fill.png", optimize=True),
            save(rgba(building, (240, 60, 60), 150), "bldg.png", optimize=True),
        ]
    )
    return {
        "name": name,
        "label": name.rsplit("-", 2)[0],
        "width": size[0],
        "height": size[1],
        "mpp": meta["mpp"] / scale,
        "polygons": len(polygons),
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--quality", type=int, default=82)
    parser.add_argument("--max-width", type=int, default=3328)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name in args.names:
        entry = export(name, args.quality, args.max_width)
        total = sum(entry["files"].values()) / 1e6
        print(f"{entry['label']:14s} {entry['width']}x{entry['height']}  {total:5.2f} MB", flush=True)
        manifest.append(entry)

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"total {sum(sum(e['files'].values()) for e in manifest)/1e6:.1f} MB -> {OUT}")


if __name__ == "__main__":
    main()
