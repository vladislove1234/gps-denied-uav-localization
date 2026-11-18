"""Probe 4b -- can YOLOE draw a SMALL box, or only a frame-sized one?

The user's observation from `figures/yoloe-overview.png`: the predicted masks
are roughly tile-sized rectangles. That is a different failure from "the model
does not understand nadir imagery" -- it would mean the model is emitting one
blanket box per frame regardless of content, and that feeding it a smaller patch
containing a single object might get a usable box.

So this asks one question directly: crop a window centred on ONE known OSM
building, at four ground extents, and measure

  * `frame_frac` -- area of the largest mask / area of the frame. This is the
    user's observation turned into a number. Near 1.0 means the model cannot
    draw small, whatever the content.
  * `best_iou`   -- best IoU of any mask against the target building.
  * `hit`        -- did any mask overlap the building at all.

Ground extent is varied, not upsample factor (bug-015). Every crop is resized to
one `imgsz` so the only thing changing is how much ground is in frame and
therefore how large the building appears.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import yoloe_probe as probe  # noqa: E402

warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = None

OUT = Path(__file__).resolve().parent


def building_centres(control: np.ndarray, want: int, min_px: int, rng) -> list:
    """Centroids of OSM building blobs, sampled, largest-first bias removed."""
    labelled, count = ndimage.label(control)
    if count == 0:
        return []
    sizes = ndimage.sum(control, labelled, range(1, count + 1))
    keep = [i + 1 for i, s in enumerate(sizes) if s >= min_px]
    if not keep:
        return []
    picked = rng.choice(keep, size=min(want, len(keep)), replace=False)
    out = []
    for index in picked:
        ys, xs = np.where(labelled == index)
        out.append((int(xs.mean()), int(ys.mean()), labelled == index))
    return out


def run(names, extents_m, per_loc, imgsz, conf, variant, device, seed):
    from ultralytics import YOLOE

    model = YOLOE(variant)
    prompts = probe.PROMPTS["satellite"]
    model.set_classes(prompts, model.get_text_pe(prompts))
    rng = np.random.default_rng(seed)
    rows = []

    for name in names:
        image, label, meta = probe.load_location(name)
        control = probe.control_masks(label, meta)["building"]
        mpp = meta["mpp"]
        height, width = meta["shape"]
        centres = building_centres(control, per_loc, min_px=60, rng=rng)

        for cx, cy, blob in centres:
            for extent in extents_m:
                half = int(round(extent / mpp / 2))
                x0, y0 = max(0, cx - half), max(0, cy - half)
                x1, y1 = min(width, cx + half), min(height, cy + half)
                if x1 - x0 < 16 or y1 - y0 < 16:
                    continue
                crop = image.crop((x0, y0, x1, y1)).resize((imgsz, imgsz), Image.LANCZOS)
                target = blob[y0:y1, x0:x1]
                sx, sy = imgsz / (x1 - x0), imgsz / (y1 - y0)
                target_small = np.array(Image.fromarray(target.astype("uint8") * 255)
                                        .resize((imgsz, imgsz), Image.NEAREST)) > 127

                result = model.predict(np.array(crop), imgsz=imgsz, conf=conf,
                                       device=device, verbose=False, retina_masks=True)[0]
                if result.masks is None or len(result.masks.xy) == 0:
                    rows.append({"loc": name, "extent_m": extent, "dets": 0,
                                 "frame_frac": None, "best_iou": 0.0, "hit": False,
                                 "building_frac": float(target_small.mean())})
                    continue

                best_iou, biggest = 0.0, 0.0
                hit = False
                for polygon in result.masks.xy:
                    mask = probe.burn([polygon], (imgsz, imgsz))
                    frac = mask.mean()
                    biggest = max(biggest, frac)
                    inter = (mask & target_small).sum()
                    union = (mask | target_small).sum()
                    if inter > 0:
                        hit = True
                    if union:
                        best_iou = max(best_iou, inter / union)
                rows.append({"loc": name, "extent_m": extent,
                             "dets": len(result.masks.xy),
                             "frame_frac": float(biggest), "best_iou": float(best_iou),
                             "hit": bool(hit),
                             "building_frac": float(target_small.mean())})
        print(f"  {name}: {len(centres)} buildings x {len(extents_m)} extents", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=9)
    parser.add_argument("--per-loc", type=int, default=8)
    parser.add_argument("--extents", type=float, nargs="*", default=[25, 50, 100, 200])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--variant", default="yoloe-11l-seg.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    probe1 = json.loads((OUT / "results-field.json").read_text())
    names = list(probe1)[: args.n]
    rows = run(names, args.extents, args.per_loc, args.imgsz, args.conf,
               args.variant, args.device, args.seed)
    (OUT / "results-yoloe-smallcrop.json").write_text(
        json.dumps({"config": vars(args), "rows": rows}, indent=2))

    print(f"\n{'extent':>8s} {'n':>4s} {'dets':>6s} {'frame_frac':>11s} "
          f"{'bldg_frac':>10s} {'hit%':>6s} {'best_iou':>9s}")
    for extent in args.extents:
        sub = [r for r in rows if r["extent_m"] == extent]
        if not sub:
            continue
        ff = [r["frame_frac"] for r in sub if r["frame_frac"] is not None]
        print(f"{extent:8.0f} {len(sub):4d} {np.mean([r['dets'] for r in sub]):6.1f} "
              f"{(np.median(ff) if ff else float('nan')):11.3f} "
              f"{np.mean([r['building_frac'] for r in sub]):10.3f} "
              f"{100*np.mean([r['hit'] for r in sub]):6.1f} "
              f"{np.mean([r['best_iou'] for r in sub]):9.3f}")


if __name__ == "__main__":
    main()
