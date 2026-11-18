"""Probe 4b evidence: every mask, at four ground extents.

One row per ground extent, one column per sampled building. OSM building in
green; every predicted mask drawn, red if it covers >50% of the frame and
orange otherwise.

The first version of this figure plotted only the LARGEST mask per crop and was
misleading -- it supported a claim ("the model cannot draw a small box") that
the full data contradicts. 27% of masks are small. They are simply almost never
on the building. Draw them all.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import yoloe_probe as probe  # noqa: E402
import yoloe_smallcrop as small  # noqa: E402

warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = None

OUT = Path(__file__).resolve().parent / "figures"
EXTENTS = [25, 50, 100, 200]
IMGSZ = 640
LOC = "maidan-50p2512-26p0155"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    from ultralytics import YOLOE

    model = YOLOE("yoloe-11l-seg.pt")
    prompts = probe.PROMPTS["satellite"]
    model.set_classes(prompts, model.get_text_pe(prompts))

    image, label, meta = probe.load_location(LOC)
    control = probe.control_masks(label, meta)["building"]
    mpp = meta["mpp"]
    height, width = meta["shape"]
    rng = np.random.default_rng(3)
    centres = small.building_centres(control, 4, min_px=200, rng=rng)

    fig, axes = plt.subplots(len(EXTENTS), len(centres),
                             figsize=(4.0 * len(centres), 4.0 * len(EXTENTS)))
    for row, extent in enumerate(EXTENTS):
        for col, (cx, cy, blob) in enumerate(centres):
            half = int(round(extent / mpp / 2))
            x0, y0 = max(0, cx - half), max(0, cy - half)
            x1, y1 = min(width, cx + half), min(height, cy + half)
            crop = image.crop((x0, y0, x1, y1)).resize((IMGSZ, IMGSZ), Image.LANCZOS)
            target = np.array(Image.fromarray(
                (blob[y0:y1, x0:x1]).astype("uint8") * 255)
                .resize((IMGSZ, IMGSZ), Image.NEAREST)) > 127

            result = model.predict(np.array(crop), imgsz=IMGSZ, conf=0.05, device="cpu",
                                   verbose=False, retina_masks=True)[0]
            ax = axes[row, col]
            ax.imshow(crop)
            ax.contour(target, levels=[0.5], colors="#3cb44b", linewidths=2.2)

            # EVERY mask, not just the largest. Plotting only the biggest one
            # hid that ~27% of masks are small -- they are simply almost never
            # on the building. Colour by size so both facts are visible at once.
            fracs, best_iou = [], 0.0
            if result.masks is not None and len(result.masks.xy):
                for poly in result.masks.xy:
                    mask = probe.burn([poly], (IMGSZ, IMGSZ))
                    frac = mask.mean()
                    fracs.append(frac)
                    inter = (mask & target).sum()
                    union = (mask | target).sum()
                    if union:
                        best_iou = max(best_iou, inter / union)
                    colour = "#e6194b" if frac > 0.5 else "#ff8c00"
                    ax.contour(mask, levels=[0.5], colors=colour,
                               linewidths=2.0 if frac > 0.5 else 1.5)

            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(f"{extent:.0f} m across", fontsize=13)
            n = len(fracs)
            small_n = sum(1 for f in fracs if f <= 0.5)
            ax.set_title(f"{n} masks ({small_n} small) · best IoU {best_iou:.2f}",
                         fontsize=10)

    handles = [plt.Line2D([], [], color="#3cb44b", lw=3, label="OSM building"),
               plt.Line2D([], [], color="#e6194b", lw=3, label="mask > 50% of frame"),
               plt.Line2D([], [], color="#ff8c00", lw=3, label="mask \u2264 50% of frame")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=13)
    fig.suptitle("Probe 4b — every mask drawn: small ones exist, they are just not on the building",
                 fontsize=15)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(OUT / "yoloe-smallcrop.png", dpi=105)
    print(f"wrote {OUT}/yoloe-smallcrop.png")


if __name__ == "__main__":
    main()
