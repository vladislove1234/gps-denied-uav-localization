"""Evidence figures for probe 4.

Two figures, because the probe has two separable claims.

`yoloe-overview.png` is the one that matters: imagery, prediction and OSM
control side by side, plus a native-resolution zoom. Probe 1 was written up as
`worked` on an aggregate score and then rejected on sight, so the rule now is
that the picture goes in before the verdict does.

`yoloe-scores.png` carries the numbers, including the null baseline -- the
prediction randomly translated, which preserves blob count and coverage and
destroys only position.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

import yoloe_probe as probe  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures"
COLOURS = {"building": "#e6194b", "track": "#ffe119", "woody": "#3cb44b", "water": "#4363d8"}


def overlay(base: Image.Image, masks: dict[str, np.ndarray], alpha: float = 0.55):
    rgb = np.array(base.convert("RGB"), dtype=float) / 255
    for name, mask in masks.items():
        if mask is None or not mask.any():
            continue
        colour = np.array(matplotlib.colors.to_rgb(COLOURS[name]))
        rgb[mask] = (1 - alpha) * rgb[mask] + alpha * colour
    return rgb


def predicted_masks(model, image, meta, conf, tile, device):
    found = probe.predict_multiclass(model, image, tile, 128, conf, device)
    shape = tuple(meta["shape"])
    return {n: probe.burn([p for c, p, _ in found if c == i], shape)
            for i, n in enumerate(probe.CLASSES)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=None)
    parser.add_argument("--names", nargs="*", default=None)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--tile", type=int, default=1024)
    parser.add_argument("--variant", default="yoloe-11l-seg.pt")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    OUT.mkdir(exist_ok=True)
    path = args.results or str(Path(__file__).resolve().parent /
                               "results-yoloe-text-satellite.json")
    data = json.loads(Path(path).read_text())["locations"]
    names = args.names or list(data)[:3]

    from ultralytics import YOLOE
    model = YOLOE(args.variant)
    prompts = probe.PROMPTS["satellite"]
    model.set_classes(prompts, model.get_text_pe(prompts))

    # ---- overview -------------------------------------------------------
    fig, axes = plt.subplots(len(names), 4, figsize=(19, 4.6 * len(names)))
    axes = np.atleast_2d(axes)
    for row, name in enumerate(names):
        image, label, meta = probe.load_location(name)
        control = probe.control_masks(label, meta)
        pred = predicted_masks(model, image, meta, args.conf, args.tile, args.device)
        small = (image.size[0] // 3, image.size[1] // 3)

        axes[row, 0].imshow(image.resize(small))
        axes[row, 0].set_title(f"{name.split('-')[0]} — imagery", fontsize=11)

        axes[row, 1].imshow(np.array(Image.fromarray(
            (overlay(image, pred) * 255).astype("uint8")).resize(small)))
        axes[row, 1].set_title("YOLOE text prompt", fontsize=11)

        axes[row, 2].imshow(np.array(Image.fromarray(
            (overlay(image, control) * 255).astype("uint8")).resize(small)))
        axes[row, 2].set_title("OSM control", fontsize=11)

        # native-resolution zoom on the densest built-up corner
        ys, xs = np.where(control["building"])
        if len(xs):
            cx, cy = int(np.median(xs)), int(np.median(ys))
        else:
            cy, cx = meta["shape"][0] // 2, meta["shape"][1] // 2
        half = 400
        box = (max(0, cx - half), max(0, cy - half),
               min(meta["shape"][1], cx + half), min(meta["shape"][0], cy + half))
        zoom_rgb = overlay(image.crop(box),
                           {k: v[box[1]:box[3], box[0]:box[2]] for k, v in pred.items()})
        axes[row, 3].imshow(zoom_rgb)
        axes[row, 3].set_title("native 0.386 m/px — prediction", fontsize=11)

        for ax in axes[row]:
            ax.set_xticks([]); ax.set_yticks([])

    handles = [plt.Line2D([], [], marker="s", ls="", ms=11, color=c, label=n)
               for n, c in COLOURS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=12)
    fig.suptitle("Probe 4 — YOLOE open-vocabulary prompting over z18 satellite tiles",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0.035, 1, 0.97])
    fig.savefig(OUT / "yoloe-overview.png", dpi=110)
    plt.close(fig)

    # ---- scores ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    classes = probe.CLASSES
    x = np.arange(len(classes))

    real, null = [], []
    for name in classes:
        r = [data[k]["classes"][name]["instance_recall"] for k in data
             if data[k]["classes"][name]["instance_recall"] is not None]
        n = [data[k]["classes"][name]["null_recall"] for k in data
             if data[k]["classes"][name]["null_recall"] is not None]
        real.append(np.mean(r) if r else 0)
        null.append(np.mean(n) if n else 0)
    axes[0].bar(x - 0.2, real, 0.4, label="YOLOE", color="#4363d8")
    axes[0].bar(x + 0.2, null, 0.4, label="null (randomly translated)", color="#999999")
    axes[0].set_xticks(x); axes[0].set_xticklabels(classes)
    axes[0].set_ylabel("instance recall (any overlap)")
    axes[0].set_title("Recall against a position-destroying null")
    axes[0].legend(frameon=False)

    pred_c, osm_c = [], []
    for name in classes:
        pred_c.append(np.mean([data[k]["classes"][name]["pred_cover"] for k in data]) * 100)
        osm_c.append(np.mean([data[k]["classes"][name]["osm_cover"] for k in data]) * 100)
    axes[1].bar(x - 0.2, pred_c, 0.4, label="YOLOE predicted", color="#e6194b")
    axes[1].bar(x + 0.2, osm_c, 0.4, label="OSM actual", color="#3cb44b")
    axes[1].set_xticks(x); axes[1].set_xticklabels(classes)
    axes[1].set_ylabel("% of tile covered")
    axes[1].set_title("How much of the tile each class claims")
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(OUT / "yoloe-scores.png", dpi=110)
    print(f"wrote {OUT}/yoloe-overview.png and yoloe-scores.png")


if __name__ == "__main__":
    main()
