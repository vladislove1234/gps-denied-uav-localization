"""Run both label sources over the 2025 Gonchary orthophoto and draw the evidence.

    python pocs/map-derived-labels/run_gonchary.py

Answers one question: do OSM features and Delineate Anything boundaries line up
with Google Earth imagery well enough to use as segmentation labels? Writes
three figures to ``figures/`` and prints the numbers behind them.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
Image.MAX_IMAGE_PIXELS = None

import align                      # noqa: E402
import boundaries as da           # noqa: E402
import osm                        # noqa: E402
from geo import GONCHARY          # noqa: E402

SRC = HERE.parents[1]
ORTHO = SRC.parent / "archive/uav_localization_2025/data/satelites/satellite_gon_corrected.jpg"
DERIVED = SRC / "data/derived"
FIGURES = HERE / "figures"
WORK_MPP = 0.25          # the resolution Delineate Anything performed best at
CONF = 0.10


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    FIGURES.mkdir(exist_ok=True)
    (DERIVED / "boundaries").mkdir(parents=True, exist_ok=True)

    print(f"orthophoto  {ORTHO.name}")
    image = Image.open(ORTHO)
    image.load()
    print(f"  native    {image.size[0]}x{image.size[1]}  "
          f"{GONCHARY.width_m():.0f} x {GONCHARY.height_m():.0f} m  "
          f"{GONCHARY.mpp((image.size[1], image.size[0]))[0]*100:.2f} cm/px")

    work = da.scale_for_model(image, GONCHARY, WORK_MPP)
    shape = (work.size[1], work.size[0])
    print(f"  working   {work.size[0]}x{work.size[1]}  ({WORK_MPP} m/px)")

    # --- OSM -------------------------------------------------------------
    elements = osm.fetch(GONCHARY, cache_dir=DERIVED / "osm-cache")
    osm_mask, drawn = osm.rasterise(elements, GONCHARY, shape)
    print(f"\nOSM         {len(elements)} ways -> {drawn}")
    values, counts = np.unique(osm_mask, return_counts=True)
    names = {v: k for k, v in osm.CLASSES.items()}
    for value, count in zip(values, counts):
        print(f"  {names[value]:<11} {100*count/osm_mask.size:6.2f}% of pixels")

    # --- Delineate Anything ----------------------------------------------
    valid = da.valid_mask(work)
    print(f"\nvalid imagery  {100*valid.mean():.1f}% of the orthophoto "
          f"(the rest is black padding from a rotated export)")
    for name in ("building", "road"):
        sel = osm_mask == osm.CLASSES[name]
        if sel.any():
            print(f"  OSM {name:<9} {100*valid[sel].mean():5.1f}% of its pixels "
                  f"fall on real imagery")

    started = time.time()
    model = da.load("v2")
    polygons = da.predict(model, work, conf=CONF, device="cpu")
    da_edges = da.rasterise_edges(polygons, shape, width_px=2)
    print(f"\nDelineate Anything v2  {len(polygons)} field polygons in "
          f"{time.time()-started:.1f}s  ({100*da_edges.mean():.2f}% of pixels are edge)")

    # --- alignment --------------------------------------------------------
    grey = np.array(work.convert("L"), dtype=np.float32)
    image_edges = align.gradient_magnitude(grey)
    building_edges = align.outline(osm_mask == osm.CLASSES["building"])
    result = align.offset(image_edges, building_edges, WORK_MPP, max_shift_m=15.0)
    print(f"\nOSM buildings vs imagery")
    print(f"  offset        dx {result['dx_m']:+.2f} m  dy {result['dy_m']:+.2f} m"
          f"  -> {result['distance_m']:.2f} m")
    verdict = ("trustworthy" if result["peak_z"] > 4 and not result["at_search_limit"]
               else "WEAK — no shared structure, the offset is meaningless")
    print(f"  peak z        {result['peak_z']:.2f}   ({verdict})")
    print(f"  zero-shift z  {result['zero_shift_z']:.2f}"
          f"   at search limit: {result['at_search_limit']}")

    (HERE / "results.json").write_text(json.dumps(
        {"n_osm_ways": len(elements), "osm_drawn": drawn, "n_da_polygons": len(polygons),
         "work_mpp": WORK_MPP, "conf": CONF, "alignment": result}, indent=2))

    # --- figure 1: the four-panel overlay ---------------------------------
    rgb = np.array(work.convert("RGB"))
    panels = [("Google Earth imagery", None),
              ("+ OpenStreetMap", "osm"),
              ("+ Delineate Anything", "da"),
              ("+ both", "both")]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for ax, (title, layer) in zip(axes.ravel(), panels):
        ax.imshow(rgb)
        if layer in ("osm", "both"):
            for name, colour in (("building", (1, 0.25, 0.25)), ("road", (1, 0.85, 0.1)),
                                 ("water", (0.2, 0.6, 1)), ("wood", (0.2, 0.9, 0.4))):
                sel = osm_mask == osm.CLASSES[name]
                if sel.any():
                    overlay = np.zeros((*shape, 4))
                    overlay[sel] = (*colour, 0.85)
                    ax.imshow(overlay)
        if layer in ("da", "both"):
            overlay = np.zeros((*shape, 4))
            overlay[da_edges > 0] = (0.1, 1.0, 1.0, 0.95)
            ax.imshow(overlay)
        ax.set_title(title, fontsize=13)
        ax.axis("off")
    handles = [Patch(color=(1, 0.25, 0.25), label=f"OSM building ({drawn.get('building',0)})"),
               Patch(color=(1, 0.85, 0.1), label=f"OSM road ({drawn.get('road',0)})"),
               Patch(color=(0.1, 1, 1), label=f"DA field boundary ({len(polygons)})")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=12)
    fig.suptitle(f"Map-derived labels over Gonchary  ({WORK_MPP} m/px, "
                 f"{GONCHARY.width_m():.0f} x {GONCHARY.height_m():.0f} m)", fontsize=15)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(FIGURES / "gonchary-overlay.png", dpi=110)
    plt.close(fig)

    # --- figure 2: alignment, zoomed on the buildings ---------------------
    ys, xs = np.nonzero(building_edges)
    if len(xs):
        pad = 120
        y0, y1 = max(0, ys.min() - pad), min(shape[0], ys.max() + pad)
        x0, x1 = max(0, xs.min() - pad), min(shape[1], xs.max() + pad)
        dx, dy = result["dx_px"], result["dy_px"]
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        for ax, (title, shift) in zip(axes, [("OSM as published", (0, 0)),
                                             (f"shifted by best fit "
                                              f"({result['dx_m']:+.1f}, {result['dy_m']:+.1f}) m",
                                              (dx, dy))]):
            ax.imshow(rgb[y0:y1, x0:x1])
            moved = np.roll(np.roll(building_edges, shift[1], axis=0), shift[0], axis=1)
            overlay = np.zeros((y1 - y0, x1 - x0, 4))
            overlay[moved[y0:y1, x0:x1] > 0] = (1, 0.2, 0.2, 1)
            ax.imshow(overlay)
            ax.set_title(title, fontsize=13)
            ax.axis("off")
        fig.suptitle(f"OSM building footprints vs imagery — "
                     f"disagreement {result['distance_m']:.2f} m "
                     f"(peak z {result['peak_z']:.2f})", fontsize=15)
        fig.tight_layout()
        fig.savefig(FIGURES / "gonchary-osm-offset.png", dpi=110)
        plt.close(fig)

    # --- figure 3: boundaries alone ---------------------------------------
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(rgb)
    overlay = np.zeros((*shape, 4))
    overlay[da_edges > 0] = (0.1, 1, 1, 0.95)
    ax.imshow(overlay)
    ax.set_title(f"Delineate Anything v2 — {len(polygons)} field polygons "
                 f"at {WORK_MPP} m/px, conf {CONF}", fontsize=14)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIGURES / "gonchary-boundaries.png", dpi=110)
    plt.close(fig)

    np.save(DERIVED / "boundaries" / "gonchary-edges.npy", da_edges)
    np.save(DERIVED / "boundaries" / "gonchary-osm.npy", osm_mask)
    print(f"\nfigures -> {FIGURES}")


if __name__ == "__main__":
    main()
