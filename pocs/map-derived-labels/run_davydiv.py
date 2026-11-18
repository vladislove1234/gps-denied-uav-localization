"""OSM and Delineate Anything over the Davydiv Google Earth pair.

    python pocs/map-derived-labels/run_davydiv.py

The georeference comes from four placemarks located by differencing the marked
and unmarked exports. The coordinates are exact as stated -- `satelites/points.kml`
confirms Google Earth wrote full precision and the pins really sit on a
round-number grid -- and the pixel positions are good to ~1 px, verified by
cross-correlating the four icon patches.

Even so, **four pins are not a georeference**. They leave a 3.4 m residual that
no affine removes, and refitting with any one pin dropped moves the transform by
up to 13.9 m across the frame. Do not try to fix this by picking the tips more
carefully or by hunting for more decimal places; both were tried and neither
moves the number (bug-011, and the post-mortem at the end of notes.md).

What the transform is actually good for is checked against OSM itself, tiled
across the frame: no detectable drift, mean offset +0.45/+1.00 m over the tiles
with a strong correlation peak. The figure to quote for label noise is the
3-4 m tile-to-tile scatter, not the single global number below.
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
from geo import Affine2D          # noqa: E402

SRC = HERE.parents[1]
CLEAN = HERE / "satelites/davydiv_no_marks.png"
DERIVED = SRC / "data/derived"
FIGURES = HERE / "figures"
CONF = 0.10

# (lat, lon, x, y) -- pixel positions are the pushpin needle tips, found by
# differencing the marked and unmarked exports.
CONTROL = [
    (49.753400, 24.136800, 243.50, 501.0),    # lt
    (49.753300, 24.147900, 3291.00, 572.0),   # rt
    (49.749300, 24.147000, 3005.50, 2240.0),  # rb
    (49.749300, 24.136800, 238.00, 2248.0),   # lb
]
COORD_DECIMALS = 4


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    FIGURES.mkdir(exist_ok=True)
    image = Image.open(CLEAN).convert("RGB")
    rgb = np.array(image)
    shape = rgb.shape[:2]

    projection = Affine2D.fit(CONTROL)
    residuals = projection.residuals(CONTROL)
    mpp_x, mpp_y = projection.mpp()
    quantisation_m = 0.5 * 10 ** -COORD_DECIMALS * 110_574
    print(f"georeference from {len(CONTROL)} placemarks")
    print(f"  residuals    {residuals.round(2)} px   max {residuals.max()*mpp_x:.2f} m")
    print(f"  scale        {mpp_x:.4f} / {mpp_y:.4f} m/px")
    print(f"  coordinates rounded to {COORD_DECIMALS} dp -> +/-{quantisation_m:.1f} m "
          f"of latitude quantisation alone")

    bounds = projection.image_bounds(shape)
    elements = osm.fetch(bounds, cache_dir=DERIVED / "osm-cache")
    osm_mask, drawn = osm.rasterise(elements, projection, shape)
    print(f"\nOSM          {len(elements)} ways -> {drawn}")
    values, counts = np.unique(osm_mask, return_counts=True)
    names = {v: k for k, v in osm.CLASSES.items()}
    for value, count in zip(values, counts):
        print(f"  {names[value]:<11} {100*count/osm_mask.size:6.2f}% of pixels")

    started = time.time()
    model = da.load("v2")
    polygons = da.predict(model, image, conf=CONF, device="cpu")
    da_edges = da.rasterise_edges(polygons, shape, width_px=2)
    print(f"\nDelineate Anything v2  {len(polygons)} polygons in {time.time()-started:.1f}s")

    grey = np.array(image.convert("L"), dtype=np.float32)
    image_edges = align.gradient_magnitude(grey)
    building_edges = align.outline(osm_mask == osm.CLASSES["building"])
    result = align.offset(image_edges, building_edges, mpp_x, max_shift_m=15.0)
    trusted = result["peak_z"] > 4 and not result["at_search_limit"]
    print(f"\nOSM buildings vs imagery")
    print(f"  offset       dx {result['dx_m']:+.2f} m  dy {result['dy_m']:+.2f} m"
          f"  -> {result['distance_m']:.2f} m")
    print(f"  peak z       {result['peak_z']:.2f}"
          f"   ({'real peak' if trusted else 'WEAK — no shared structure'})")
    print(f"  NOTE: georeference itself is uncertain to ~{residuals.max()*mpp_x:.1f} m; "
          f"this is an upper bound on OSM's own error")

    (HERE / "results-davydiv.json").write_text(json.dumps(
        {"n_osm_ways": len(elements), "osm_drawn": drawn, "n_da_polygons": len(polygons),
         "mpp": [mpp_x, mpp_y], "georef_residual_px": residuals.tolist(),
         "coord_decimals": COORD_DECIMALS, "alignment": result}, indent=2))

    panels = [("Google Earth imagery", None), ("+ OpenStreetMap", "osm"),
              ("+ Delineate Anything", "da"), ("+ both", "both")]
    fig, axes = plt.subplots(2, 2, figsize=(19, 14))
    for ax, (title, layer) in zip(axes.ravel(), panels):
        ax.imshow(rgb)
        if layer in ("osm", "both"):
            for name, colour in (("building", (1, 0.2, 0.2)), ("road", (1, 0.85, 0.1)),
                                 ("water", (0.2, 0.6, 1)), ("wood", (0.25, 0.95, 0.45))):
                sel = osm_mask == osm.CLASSES[name]
                if sel.any():
                    overlay = np.zeros((*shape, 4))
                    overlay[sel] = (*colour, 0.55 if name in ("wood", "water") else 0.85)
                    ax.imshow(overlay)
        if layer in ("da", "both"):
            overlay = np.zeros((*shape, 4))
            overlay[da_edges > 0] = (0.1, 1.0, 1.0, 0.9)
            ax.imshow(overlay)
        ax.set_title(title, fontsize=14)
        ax.axis("off")
    handles = [Patch(color=(1, 0.2, 0.2), label=f"OSM building ({drawn.get('building',0)})"),
               Patch(color=(1, 0.85, 0.1), label=f"OSM road ({drawn.get('road',0)})"),
               Patch(color=(0.25, 0.95, 0.45), label=f"OSM wood ({drawn.get('wood',0)})"),
               Patch(color=(0.2, 0.6, 1), label=f"OSM water ({drawn.get('water',0)})"),
               Patch(color=(0.1, 1, 1), label=f"DA field boundary ({len(polygons)})")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=12)
    fig.suptitle(f"Davydiv — map-derived labels  ({mpp_x:.3f} m/px)   "
                 f"georeference uncertain to ~{residuals.max()*mpp_x:.1f} m "
                 f"(coordinates rounded to {COORD_DECIMALS} dp)", fontsize=15)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(FIGURES / "davydiv-overlay.png", dpi=95)
    plt.close(fig)

    # zoom on the densest building cluster -- where alignment is judgeable by eye
    ys, xs = np.nonzero(building_edges)
    if len(xs):
        cy, cx = int(np.median(ys)), int(np.median(xs))
        half = 420
        y0, y1 = max(0, cy - half), min(shape[0], cy + half)
        x0, x1 = max(0, cx - half), min(shape[1], cx + half)
        fig, axes = plt.subplots(1, 2, figsize=(17, 8))
        for ax, (title, on) in zip(axes, [("imagery", False), ("+ OSM buildings & roads", True)]):
            ax.imshow(rgb[y0:y1, x0:x1])
            if on:
                for name, colour in (("building", (1, 0.2, 0.2)), ("road", (1, 0.85, 0.1))):
                    sel = (osm_mask == osm.CLASSES[name])[y0:y1, x0:x1]
                    if sel.any():
                        ov = np.zeros((y1 - y0, x1 - x0, 4))
                        ov[sel] = (*colour, 0.5)
                        ax.imshow(ov)
            ax.set_title(title, fontsize=13)
            ax.axis("off")
        fig.suptitle("Davydiv — do OSM footprints sit on the buildings?", fontsize=15)
        fig.tight_layout()
        fig.savefig(FIGURES / "davydiv-osm-zoom.png", dpi=110)
        plt.close(fig)
    print(f"\nfigures -> {FIGURES}")


if __name__ == "__main__":
    main()
