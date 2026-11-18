"""Does an XYZ/Wayback georeference beat a four-pushpin one? Measured over Davydiv.

Davydiv on purpose: ``pocs/map-derived-labels`` already characterised this ground
with a Google Earth export, so every number here has something to beat.

Three questions, bars fixed before running:

1. **Georeference.** OSM-to-imagery offset with nothing fitted. The Google Earth
   route gave 3-4 m with a further 3.4 m of fit residual underneath it. Bar:
   <= 2 m, and it must not need a control point.
2. **Cross-date registration.** The same ground on N dates, aligned to each
   other directly. This is the label-noise floor for the whole multi-date idea:
   if Esri's own dates drift against each other, "same tile coords" does not
   mean "same pixels". Bar: <= 2 px median.
3. **Appearance variation.** Distinct dates are worthless if they look
   identical. Bar: >= 4 distinct dates, visibly different.

    python pocs/map-derived-labels/run_davydiv_tiles.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1]
sys.path.insert(0, str(HERE))

import align                      # noqa: E402
import osm                        # noqa: E402
import tiles                      # noqa: E402
import wayback                    # noqa: E402

CACHE = SRC / "data/derived/tile-cache"
OSM_CACHE = SRC / "data/derived/osm-cache"
FIGURES = HERE / "figures"

# Same ground as the Google Earth export: centre of the Davydiv frame.
LAT, LON = 49.75135, 24.14235
WIDTH_M, HEIGHT_M = 790.0, 590.0
ZOOM = 19


def phase_correlate(a: np.ndarray, b: np.ndarray, max_shift: int = 40) -> tuple[float, float, float]:
    """Translation taking ``b`` onto ``a``, by phase correlation. Returns dx, dy, peak ratio.

    Phase correlation rather than the gradient trick in ``align.py``: these are
    two versions of the *same* product, so they share real intensity structure
    and the whitened spectrum gives a far sharper peak than binarised edges.
    """
    fa = np.fft.rfft2(a - a.mean())
    fb = np.fft.rfft2(b - b.mean())
    cross = fa * np.conj(fb)
    magnitude = np.abs(cross)
    cross = np.divide(cross, magnitude, out=np.zeros_like(cross), where=magnitude > 1e-9)
    surface = np.fft.fftshift(np.fft.irfft2(cross, s=a.shape))

    cy, cx = surface.shape[0] // 2, surface.shape[1] // 2
    window = surface[cy - max_shift:cy + max_shift + 1, cx - max_shift:cx + max_shift + 1]
    py, px = np.unravel_index(np.argmax(window), window.shape)

    def refine(centre, left, right):
        denominator = left - 2 * centre + right
        return 0.0 if denominator == 0 else 0.5 * (left - right) / denominator

    sy = refine(window[py, px], window[py - 1, px], window[py + 1, px]) if 0 < py < window.shape[0] - 1 else 0.0
    sx = refine(window[py, px], window[py, px - 1], window[py, px + 1]) if 0 < px < window.shape[1] - 1 else 0.0
    peak = window.max()
    ratio = float(peak / (np.abs(window).mean() + 1e-12))
    return float(px + sx - max_shift), float(py + sy - max_shift), ratio


def grey(rgb: np.ndarray) -> np.ndarray:
    return rgb.astype(np.float32).mean(axis=2)


def valid_fraction(rgb: np.ndarray) -> float:
    return float((rgb.any(axis=2)).mean())


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    grid = tiles.TileGrid.covering(LAT, LON, WIDTH_M, HEIGHT_M, ZOOM)
    mpp, _ = grid.mpp()
    height, width = grid.shape
    print(f"grid z{grid.z}  {grid.nx}x{grid.ny} tiles = {width}x{height} px")
    print(f"  {width*mpp:.0f} x {height*mpp:.0f} m at {mpp:.4f} m/px")
    print(f"  centre {grid.centre()[0]:.6f}, {grid.centre()[1]:.6f}")
    b = grid.bounds()
    print(f"  bounds N{b.top:.6f} W{b.left:.6f} S{b.bottom:.6f} E{b.right:.6f}")
    print("  georeference is closed-form: no control points, no fit, no residual")

    print("\n[1/4] discovering distinct Wayback releases over this ground ...", flush=True)
    started = time.time()
    distinct = wayback.distinct_releases(grid, CACHE, every=2, probes=3)
    print(f"      {len(distinct)} distinct in {time.time()-started:.0f}s: "
          f"{', '.join(d for d, _ in distinct)}")

    print("\n[2/4] fetching mosaics ...", flush=True)
    images = {}
    for date, source in distinct:
        started = time.time()
        rgb = tiles.mosaic(source, grid, CACHE)
        images[date] = rgb
        print(f"      {date}  {time.time()-started:5.1f}s  valid {100*valid_fraction(rgb):5.1f}%  "
              f"mean RGB {rgb.reshape(-1,3).mean(axis=0).round(1)}")
    images["current"] = tiles.mosaic(tiles.ESRI_CURRENT, grid, CACHE)
    print(f"      current   valid {100*valid_fraction(images['current']):5.1f}%")

    print("\n[3/4] OSM over the same box ...", flush=True)
    elements = osm.fetch(grid.image_bounds(), cache_dir=OSM_CACHE)
    mask, drawn = osm.rasterise(elements, grid, (height, width))
    print(f"      {len(elements)} ways -> {drawn}")
    labelled = float((mask > 0).mean())
    print(f"      {100*labelled:.2f}% of pixels labelled")
    building = align.outline((mask == osm.CLASSES["building"]).astype(np.float32))
    print(f"      {int(building.sum())} building-outline pixels")

    print("\n[4/4] measurements\n")
    results = {"grid": {"z": grid.z, "x0": grid.x0, "y0": grid.y0,
                        "nx": grid.nx, "ny": grid.ny},
               "mpp": mpp, "shape": [height, width],
               "osm": {"ways": len(elements), "drawn": drawn, "labelled_frac": labelled},
               "dates": [d for d, _ in distinct], "per_date": {}, "cross_date": []}

    print("  Q1 georeference: OSM building outlines vs each date's imagery")
    print(f"  {'date':>12} {'dx_m':>7} {'dy_m':>7} {'dist_m':>7} {'peak_z':>7}")
    for date in [d for d, _ in distinct] + ["current"]:
        edges = align.gradient_magnitude(grey(images[date]))
        res = align.offset(edges, building, mpp, max_shift_m=15.0)
        results["per_date"][date] = res
        print(f"  {date:>12} {res['dx_m']:+7.2f} {res['dy_m']:+7.2f} "
              f"{res['distance_m']:7.2f} {res['peak_z']:7.2f}")
    good = [r["distance_m"] for d, r in results["per_date"].items() if r["peak_z"] >= 4.0]
    if good:
        print(f"  -> {len(good)} dates with peak_z>=4: median offset "
              f"{np.median(good):.2f} m, max {max(good):.2f} m")

    print("\n  Q2 cross-date registration: each date against the newest, phase correlation")
    reference_date = distinct[-1][0]
    reference = grey(images[reference_date])
    print(f"  {'date':>12} {'dx_px':>7} {'dy_px':>7} {'shift_m':>8} {'peak':>7}")
    for date, _ in distinct:
        if date == reference_date:
            continue
        dx, dy, ratio = phase_correlate(reference, grey(images[date]))
        shift = float(np.hypot(dx, dy) * mpp)
        results["cross_date"].append({"date": date, "vs": reference_date,
                                      "dx_px": dx, "dy_px": dy,
                                      "shift_m": shift, "peak_ratio": ratio})
        print(f"  {date:>12} {dx:+7.2f} {dy:+7.2f} {shift:8.2f} {ratio:7.1f}")
    if results["cross_date"]:
        shifts = [c["shift_m"] for c in results["cross_date"]]
        pix = [np.hypot(c["dx_px"], c["dy_px"]) for c in results["cross_date"]]
        print(f"  -> median {np.median(pix):.2f} px = {np.median(shifts):.2f} m, "
              f"max {max(pix):.2f} px = {max(shifts):.2f} m")

    print("\n  Q3 appearance variation between dates")
    dates = [d for d, _ in distinct]
    stats = []
    for date in dates:
        g = grey(images[date])
        stats.append((date, g.mean(), g.std()))
        print(f"  {date:>12}  mean {g.mean():6.1f}  std {g.std():5.1f}")
    ref = grey(images[reference_date])
    print(f"\n  {'date':>12} {'corr vs newest':>15}")
    correlations = []
    for date in dates:
        if date == reference_date:
            continue
        g = grey(images[date])
        c = float(np.corrcoef(ref.ravel(), g.ravel())[0, 1])
        correlations.append(c)
        print(f"  {date:>12} {c:15.3f}")
    if correlations:
        print(f"  -> correlation range {min(correlations):.3f} - {max(correlations):.3f} "
              f"(1.0 would mean the dates are the same picture)")
    results["appearance"] = {"stats": stats, "correlations": correlations}

    (HERE / "results-davydiv-tiles.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"\nwrote {HERE/'results-davydiv.json'}")
    _figures(grid, images, dates + ["current"], mask, mpp)


def _figures(grid, images, dates, mask, mpp) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    height, width = grid.shape
    size = 700
    r0, c0 = height // 2 - size // 2, width // 2 - size // 2
    crop = (slice(r0, r0 + size), slice(c0, c0 + size))

    n = len(dates)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    for ax, date in zip(np.ravel(axes), dates):
        ax.imshow(images[date][crop])
        ax.set_title(date, fontsize=11)
        ax.axis("off")
    for ax in np.ravel(axes)[n:]:
        ax.axis("off")
    fig.suptitle(f"Same {size*mpp:.0f} m of Davydiv, identical tile coordinates — "
                 f"co-registered by construction", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES / "tiles-davydiv-dates.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    cmap = ListedColormap([(0, 0, 0, 0), (1, 0.2, 0.2, 0.55), (0.2, 0.6, 1, 0.55),
                           (0, 0.9, 1, 0.6), (0.2, 0.9, 0.2, 0.45)])
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.imshow(images[dates[-1]])
    ax.imshow(mask, cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
    ax.set_title("OSM over Esri World Imagery — georeference is closed-form, nothing fitted")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIGURES / "tiles-davydiv-osm-overlay.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIGURES}/tiles-davydiv-dates.png, tiles-davydiv-osm-overlay.png")


if __name__ == "__main__":
    main()
