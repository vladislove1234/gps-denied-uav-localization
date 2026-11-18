"""Figures for probe 1 -- what Delineate Anything actually drew, for eyeballing.

Three panels per location, because the numbers alone cannot separate the two
ways this probe can fail:

* imagery -- what is on the ground
* DA outlines over imagery -- did it trace things that are really there
* agreement -- DA fill against OSM farmland, as hit / DA-only / OSM-only

The third panel is the honest one. OSM farmland is a *yardstick, not truth*:
it is sparse, traced against a different basemap, and carries the per-location
offset recorded in the dataset card. DA-only is therefore not automatically an
error, and the panel is labelled to say so.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "map-derived-labels"))

from field_da import POLY_DIR, burn, load_location, osm_masks  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures"
VIEW_W = 760          # panels are for looking at, not for measuring


def load_polygons(name: str) -> list[np.ndarray]:
    data = np.load(POLY_DIR / f"{name}.npz")
    return [data[k] for k in data.files]


def shrink(array: np.ndarray, width: int = VIEW_W) -> np.ndarray:
    image = Image.fromarray(array)
    height = round(array.shape[0] * width / array.shape[1])
    return np.array(image.resize((width, height), Image.NEAREST if array.ndim == 2 else Image.LANCZOS))


def agreement_rgb(field: np.ndarray, farm: np.ndarray) -> np.ndarray:
    """Green = both, red = DA only, blue = OSM only, black = neither."""
    rgb = np.zeros(field.shape + (3,), np.uint8)
    both = (field > 0) & (farm > 0)
    da_only = (field > 0) & (farm == 0)
    osm_only = (field == 0) & (farm > 0)
    rgb[both] = (60, 200, 90)
    rgb[da_only] = (215, 70, 60)
    rgb[osm_only] = (70, 120, 235)
    return rgb


def panel_row(axes, name: str, result: dict) -> None:
    image, grid, meta = load_location(name)
    shape = tuple(meta["shape"])
    polygons = load_polygons(name)
    farm, _ = osm_masks(grid, shape)
    field = burn(polygons, shape)
    outline = burn(polygons, shape, outline=True, width_px=3)

    photo = np.array(image)
    over = photo.copy()
    over[outline > 0] = (255, 235, 60)

    for ax, art, title in (
        (axes[0], shrink(photo), name.rsplit("-", 2)[0]),
        (axes[1], shrink(over), f"{len(polygons)} parcels"),
        (axes[2], shrink(agreement_rgb(field, farm)), "DA vs OSM farmland"),
    ):
        ax.imshow(art)
        ax.set_title(title, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

    cov = result["coverage"]
    agr = result["agreement"]
    recall = agr["recall_on_farmland"]
    axes[2].set_xlabel(
        f"DA {cov['da_field']*100:.0f}%  OSM-farm {cov['osm_farmland']*100:.0f}%"
        + (f"  recall {recall*100:.0f}%" if recall is not None else "  no farmland"),
        fontsize=7,
    )


def overview(results: dict, names: list[str], path: Path) -> None:
    fig, axes = plt.subplots(len(names), 3, figsize=(10.5, 2.6 * len(names)))
    axes = np.atleast_2d(axes)
    for row, name in enumerate(names):
        panel_row(axes[row], name, results[name])
    fig.legend(
        handles=[
            Patch(color="#3cc85a", label="DA and OSM agree"),
            Patch(color="#d7463c", label="DA only (not necessarily wrong -- OSM is sparse)"),
            Patch(color="#4678eb", label="OSM farmland only (DA missed)"),
        ],
        loc="lower center", ncol=3, fontsize=8, frameon=False,
    )
    fig.suptitle("Probe 1 -- Delineate Anything as a source for class `field`", fontsize=11)
    fig.tight_layout(rect=(0, 0.035, 1, 0.985))
    fig.savefig(path, dpi=125)
    plt.close(fig)
    print(f"wrote {path}")


def distributions(results: dict, path: Path) -> None:
    """Parcel size by how built-up the location is -- the over-segmentation test."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))

    for name, result in results.items():
        areas = np.array([s["area_m2"] for s in result["stats"]])
        if not len(areas):
            continue
        farm_cov = result["coverage"]["osm_farmland"]
        rural = farm_cov >= 0.01
        axes[0].plot(
            np.sort(areas), np.linspace(0, 1, len(areas)),
            color="#2a7f3e" if rural else "#b03a2e", alpha=0.8, lw=1.4,
            label=f"{name.rsplit('-', 2)[0]} ({farm_cov*100:.0f}% farm)",
        )

    axes[0].set_xscale("log")
    axes[0].axvline(11400, color="k", ls="--", lw=1)
    axes[0].text(11900, 0.05, "1.14 ha\n(median parcel,\ndrone-frame PoC)", fontsize=6.5)
    axes[0].set_xlabel("parcel area (m²)")
    axes[0].set_ylabel("cumulative fraction")
    axes[0].set_title("Parcel size — green = farmland present, red = built-up", fontsize=9)
    axes[0].legend(fontsize=5.5, loc="lower right")
    axes[0].grid(alpha=0.25)

    farm_cov = [r["coverage"]["osm_farmland"] * 100 for r in results.values()]
    recall = [
        (r["agreement"]["recall_on_farmland"] or 0) * 100 for r in results.values()
    ]
    leak = [(r["agreement"]["leak_onto_buildings"] or 0) * 100 for r in results.values()]
    axes[1].scatter(farm_cov, recall, s=34, color="#2a7f3e", label="recall on OSM farmland")
    axes[1].scatter(farm_cov, leak, s=34, color="#b03a2e", marker="x", label="leak onto buildings")
    axes[1].set_xlabel("OSM farmland coverage (%)")
    axes[1].set_ylabel("percent")
    axes[1].set_ylim(-3, 103)
    axes[1].set_title("Does DA cover the ground OSM calls farmland?", fontsize=9)
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=125)
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=str(Path(__file__).resolve().parent / "results-field.json"))
    args = parser.parse_args()

    results = json.loads(Path(args.results).read_text())
    OUT.mkdir(parents=True, exist_ok=True)

    order = sorted(results, key=lambda n: -results[n]["coverage"]["osm_farmland"])
    overview(results, order, OUT / "field-da-overview.png")
    distributions(results, OUT / "field-da-distributions.png")


if __name__ == "__main__":
    main()
