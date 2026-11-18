"""Build `ua-village-multidate`: the same mapped ground, many dates, one label.

    python datasets/ua-village-multidate/build.py --villages 50

Why this shape. The encoder's job is to make a drone frame and a satellite tile
of the same place look alike despite season, sun angle, sensor and years of
staleness. Supervising that needs pairs of *the same ground looking different*,
co-registered well enough that a per-pixel loss means something. Esri Wayback
gives exactly that for free: archived World Imagery releases under the same XYZ
scheme, so two dates of a tile share a closed-form georeference with nothing
fitted (see the post-mortem in pocs/map-derived-labels/notes.md for what the
Google Earth route cost instead).

One sample is a *location*: one OSM label mask plus every distinct date of
imagery over it. Crops are a dataloader concern, not a build concern -- cutting
them here would bake in a window size and multiply the bytes for no reason.

Two things this build does that are not obvious:

* **Villages, not the flight area.** 0 of 4,181 positions in the 2025 flight have
  a mapped feature in frame. Training ground and flight ground are decoupled on
  purpose; the invariance being learned is not a property of one village.
* **Dates are co-registered to the newest.** Esri's older releases over rural
  Ukraine sit up to 3.1 m east of the current one. Measured per village by phase
  correlation and corrected, because "same tile coords" turned out not to mean
  "same pixels".
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import threading
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

SRC = Path(__file__).resolve().parents[2]
# The tile/OSM/alignment code still lives in the PoC that produced it. Moving it
# into uavloc/ is a maintenance commitment and therefore the user's call, not
# this script's.
sys.path.insert(0, str(SRC / "pocs/map-derived-labels"))
sys.path.insert(0, str(SRC))

import align        # noqa: E402
import areas        # noqa: E402
import osm          # noqa: E402
import tiles        # noqa: E402
import wayback      # noqa: E402

from uavloc.data.google_maps import create_session, load_api_key, satellite_tile_template   # noqa: E402
from uavloc.datasets import data_root, write_manifest   # noqa: E402
from uavloc.matching.registration import apply_shift, phase_shift   # noqa: E402

NAME = "ua-village-multidate"
VERSION = "2"   # 2: adds a co-registered Google satellite look, `google.jpg`
# z18, not z19. Wayback's archive over rural Ukraine stops at z17-z18 even where
# *current* World Imagery serves z19: screened over 150 villages, 100% have at
# least 2 z18 releases and 97% have 4, against 2% at z19. Buying 0.193 m/px
# would cost the multi-date signal that is the entire point of this dataset.
# 0.386 m/px still resolves a 10 m house as 26 px and a 6 m lane as 16 px.
#
# Tiles are 4x cheaper at z18, so each village covers 1200 x 900 m instead of
# 840 x 640 -- more ground, more field/village mix, same request budget.
WIDTH_M, HEIGHT_M = 1200.0, 900.0
# Measured, not guessed: the median village has 6.06% of pixels labelled by OSM
# and 2,994 building-outline pixels. A 4,000 px floor rejected 11 of 18 villages
# that were perfectly usable.
MIN_LABELLED = 0.025
MIN_BUILDING_PX = 1500
MAX_DATES = 8
PROBE_EVERY = 5
# Two dates correlating above this are the same capture re-encoded, not two
# looks at the ground. Measured: byte-level dedup left a median correlation of
# 0.9988 between "distinct" dates, against 0.34 for a genuinely different one.
APPEARANCE_MAX_CORR = 0.97
MIN_SHIFT_PEAK = 15.0        # phase-correlation peak ratio below this is noise
MAX_SHIFT_PX = 30.0          # ~11.6 m at z18; larger is a spurious peak
# Looser on the 2x2 probe than on the full mosaic: a small patch correlates
# higher by chance, and a false reject here costs a date we can never recover,
# while a false keep only costs 130 tiles and is caught downstream.
PROBE_MAX_CORR = 0.995
MIN_DISTINCT_LOOKS = 2
WORKERS = 6              # sample 1 Wayback release in 4 when deduplicating
TILE_CACHE = SRC / "data/derived/tile-cache"
MIN_COVERAGE = 0.995    # a hole in the mosaic poisons a crop
# Google gives no capture date, so its look is stored under a name, not a date.
# It sorts after every ISO date and before "current" in the annotator.
GOOGLE_DATE = "google"
GOOGLE_CACHE_KEY = "google-satellite"   # the tile-cache folder pocs/google-satellite filled
OSM_CACHE = SRC / "data/derived/osm-cache"


# NFKD + ascii-ignore deletes Cyrillic outright, so every folder came out named
# "place-". Transliteration table is the Ukrainian national standard (KMU 55),
# simplified: folder names only need to be readable and unique.
_TRANSLIT = {
    "\u0430": "a", "\u0431": "b", "\u0432": "v", "\u0433": "h", "\u0491": "g",
    "\u0434": "d", "\u0435": "e", "\u0454": "ie", "\u0436": "zh", "\u0437": "z",
    "\u0438": "y", "\u0456": "i", "\u0457": "i", "\u0439": "i", "\u043a": "k",
    "\u043b": "l", "\u043c": "m", "\u043d": "n", "\u043e": "o", "\u043f": "p",
    "\u0440": "r", "\u0441": "s", "\u0442": "t", "\u0443": "u", "\u0444": "f",
    "\u0445": "kh", "\u0446": "ts", "\u0447": "ch", "\u0448": "sh", "\u0449": "shch",
    "\u044c": "", "\u044e": "iu", "\u044f": "ia", "\u0027": "",
}


def slug(name: str, lat: float, lon: float) -> str:
    lowered = (name or "").lower()
    latin = "".join(_TRANSLIT.get(ch, ch) for ch in lowered)
    latin = unicodedata.normalize("NFKD", latin).encode("ascii", "ignore").decode()
    latin = re.sub(r"[^A-Za-z0-9]+", "-", latin).strip("-").lower()
    return f"{latin or 'place'}-{lat:.4f}-{lon:.4f}".replace(".", "p")


def build_one(place: dict, out_dir: Path, zoom: int) -> dict | None:
    """One village -> one folder, or ``None`` with a reason if it does not qualify."""
    grid = tiles.TileGrid.covering(place["lat"], place["lon"], WIDTH_M, HEIGHT_M, zoom)
    mpp, _ = grid.mpp()
    shape = grid.shape

    # OSM first: it is one cheap query, and it rejects most ground before we
    # spend a thousand tile requests on imagery we would then throw away.
    elements = osm.fetch(grid.image_bounds(), cache_dir=OSM_CACHE)
    mask, drawn = osm.rasterise(elements, grid, shape)
    labelled = float((mask > 0).mean())
    building = align.outline((mask == osm.CLASSES["building"]).astype(np.float32))
    building_px = int(building.sum())
    if labelled < MIN_LABELLED:
        return {"skipped": "too little OSM", "labelled": labelled}
    if building_px < MIN_BUILDING_PX:
        return {"skipped": "too few building edges", "building_px": building_px}

    distinct = wayback.distinct_releases(grid, TILE_CACHE, every=PROBE_EVERY, probes=2)
    distinct = distinct[-MAX_DATES:]

    # Decide which dates are worth a full mosaic using a 2x2-tile probe at the
    # centre. Byte-level dedup keeps ~5 "dates" per village of which ~2 are
    # visually distinct, and a full mosaic is 130 tiles against the probe's 4 --
    # so screening here is the difference between fetching 900 tiles per village
    # and fetching 300. The same correlation test runs again on the full mosaics;
    # this only avoids paying for the obvious duplicates.
    probe = tiles.TileGrid(grid.z, grid.x0 + grid.nx // 2 - 1,
                           grid.y0 + grid.ny // 2 - 1, 2, 2)
    looks, kept_sources = [], []
    for date, source in sorted(list(distinct) + [("current", tiles.ESRI_CURRENT)],
                               reverse=True):
        patch = tiles.mosaic(source, probe, TILE_CACHE)
        if (patch.any(axis=2)).mean() < 0.99:
            continue
        g = patch.astype(np.float32).mean(axis=2)
        if g.std() < 1e-6:
            continue
        if any(float(np.corrcoef(g.ravel(), o.ravel())[0, 1]) >= PROBE_MAX_CORR
               for o in looks):
            continue
        looks.append(g)
        kept_sources.append((date, source))
    if len(kept_sources) < MIN_DISTINCT_LOOKS:
        return {"skipped": "too few distinct looks on probe", "looks": len(kept_sources)}

    # "current" is always a candidate above: it is the only layer with high-zoom
    # coverage everywhere, so without it a village with a thin archive yields
    # nothing.
    images = {}
    for date, source in kept_sources:
        rgb = tiles.mosaic(source, grid, TILE_CACHE)
        if (rgb.any(axis=2)).mean() < MIN_COVERAGE:
            continue
        images[date] = rgb
    if len(images) < 2:
        return {"skipped": "fewer than 2 complete mosaics", "dates": len(images)}

    reference_date = max(images)
    reference = images[reference_date].astype(np.float32).mean(axis=2)

    # Co-register everything to the newest before judging anything else: a
    # misregistered copy correlates poorly and would survive the appearance
    # filter below by looking "different" for the wrong reason.
    shifts, greys, dropped_shift = {}, {}, []
    for date, rgb in list(images.items()):
        if date == reference_date:
            dx = dy = 0.0
            ratio = None
        else:
            dx, dy, ratio = phase_shift(reference, rgb.astype(np.float32).mean(axis=2))
            # A weak peak means the correlation found texture, not the ground.
            # Measured over the first 61-village build: the median peak ratio on
            # a real shift is 52.9, and the only two nonsense shifts (16.8 m and
            # 15.7 m, both near the 48 px search limit) scored 11.7 and 7.7. An
            # unreliable shift is worse than none -- it silently moves imagery
            # away from its label -- so the date is dropped, not un-shifted.
            if ratio < MIN_SHIFT_PEAK or np.hypot(dx, dy) > MAX_SHIFT_PX:
                dropped_shift.append({"date": date, "peak_ratio": ratio,
                                      "shift_px": [dx, dy]})
                continue
            images[date] = rgb = apply_shift(rgb, dx, dy)
        shifts[date] = (dx, dy, ratio)
        greys[date] = rgb.astype(np.float32).mean(axis=2)

    # Deduplicate by *appearance*, not by tile bytes. Wayback re-serves the same
    # capture across releases with different JPEG bytes, so hashing keeps copies
    # that correlate at 0.999 -- five "dates" that are really two pictures. The
    # whole dataset is a bet on appearance variation, so near-duplicates are not
    # padding, they are a silent failure of the premise.
    for date in dropped_shift:
        images.pop(date["date"], None)
    kept = [reference_date]
    for date in sorted(greys, reverse=True):
        if date == reference_date:
            continue
        if max(float(np.corrcoef(greys[date].ravel(), greys[k].ravel())[0, 1])
               for k in kept) < APPEARANCE_MAX_CORR:
            kept.append(date)
    if len(kept) < MIN_DISTINCT_LOOKS:
        return {"skipped": "too few visually distinct dates", "looks": len(kept),
                "fetched": len(images)}

    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(out_dir / "label.png", optimize=True)

    per_date = {}
    for date in sorted(kept):
        rgb, grey = images[date], greys[date]
        dx, dy, ratio = shifts[date]
        Image.fromarray(rgb).save(out_dir / f"{date}.jpg", quality=92, optimize=True)
        per_date[date] = {
            "shift_px": [dx, dy], "shift_m": float(np.hypot(dx, dy) * mpp),
            "peak_ratio": ratio,
            "mean": float(grey.mean()), "std": float(grey.std()),
            "corr_vs_reference": float(np.corrcoef(reference.ravel(), grey.ravel())[0, 1]),
        }
    meta_extra = {"dates_fetched": len(images), "dates_kept": len(kept)}

    # How well the labels sit on the imagery, on this ground. Recorded per
    # village so a bad village can be excluded later instead of poisoning a run.
    edges = align.gradient_magnitude(reference)
    offset = align.offset(edges, building, mpp, max_shift_m=15.0)

    meta = {
        "name": place["name"], "lat": place["lat"], "lon": place["lon"],
        "grid": {"z": grid.z, "x0": grid.x0, "y0": grid.y0, "nx": grid.nx, "ny": grid.ny},
        "shape": list(shape), "mpp": mpp,
        "bounds": grid.bounds().__dict__,
        "reference_date": reference_date,
        "osm": {"ways": len(elements), "drawn": drawn,
                "labelled_frac": labelled, "building_px": building_px},
        "label_offset": offset,
        "dates": per_date, **meta_extra, "dropped_shift": dropped_shift,
        "inset_px": int(np.ceil(max(
            [max(abs(v["shift_px"][0]), abs(v["shift_px"][1])) for v in per_date.values()]
            or [0]))) + 2,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def add_google(root: Path, key: str, source: tiles.TileSource, fetched: str) -> dict:
    """Add Google as one more look of a built location, under the same rules as an Esri date.

    Co-registered to the location's reference by the same phase correlation, rejected on
    the same weak-peak and too-far limits, and dropped if it duplicates a stored date.
    The outcome is recorded in ``meta.json`` either way, so a rerun does not redo it.
    """
    meta_path = root / key / "meta.json"
    meta = json.loads(meta_path.read_text())
    if GOOGLE_DATE in meta["dates"] or "google_skipped" in meta:
        return meta
    rgb = tiles.mosaic(source, tiles.TileGrid(**meta["grid"]), TILE_CACHE)
    coverage = float(rgb.any(axis=2).mean())
    reference = np.asarray(Image.open(root / key / f"{meta['reference_date']}.jpg"),
                           dtype=np.float32).mean(axis=2)
    dx, dy, ratio = phase_shift(reference, rgb.astype(np.float32).mean(axis=2))
    if coverage < MIN_COVERAGE:
        meta["google_skipped"] = {"reason": "incomplete mosaic", "coverage": coverage}
    elif ratio < MIN_SHIFT_PEAK or np.hypot(dx, dy) > MAX_SHIFT_PX:
        meta["google_skipped"] = {"reason": "unreliable shift", "peak_ratio": ratio, "shift_px": [dx, dy]}
    else:
        rgb = apply_shift(rgb, dx, dy)
        grey = rgb.astype(np.float32).mean(axis=2)
        corr = {date: float(np.corrcoef(grey.ravel(), np.asarray(
                    Image.open(root / key / f"{date}.jpg"), dtype=np.float32).mean(axis=2).ravel())[0, 1])
                for date in meta["dates"]}
        if max(corr.values()) >= APPEARANCE_MAX_CORR:
            meta["google_skipped"] = {"reason": "duplicates a stored date", "corr": corr}
        else:
            Image.fromarray(rgb).save(root / key / f"{GOOGLE_DATE}.jpg", quality=92, optimize=True)
            meta["dates"][GOOGLE_DATE] = {
                "shift_px": [dx, dy], "shift_m": float(np.hypot(dx, dy) * meta["mpp"]),
                "peak_ratio": ratio, "mean": float(grey.mean()), "std": float(grey.std()),
                "corr_vs_reference": corr[meta["reference_date"]], "max_corr_vs_stored": max(corr.values()),
                "source": "Google Map Tiles API, satellite", "fetched": fetched, "capture_date": None,
            }
            meta["inset_px"] = max(meta["inset_px"], int(np.ceil(max(abs(dx), abs(dy)))) + 2)
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


def add_google_all(root: Path) -> None:
    """``add_google`` over every built location, one Map Tiles session for the run."""
    api_key = load_api_key()
    source = tiles.TileSource(GOOGLE_CACHE_KEY, satellite_tile_template(api_key, create_session(api_key)))
    fetched = time.strftime("%Y-%m-%d")
    built = json.loads((root / "index.json").read_text())["built"]
    for i, key in enumerate(built):
        meta = add_google(root, key, source, fetched)
        google = meta["dates"].get(GOOGLE_DATE)
        outcome = (f"ok    shift {google['shift_m']:.1f} m, peak {google['peak_ratio']:.0f}, "
                   f"max corr {google['max_corr_vs_stored']:.2f}") if google \
            else f"skip  {meta['google_skipped']['reason']}"
        print(f"[{i + 1:2d}/{len(built)}] {key:44s} {outcome}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--villages", type=int, default=50)
    parser.add_argument("--zoom", type=int, default=18,
                        help="Web Mercator zoom. Wayback's archive over rural "
                             "Ukraine caps below current World Imagery, so this "
                             "trades resolution against how many dates exist.")
    parser.add_argument("--separation-km", type=float, default=12.0)
    parser.add_argument("--attempts", type=int, default=0,
                        help="how many candidates to try (default: 3x villages)")
    parser.add_argument("--add-google", action="store_true",
                        help="add google.jpg to every already-built location, then "
                             "rewrite the manifest; builds nothing new")
    args = parser.parse_args()
    if args.add_google:
        root = data_root(NAME)
        add_google_all(root)
        summarise(root, json.loads((root / "index.json").read_text())["built"], args.zoom)
        return
    attempts = args.attempts or args.villages * 3

    root = data_root(NAME, create=True)
    candidates = areas.spread(areas.villages(), attempts, args.separation_km)
    print(f"{len(candidates)} candidate villages, want {args.villages}", flush=True)

    built, skipped = [], []
    started = time.time()
    lock = threading.Lock()

    def attempt(item):
        i, place = item
        key = slug(place["name"], place["lat"], place["lon"])
        out_dir = root / key
        if (out_dir / "meta.json").exists():
            return i, key, json.loads((out_dir / "meta.json").read_text())
        try:
            return i, key, build_one(place, out_dir, args.zoom)
        except Exception as exc:                      # noqa: BLE001
            return i, key, {"skipped": f"{type(exc).__name__}: {exc}"}

    # Nearly all the wall clock is waiting on Overpass and on Esri, so this is
    # I/O-bound and threads help. Kept modest on purpose: each worker spawns its
    # own tile pool underneath, so WORKERS x 8 requests can be in flight.
    with ThreadPoolExecutor(WORKERS) as pool:
        pending = iter(list(enumerate(candidates)))
        futures = {}
        for _ in range(WORKERS):
            item = next(pending, None)
            if item is not None:
                futures[pool.submit(attempt, item)] = item

        while futures:
            for future in as_completed(list(futures)):
                futures.pop(future, None)
                i, key, meta = future.result()
                with lock:
                    enough = len(built) >= args.villages
                    if meta is None or "skipped" in meta:
                        skipped.append({"key": key, **(meta or {})})
                        print(f"[{i:3d}] {key:44s} skip  "
                              f"{meta.get('skipped') if meta else '?'}", flush=True)
                    else:
                        built.append(key)
                        print(f"[{i:3d}] {key:44s} ok    {len(meta['dates'])} dates  "
                              f"OSM {100*meta['osm']['labelled_frac']:4.1f}%  "
                              f"label offset {meta['label_offset']['distance_m']:4.1f} m "
                              f"(z {meta['label_offset']['peak_z']:.1f})  "
                              f"[{len(built)}/{args.villages}, "
                              f"{(time.time()-started)/60:.0f} min]", flush=True)
                        enough = len(built) >= args.villages
                    if not enough:
                        item = next(pending, None)
                        if item is not None:
                            futures[pool.submit(attempt, item)] = item
                break

    (root / "index.json").write_text(json.dumps(
        {"built": built, "skipped": skipped,
         "config": {"zoom": args.zoom, "width_m": WIDTH_M, "height_m": HEIGHT_M,
                    "min_labelled": MIN_LABELLED, "min_building_px": MIN_BUILDING_PX,
                    "max_dates": MAX_DATES, "probe_every": PROBE_EVERY,
                    "separation_km": args.separation_km}}, indent=2))
    print(f"\n{len(built)} built, {len(skipped)} skipped, {(time.time()-started)/60:.0f} min")
    summarise(root, built, args.zoom)


def summarise(root: Path, built: list[str], zoom: int) -> None:
    """Acceptance evidence. A dataset has no bar, but it must carry its own numbers."""
    metas = [json.loads((root / k / "meta.json").read_text()) for k in built]
    if not metas:
        print("nothing built -- no manifest written")
        return
    dates = [len(m["dates"]) for m in metas]
    labelled = [m["osm"]["labelled_frac"] for m in metas]
    offsets = [m["label_offset"]["distance_m"] for m in metas]
    peaks = [m["label_offset"]["peak_z"] for m in metas]
    strong = [o for o, z in zip(offsets, peaks) if z >= 4.5]
    # Esri and Google are reported apart, so the Esri checks stay comparable with version 1.
    esri = [d for m in metas for name, d in m["dates"].items() if name != GOOGLE_DATE]
    google = [m["dates"][GOOGLE_DATE] for m in metas if GOOGLE_DATE in m["dates"]]
    shifts = [d["shift_m"] for d in esri]
    corrs = [d["corr_vs_reference"] for d in esri if d["peak_ratio"] is not None]
    pairs = sum(dates)

    checks = {
        "villages": len(metas),
        "image_label_pairs": pairs,
        "dates_per_village_median": float(np.median(dates)),
        "dates_per_village_min": int(min(dates)),
        "labelled_frac_median": float(np.median(labelled)),
        "label_offset_m_median": float(np.median(offsets)),
        "label_offset_m_median_strong_peak": float(np.median(strong)) if strong else None,
        "villages_with_strong_peak": len(strong),
        "coregistration_shift_m_median": float(np.median(shifts)),
        "coregistration_shift_m_max": float(max(shifts)),
        "appearance_corr_median": float(np.median(corrs)) if corrs else None,
        "appearance_corr_min": float(min(corrs)) if corrs else None,
        "google_villages": len(google),
        "google_skipped": dict(Counter(m["google_skipped"]["reason"] for m in metas if "google_skipped" in m)),
        "google_shift_m_median": float(np.median([d["shift_m"] for d in google])) if google else None,
        "google_shift_m_max": float(max(d["shift_m"] for d in google)) if google else None,
        "google_max_corr_vs_stored_max": float(max(d["max_corr_vs_stored"] for d in google)) if google else None,
    }
    print("\nacceptance evidence")
    for k, v in checks.items():
        print(f"  {k:38s} {v}")

    write_manifest(
        NAME, version=VERSION,
        description=("Ukrainian village ground at 0.386 m/px, every distinct Esri "
                     "World Imagery date plus a Google satellite look, co-registered "
                     "to the newest Esri, with one OSM class mask per location."),
        source=("Esri World Imagery + Wayback (XYZ z18); Google Map Tiles API satellite "
                "(z18, no capture date); OpenStreetMap via Overpass"),
        config={"zoom": zoom, "width_m": WIDTH_M, "height_m": HEIGHT_M,
                "classes": osm.CLASSES, "road_width_m": osm.ROAD_WIDTH_M,
                "min_labelled": MIN_LABELLED, "min_building_px": MIN_BUILDING_PX,
                "max_dates": MAX_DATES, "region": areas.REGION,
                "google": {"date_name": GOOGLE_DATE, "zoom": zoom, "min_coverage": MIN_COVERAGE,
                           "min_shift_peak": MIN_SHIFT_PEAK, "max_shift_px": MAX_SHIFT_PX,
                           "appearance_max_corr": APPEARANCE_MAX_CORR}},
        samples=pairs, checks=checks)
    print(f"\nwrote datasets/{NAME}/manifest.json")


if __name__ == "__main__":
    main()
