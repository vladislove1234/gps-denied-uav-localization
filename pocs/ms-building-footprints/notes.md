# ms-building-footprints

**Status:** worked (with one caveat) · **Question:** `Does Microsoft's Global ML Building Footprints agree with OSM over our 59 locations, add buildings OSM is missing, and independently confirm the 2.84 m label offset?`

## The question

`building` is one of only two classes measured good, and the entire quality story
rests on a single source. Microsoft's footprints are model-derived from Bing
aerial imagery — **independent of OSM in both data and method** — so they can
settle three things OSM alone cannot:

1. Does OSM miss buildings over rural Ukraine?
2. Is the recorded 2.84 m label offset a property of the *maps*, or of one map?
   Two independent sources off by the same vector is confirmation; off by
   different vectors is not.
3. Can the two be merged into a denser, better-conditioned `building` class?

Falsifiable: if MS agrees with OSM no better than the null baseline, it is not an
independent source. If the two offsets are uncorrelated, the per-location
correction in `crops.py` is unjustified.

## Verdict

**Worked as an independent source; did NOT confirm the offset hypothesis.**
Run 2026-09-23 over 9 locations, `runs/probe-2026-09-23.ipynb`.

MS agrees with OSM at a **median IoU of 0.204 against a null of 0.006**, with
**8/9 locations beating 2x their null** — so it is a genuine, independent
building signal, not a copy and not noise. Median **recall 0.561**: MS finds
about 56% of the buildings OSM maps.

**It adds substantially more than it confirms.** `ms_only` area is a median
**1.66%** of the tile against an OSM building coverage of ~1.8% — MS roughly
doubles the mapped building area, and at komariv it draws 6.99% MS-only against
0.02% OSM-only while still recovering 95% of OSM's buildings. The
native-resolution zooms show these are real, building-shaped polygons on real
roofs, not spurious blobs. **So yes, OSM misses buildings over rural Ukraine**,
which is question 1 answered.

### The offset hypothesis is not supported

Only **4 of 9** locations have a trustworthy offset on both sides
(`peak_z >= 4`), and across those four:

| | |
|---|---|
| median \|MS - OSM\| | **8.83 px = 3.4 m** |
| corr(dx) | **-0.073** |
| corr(dy) | **+0.611** |

Two independent maps do **not** agree on the offset vector. `dx` is
uncorrelated outright; `dy` is moderate but rests on n=4, which cannot carry a
conclusion either way. The zooms show red and green polygons of near-identical
shape sitting systematically apart, so the disagreement is registration, not
detection.

**Consequence for the dataset:** the per-location `label_offset` correction in
`crops.py` gains no independent support from this. It is not refuted either —
this probe simply lacks the power to settle it at n=4. Settling it needs the
full 59, where ~26 locations should clear `peak_z >= 4` on both sides.

### The caveat: a coverage hole

**`ploske` returned 0 footprints** — its quadkey is present in the index and
downloaded fine, but contains nothing over that ground, while OSM maps 1.96% of
the tile as building. Any merge has to treat MS absence as *unknown*, never as
*no building*, or it will erase real buildings at locations like this one.

## What I ran

- `probe.ipynb` — the whole probe. Fetches `dataset-links.csv`, maps each
  location to its z9 quadkey, downloads the matching `.csv.gz` (contents are
  **geojsonl**, not CSV — the extension lies), rasterises through
  `TileGrid.to_pixel` so the footprints land where the dataset's own labels land,
  and scores against `label.png` shifted back by the recorded `label_offset`.
- Offsets are measured with `align.offset(gradient_magnitude(grey), outline(mask), mpp)`
  — the same call the build used — and only trusted at `peak_z >= 4`.
- Cache: `data/derived/ms-buildings/` (gitignored).

Licence: **CDLA Permissive 2.0**. Clean, unlike the NC datasets.

## Evidence

- `figures/ms-vs-osm.png` — imagery, red/green/yellow agreement overlay, and a
  native-resolution zoom. The zoom is what decides whether these are real
  footprints on real roofs.

## Cost

8 z9 quadkeys, **84 MB**, 27 s to fetch and parse **1.1 M footprints** from
cache. Whole 9-location run is a few minutes and needs no GPU. All 59 locations
need 27 quadkeys, ~284 MB.

## If it graduates

`burn()` → `uavloc/data/buildings.py` alongside `osm.rasterise`. Not before a
verdict.
