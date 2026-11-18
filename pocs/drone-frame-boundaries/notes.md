# drone-frame-boundaries

**Status:** failed · **Question:** `Can Delineate Anything label the 2025 drone frames, so that one coordinate-addressed pipeline supervises both the satellite and the drone side?`

## The question

`ua-village-multidate` labels satellite tiles from OSM. A drone frame carries a
MAVLink pose, so the same coordinates address both — the label source does not
have to be a map, only something addressable by lat/lon. Delineate Anything is
addressable that way and, unlike OSM, it covers unmapped field. So: does it fire
on a drone frame, and do the boundaries it draws there agree with the ones it
draws on the satellite tile of the same ground?

## Verdict

**No, and the reason is geometric, not a model failure.** At the 2025 flight's
41 m AGL the camera sees 50 × 26 m of ground — 0.13 ha. The median Delineate
Anything parcel over the same area is **1.14 ha**, about 107 m on a side. A
frame is **11% of one parcel**, and across 12 sampled frames **0 of the 91
control parcels** had its centroid inside any footprint. DA does return 1–4
polygons per drone frame, but since there is no parcel in view those polygons
are texture inside a single field, not boundaries.

This is not "DA fails on drone imagery" — it was never given enough ground to
answer that. The binding constraint is **altitude**: one median parcel enters
the frame at **122 m AGL** with this 63° camera. It is also not the terrain the
approach assumes: the flight ground is scrubby pasture with an unmapped dirt
track, not parcelled farmland.

## What I ran

- `run.py` — joins every frame in `../../../archive/uav_localization_2025/dataset_utils/image_pairs_v2`
  to a pose from `gps_data.csv` (same epoch-ms key; worst join gap 100 ms), then
  runs DA on the drone frame and on a satellite crop of the identical ground at
  matched pixel sizes. Writes `results.json`.
- `analyse.py` — DA over the whole 790 m mosaic, parcel-size distribution, and
  how many of those parcels a drone footprint can contain. Writes `scale.json`.
- `figure.py` — the evidence figure.
- `fetch.py` — pulls the 8 × 8 z18 Esri mosaic over the flight area through
  `pocs/map-derived-labels/tiles.py`. **Run this first.** The mosaic and DA's
  control polygons land in `satelites/`, which is gitignored — Esri tiles are
  fetched for offline label generation and are not redistributed. Only
  `gonchary-grid.json` (five integers naming the tile box) is tracked.

Run with the **`uavloc` conda env** — `ultralytics` is not in `base`. Weights
(`MykolaL/DelineateAnythingv2.pt`, 125 MB) were already in the HF cache.

Geometry uses the 2025 calibration found in the archive's
`dataset_constructor_test.ipynb`: `fov_width_deg=63`, `fov_height_deg=35`,
`height_correlation=-13` (altitude offset). Median AGL over the frames is then
39.5 m. The conclusion does not turn on that offset — dropping it gives 52 m AGL
and 0.24 parcels per frame, still an order of magnitude short.

## Evidence

- `figures/footprint-vs-parcel.png` — the whole argument in one panel. Left: the
  790 m Esri mosaic with DA's 91 parcels (yellow), the 254 OSM ways (blue), the
  flight track (cyan) and 12 camera footprints (red). The track sits in the gap
  between everything that is mapped, and the footprints are specks against the
  parcels. Right: a frame and the same ground from Esri, then the parcel-area
  histogram with the footprint marked, and the altitude curve crossing one
  parcel per frame at 122 m.
- `figures/crop-registration-check.png` — drone frame, the 2025 pipeline's own
  Google Earth patch, and my Esri crop for three frames. Included because the
  satellite crop is the one thing here that could have been silently wrong.

**Independent check of the dataset card's claim.** `card.md` says 0 of 4,181
frame positions have a mapped feature in view. Re-measured here over 207 sampled
frames against a fresh Overpass fetch: **0/207** inside the footprint, nearest
mapped feature **64 m** at closest and **137 m** median. The claim holds.

### What this does not prove

- **DA's polygons are visibly over-segmented** in the upper field of the figure
  — nested near-duplicate rectangles, the failure mode the earlier PoC already
  recorded. So "1.14 ha median" is a median over noisy polygons. The conclusion
  survives it: even the 10th-percentile parcel is 1,303 m², which is the size of
  the whole frame.
- **The satellite crop was never registered to sub-metre accuracy.** Correlation
  against the 2025 pipeline's own Google Earth patch is +0.017 median, which
  sounds damning but is uninformative — different provider, different season,
  and my crop is 131 × 67 real pixels. The registration check figure is a visual
  judgement, not a measurement.
- **The matched-pixel-size arm of `run.py` is not interpretable** and its `sat`
  counts should not be quoted. Presenting a 131 px satellite crop at 512 or 1024
  px upsamples detail that does not exist, so DA sees blur and returns nothing;
  the drone frame at the same size is a genuine downsample of 1920 px. The two
  arms are not comparable. The parcel-count argument replaced it.
- **Nothing here tests a higher flight.** 122 m is arithmetic from the FOV and
  the measured parcel size, not an observation.

## Cost

About 90 minutes. Dominated by DA inference on CPU (12 frames × 3 scales × 2
domains, plus two full-mosaic passes). Tile and OSM fetches were seconds and are
cached. Re-running `figure.py` alone is ~20 s.

## If it graduates

It does not. The pose join in `run.py:poses()` is the one piece worth keeping —
frame filename → `gps_data.csv` → lat/lon/alt/yaw is the bridge to every future
use of the 2025 flight data, and it is 12 lines. Move it only when something
else needs it.
