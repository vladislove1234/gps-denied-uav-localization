# map-derived-labels

**Status:** worked · **Question:** `Do OSM features and Delineate Anything boundaries line up with satellite imagery well enough to use as segmentation labels?`

> Answered **yes, but not with Google Earth screen exports** — those have no
> georeference and fitting one from pushpins costs 3.4 m that cannot be
> diagnosed away. With Web Mercator XYZ tiles, where the georeference is
> closed-form, OSM lands within **1.51 m** of the imagery with nothing fitted.
> The sections below are in the order the work happened; the last two are the
> ones that hold.

## The question

Segmentation labels have to come from somewhere, and hand-annotating 4,137 image
pairs is not an option. Map data is attractive because it is derived from
*geometry, not pixels* — a pretrained RGB segmenter inherits the exact domain gap
we are trying to escape, since it can label the same ground differently in a
drone frame and a stale satellite tile. Map data cannot do that.

The catch is registration. OSM and Google Earth are digitised from different
basemaps and can sit several metres apart; at the 5–10 m scale this project
measures position error at, that is label noise large enough to swamp the signal.
So: measure the disagreement before building anything on top of it.

## Verdict

**Unanswered — and the blocker is the test imagery, not the method.** The
pipeline runs end to end, but the 2025 Gonchary orthophoto cannot settle the
question: it was exported from Google Earth with the view rotated, so **49.7% of
the file is black padding**, and **every mapped OSM feature falls in the black
half** (0.0% of building pixels and 0.1% of road pixels land on real imagery).
There is literally nothing under the labels to align them to. The 20.18 m
"offset" the correlation reports is noise — peak z is 0.82 against a threshold of
4, and it is pinned against the search limit.

Two things *were* settled:

1. **Delineate Anything v2 works on this imagery when there is structure to
   find.** It traced the treeline along the field's edge accurately, following
   the canopy outline in detail (`gonchary-boundaries.png`, lower polygon).
2. **On near-uniform terrain it emits spurious nested rectangles.** The visible
   half of this orthophoto is one grass field, and the model fills it with
   overlapping boxes that are not parcels. Anything built on this needs a
   rectangularity filter, not just a confidence threshold — these survive
   `conf=0.1`.

The environment risk flagged before starting did not materialise: `ultralytics`
8.4.150 installs cleanly against the pinned torch 2.2.2, so the ONNX fallback was
not needed.

## What I ran

- `geo.py` — bbox ↔ pixel for a north-up georeferenced image. A local copy on
  purpose; the archive has three disagreeing implementations (bug-008) and
  nothing earns a place in `uavloc/` until this returns a verdict.
- `osm.py` — Overpass fetch with endpoint rotation (the main instance 504s
  constantly) and disk cache, plus rasterisation to class masks. Roads are
  centrelines and get buffered by assumed widths from `ROAD_WIDTH_M`; those
  widths are guesses and are the largest error source in anything linear.
- `boundaries.py` — Delineate Anything v2 via `ultralytics`, tiled with overlap,
  polygons kept only if their centroid lands in the tile core. Skips tiles that
  are mostly black padding.
- `align.py` — correlates OSM building outlines against image gradient to find
  the shift that lines them up.
- `run_gonchary.py` — the driver. Numbers land in `results.json`.

Weights: `MykolaL/DelineateAnythingv2.pt` (125 MB) from the Hub, cached in
`~/.cache/huggingface/`. AGPL-3.0 — irrelevant while it is used offline to
generate labels, relevant if the model is ever distributed.

`ultralytics==8.4.150`, `shapely==2.1.2`, `huggingface_hub`, installed under a
constraints file pinning `torch==2.2.2`/`torchvision==0.17.2`. **Deliberately not
added to `environment.yml`** — pinning a shared file is a commitment this PoC has
not earned.

## Evidence

- `figures/gonchary-overlay.png` — four panels: imagery, +OSM, +Delineate
  Anything, +both. Shows the black wedge and that every OSM building and road
  sits inside it.
- `figures/gonchary-boundaries.png` — the model's output alone. The treeline
  polygon is the success; the nested rectangles over the grass are the failure.
- `figures/gonchary-osm-offset.png` — the alignment attempt. Both panels are
  almost entirely black, which *is* the finding.

What this does not show: whether OSM aligns with Google Earth imagery **anywhere**.
That needs an export that is north-up and contains buildings. Nor does it show how
Delineate Anything behaves on terrain that actually has field boundaries — the one
clean rectangle of imagery here is 206 × 120 m of uniform grass.

Sanity numbers from the same run, for the record: OSM over this extent is 49 ways
(41 buildings, 6 roads) covering 4.1% of the frame; 95.9% of the rasterised mask
is background.

## Cost

About 40 minutes wall-clock end to end, dominated by a 125 MB weight download and
the resolution/confidence sweep. A re-run is ~30 s: OSM is cached on disk, the
weights are cached by the Hub, and inference over the full orthophoto is 8.8 s on
CPU. **CPU beat MPS 7.3 s vs 27.2 s** at this image size — the Metal transfer
overhead dominates for one-shot inference, so do not reach for `device="mps"` here.

Resolution and confidence sweep (polygons found, full orthophoto before the
padding guard):

| m/px | conf 0.05 | 0.15 | 0.3 |
|---|---|---|---|
| 0.25 | 38 | 17 | 6 |
| 0.5 | 11 | 4 | 3 |
| 1.0 | 11 | 2 | 2 |
| 2.0 | 0 | 0 | 0 |

0.25 m/px is the working resolution. At 2.0 m/px the model returns nothing, so
the 6.6 cm/px orthophoto must be downsampled into roughly 0.25–1.0 m/px — not
coarser.

## If it graduates

`geo.py` → `uavloc/geo.py` (and retires the archive's three variants),
`osm.py` → `uavloc/data/osm.py`, `boundaries.py` → `uavloc/data/boundaries.py`,
`align.py` → `uavloc/data/align.py`. Not before the question is answered on
imagery that can answer it.


## Second attempt: Davydiv (2026-09-13)

A fresh Google Earth export, 3454 x 2454, supplied as a matched pair — one with
four placemarks, one without — so the markers can be located by differencing the
two images rather than by eye.

**The export itself is good.** Fitting an affine from the four pins gives
rotation **+0.486 deg** and anisotropy **1.0104**, i.e. north-up with square
pixels, and **99.32%** of the frame is real imagery — none of the black wedge
that made the Gonchary orthophoto useless. Scale is ~0.263 m/px over ~909 x 643 m.
The "Умовні позначення" legend box was left enabled in Map Options, but it appears
**only in the marked copy**, so the clean image is uncontaminated.

**Delineate Anything works well here: 549 field polygons, against 12 at
Gonchary.** Median parcel 0.14 ha. The long narrow strip parcels typical of a
Ukrainian village are traced cleanly and follow boundaries that are genuinely
visible. Two caveats: it over-segments the built-up area, drawing polygons around
houses and gardens — expected from a field model, and it needs masking or a size
filter — and the tree-covered corner produces tangled overlapping polygons.

This is the terrain type the approach needs. Strip parcels are tens of metres
wide, so a 46 x 26 m frame at 40 m altitude already contains a boundary or two,
and a 116 x 65 m frame at 100 m contains several.

**Still blocked on exact coordinates.** *(Superseded 2026-09-13 — the KML
arrived and the coordinates are exact as stated; the reasoning below is wrong,
see "The KML arrived" at the end of this file and `bug-011`.)* The four stated positions are rounded to
four decimal places, and at this latitude 0.0001 deg is 11.1 m of latitude and
7.2 m of longitude. The evidence that rounding — not a misplaced pin — is the
cause: leave-one-out errors are spread across all four markers (2.25, 7.90,
10.27, 13.24 m) with no single outlier, and pairwise distances disagree with the
pixel geometry by +2.7, -4.7, +5.8, -2.3, +2.6 and -6.3 m in both directions.
A 6 m georeferencing error is the same size as the effect the alignment check is
meant to measure, so nothing is built on it.

The fix is to stop transcribing coordinates altogether: right-click the
placemark folder in Google Earth Pro, **Save Place As -> .kml**, and the file
carries full precision. Then the georeference is exact and the OSM overlay and
alignment number follow immediately.

- `figures/davydiv-boundaries.png` — the 549 polygons over the clean image.


## OSM overlaid on Davydiv — it lines up

Run with `run_davydiv.py`. The georeference is an `Affine2D` fitted from the four
placemark needle tips; `geo.py` grew that class because `Bounds` assumes a perfect
north-up box and this export carries +0.486 deg of rotation, worth ~4 m across the
frame.

**OSM has real coverage here:** 289 ways — 139 buildings, 100 roads, 6 water,
4 wood — labelling 9.3% of pixels, against 4.1% at Gonchary.

**And it aligns.** Measured offset between OSM building footprints and the
imagery is **2.92 m** (dx +1.31, dy +2.61), peak z 3.97. Visually confirmed in
`figures/davydiv-osm-zoom.png`: footprints sit on the roofs and road centrelines
follow the actual lanes.

*(Superseded — the error bar below was misattributed; see the last section.)*
The honest reading is **"OSM agrees with Google Earth imagery to within about
3 m"** — and no better statement is available yet, because the georeference is
itself uncertain to ~3.4 m from the rounded coordinates. The measured offset is
*inside* our own error bar. With a KML this resolves into a real number.

Two things to fix before this becomes labels:

* **Road buffer widths are too generous.** `ROAD_WIDTH_M` puts 6.31% of pixels in
  the road class and the bands visibly overshoot the narrow village lanes. Those
  widths were always guesses; they need tuning against the imagery.
* **peak z 3.97 is borderline** (threshold 4). It is not pinned at the search
  limit, so it is a real peak rather than the Gonchary non-result, but it is not
  a strong one — 139 buildings is not many for this method.

**The two sources are complementary, which was not obvious in advance.** OSM
covers the built-up strip and has nothing to say about the fields; Delineate
Anything covers the agricultural strips and over-segments the village. Used
together they label most of the frame.

- `figures/davydiv-overlay.png` — four panels over the whole export.
- `figures/davydiv-osm-zoom.png` — the alignment, judgeable by eye.


## The KML arrived, and it closes the question the wrong way (2026-09-13)

`satelites/points.kml` — one Placemark holding a `MultiGeometry` of four Points.
**It carries no more precision than we already had.** All four coordinates match
`CONTROL` exactly:

| pin | KML | in `CONTROL` |
|---|---|---|
| lt | 49.75339999999999, 24.1368 | 49.753400, 24.136800 |
| rt | 49.7533, 24.1479 | 49.753300, 24.147900 |
| rb | 49.7493, 24.1470 | 49.749300, 24.147000 |
| lb | 49.7493, 24.1368 | 49.749300, 24.136800 |

**Google Earth is not truncating — it emits full precision, and the full
precision is the round number.** Two proofs from inside the file: the `LookAt`
block carries 14 significant digits (`24.14218712334724`), and the lt latitude is
written as `49.75339999999999`, which is *not* the double nearest to 49.7534
(`float("49.75339999999999") == 49.7534` is `False`). A serialiser that rounded
to four places could not produce either. The pins really do sit at those
coordinates — they were placed there deliberately, on a round-number grid.

So there is nothing left to recover, and the ±3.4 m quantisation error bar was
never the explanation. Something else produces the 3.4 m inconsistency.

### The old evidence for "it is rounding" was vacuous

This is the part worth not repeating (`bug-011`). Four control points against a
six-parameter affine leave a **one-dimensional** residual space: the null space
of the 4x3 design matrix transpose is a single vector

    n = (1, -1.025, 1.1154, -1.0904)

and the x- and y-residuals are each *forced* to be a scalar multiple of it,
whatever the underlying cause. Measured, they are — to four significant figures:

    forced shape   1.0000  -1.0250   1.1154  -1.0904
    observed dx    1.0000  -1.0243   1.1144  -1.0900
    observed dy    1.0000  -1.0248   1.1156  -1.0908

Consequences: the alternating signs carry no information; leave-one-out errors
are proportional to `1/n_i` and therefore **can never single out an outlier**;
and pairwise-distance disagreements "in both directions" are likewise forced.
The previous session read all three as evidence for quantisation. None of them
discriminate between *any* two causes. Only the residual's single scalar
magnitude — 3.4 m — is real.

### The pixel positions are not the cause either

Re-derived independently: difference the two exports, label the blobs (4 pins at
~1620-1714 px, plus the 7174 px legend box), then cross-correlate each pin's
patch against the lt patch with sub-pixel parabolic peak refinement. All four
pins are the same icon at the same scale, so correlation gives their *relative*
positions far more reliably than a bounding-box corner does.

- Two independent signals — binary blob shape and raw difference magnitude —
  agree with each other to **0.4 px**.
- Both agree with the hand-picked needle tips in `CONTROL` to **1-2 px**.
- Substituting the correlated positions changes the affine residual from
  11.81 px to 11.87 px. **No improvement.**

Pixels are good to ~1 px = 0.26 m. The 12 px inconsistency is not there.

### What that leaves, and why it matters

Coordinates exact, pixels exact to a quarter-metre, and yet no affine reconciles
them better than 3.4 m over a 909 m frame — 0.38%. Four points cannot say which
assumption is wrong, and no amount of care with *these four* pins will fix it.

Worse, the four are nearly degenerate for this purpose. Refitting with each pin
dropped in turn moves the georeference by up to **13.9 m** across the frame, and
each drop-one fit disagrees with the least-squares fit by ~10 m:

    spread across the five fits: median 12.76 m, max 13.91 m

**Four pushpins are not a georeference.** That is the finding.

### OSM is the better control, and it says the transform is already fine

Independent check: instead of one global alignment, run `align.offset` over a
grid of overlapping 700 px tiles and ask whether the OSM-to-imagery offset
*drifts* across the frame. A wrong affine would show up as offset varying
linearly with position; a correct affine would show a constant offset.

Buildings alone only cover the built-up strip, so the tiles are nearly
collinear (design condition number 7337-10070) and drift is not separable in
both axes. What can be said:

| layer | tiles | strong (peak_z>=4.5) | dx | dy | mean offset |
|---|---|---|---|---|---|
| buildings | 3 | 1 | +0.00 … +2.63 | +1.58 … +2.89 | 1.6-4.5 m |
| buildings+roads | 36 | 11 | +0.45 ± 4.40 | +1.00 ± 3.55 | 5.07 m |

Fitting offset as linear-in-position barely helps — dx scatter 5.63 → 5.42 m
over all tiles, 4.40 → 3.81 m over the strong ones. **There is no detectable
systematic drift.** The per-tile scatter is dominated by correlation noise
(weak road edges, guessed widths), not by a wrong transform.

So the practical position is better than the pins imply: the mean offset over
the strong tiles is **+0.45, +1.00 m** — within a metre of zero — and the
current affine already puts OSM where it belongs, to within the 3-4 m that the
measurement itself can resolve.

**Verdict on the blocker: it is not a blocker.** The KML cannot improve the
georeference and neither can better pin-picking. Stop treating the four pins as
the arbiter. The 139 building footprints are a denser and better-conditioned
control than four pushpins, and they already agree.

The honest claim is unchanged in value but now differently sourced: *OSM agrees
with this Google Earth imagery to about 1 m in the mean, with 3-4 m of
location-to-location scatter that the current method cannot separate from its
own noise.* That 3-4 m is the label-noise figure the dataset has to live with,
and it is the number to quote — not 2.92 m from a single global correlation.

**Not verified, because it cannot be here:** whether the residual 3.4 m
inconsistency is a real non-affine property of the export or a misplaced pin.
Settling it needs control points that are not pushpins — road junctions read off
the imagery, or a second export at a different zoom.


## XYZ tiles instead of screenshots — the georeference problem goes away (2026-09-13)

Run with `run_davydiv_tiles.py`; numbers in `results-davydiv-tiles.json`. Same
ground as the Google Earth export on purpose, so every number has something to
beat.

**The premise.** A Google Earth screen export carries no geometry, which is why
four pushpins and a 3.4 m residual happened at all. A Web Mercator tile carries
its geometry by definition: `(z, x, y)` *is* a box, so pixel -> lat/lon is
closed form. Davydiv at z19 is tile 297303/178371 at 0.1929 m/px — finer than
the 0.2614 m/px of the export, and the lat/lon round-trip closes to **1e-9 m**.
Nothing is fitted, so there is no residual to diagnose.

**Esri Wayback makes the same tile available on 196 past dates**, under the same
scheme — so two dates are co-registered by construction. Over Davydiv, 7 of them
are genuinely distinct after hashing three probe tiles each: 2018-04, 2019-03,
2019-12, 2020-10, 2022-12, 2025-02, 2025-10.

### The three bars, and what happened

| | bar | result |
|---|---|---|
| Georeference (OSM vs imagery, nothing fitted) | <= 2 m | **1.51 m** on the three newest dates, `peak_z` 5.6-7.4 |
| Cross-date registration | <= 2 px median | **4.65 px = 0.90 m** median — missed in pixels, passed in metres |
| Appearance variation | >= 4 distinct dates | **7**, correlation vs newest 0.275-0.748 |

**Q1 passed and beats the export route.** 1.51 m with no control points, against
"about 1 m in the mean with 3-4 m of scatter" from the pin-fitted affine — and
that 1.51 m needs no error bar underneath it, because nothing was fitted.
Visually confirmed in `figures/tiles-davydiv-osm-overlay.png`: footprints sit on
roofs, lanes follow lanes.

**Q2 missed its stated bar, and the miss is the useful finding.** The bar was
2 px and the median is 4.65 px, so it is recorded as a miss. But the structure
is clean: 2020-10, 2022-12 and 2025-02 land within **0.04 px** of the newest,
while 2018-04, 2019-03 and 2019-12 sit **+9.2, +16.1 and +10.5 px** away — in
`dx` only, `dy` is zero to two decimals. That is not noise, it is a pure
eastward misregistration of Esri's older imagery over this ground, up to 3.10 m.

It is also *measured*, at a correlation peak ratio of 9.6-12.0, so it is
correctable by a shift. Q1's older dates carry the same error — 2019-03 measures
5.61 m against OSM and is misregistered by 3.10 m — which is most of the gap.

**Q3 passed clearly.** `figures/tiles-davydiv-dates.png` shows haze, bare trees,
low winter sun, autumn brown and full summer foliage over the same roofs and
lanes. A correlation of 0.275 between two dates is a real appearance gap, not a
re-encode.

### Verdict: **worked**

The imagery source was the problem, not the labelling idea. Switching to tiles
removes the fitted georeference, improves resolution, removes the manual export
step entirely, and makes the ground scriptable over the whole country.

Two things the build has to carry as a consequence:

* **Co-register each date to the newest.** Phase correlation, sub-pixel, per
  location. Assuming "same tile coords" means "same pixels" costs 3 m.
* **OSM describes today, the imagery may be from 2018.** A house built in 2021
  is labelled in every date. Preferring recent releases limits this; it does not
  remove it, and it belongs in the dataset card as known label noise.

Also confirmed visually, and still open: `ROAD_WIDTH_M` overshoots narrow lanes,
exactly as the earlier section said. And a single Wayback "date" can itself be a
mosaic of captures — a seam is visible in the overlay figure.

### Where the code went

`tiles.py` (Web Mercator + fetch + mosaic, duck-types `geo.Bounds`),
`wayback.py` (release list, distinct-date discovery), `areas.py` (Ukrainian
village discovery, with the national outline as a point-in-polygon filter),
`run_davydiv_tiles.py` (this run). None of it has graduated to `uavloc/` —
`datasets/ua-village-multidate/build.py` imports it from here via `sys.path`,
deliberately, because graduation is a maintenance commitment and therefore the
user's call.
