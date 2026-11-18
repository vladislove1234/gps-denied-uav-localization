# 002-perfect-satellite-map — proposal

> Written **before** the first run. The success criterion below does not move
> once runs begin. If nine runs all miss it, the honest report says the
> approach failed.

## Question

If the satellite-side class map were perfect (hand-labelled), would
class-map matching beat VGG11 on the 001 benchmark at 15 m injected error?

## Hypothesis

It would. `001-seg-map-correction/mistakes.ipynb` shows round 3 fails on the
satellite side. The drone class map finds the pasture track cleanly. The
satellite class map never contains it, so the track pixels disagree at every
shift, and the argmax drifts to whichever window is pure background.
Give the satellite side the track and the drone's strongest structure has
something to lock onto.

There is a second, independent problem with the 2025 score (per-pixel cosine
summed over all channels): background agreement dominates it. So a
class-weighted, per-class normalised correlation is pre-registered alongside.

## Success criterion

**Metric:** `mean_position_error_m` (@ 15 m injected; 44 frames × 8 directions, seed 42)
**Baseline:** `001.1` → **11.634 m** (VGG11, 2025 config, `001-seg-map-correction/runs.jsonl`)
**Bar:** either run of this experiment reaches `mean_position_error_m` **< 11.634 m**

The user chose a bar of simply beating VGG11 in 001; the same bar is kept here.

Two runs are pre-registered, and a hit on either counts:

| Run | Satellite side | Drone side | Matching |
|---|---|---|---|
| `002.1` | hand-labelled map | round 3, drone BN | `cosine`: the 2025 correlator, unchanged from 001 |
| `002.2` | hand-labelled map | round 3, drone BN | `weighted-ncc`, weights below |

### `weighted-ncc`, fixed now

For every class channel *c* of the drone map *d* and the satellite patch *s*:

    NCC_c(shift) = Σ (d_c − mean d_c) · s_c  /  ( ‖d_c − mean d_c‖ · ‖s_c − window mean‖ )
    score(shift) = Σ_c w_c · NCC_c(shift) / Σ_c w_c        (sum over classes present in the drone map)

- **A class counts as present** in the drone map when it is the argmax on
  ≥ 1% of drone pixels. Otherwise it is skipped. This stops near-zero
  probability noise from being normalised up to full weight.
- **A satellite window whose standard deviation for class *c* is < 0.05**
  contributes 0 for *c*. A flat window carries no position information.
- **Weights:** background **0**, building **1.0**, track **1.0**, woody
  **0.5**, water **0.5**, field **0**. Why: tracks and buildings are stable
  across seasons and years; woody is seasonal (frosted April shrubs vs summer
  canopy); water extent varies; field has no labels anywhere; background is
  the complement of the other classes and only rewards absence.
- The score is z-scored over shifts and the argmax is taken, as in `cosine`.
  `peak_z` stays comparable.
- If no weighted class is present in the drone map, the heatmap is flat and
  the correction keeps the prior (`peak_z` 0).

These weights are **not tuned**. If they are changed, that is a new run with
a note, and the bar still applies to the pre-registered pair.

## Requires

| Equipment | Status | Acceptance check |
|---|---|---|
| Hand-labelled Gonchary satellite map, exported by the annotator | **waiting on you** | Covers the box below. Labelled from satellite imagery only (see *What only you can provide*). |
| `uavloc.data.annotated.load_label_map(location, date)` → `GeoImage` of class ids from `ua-village-annotated` | built (`builds/perfect-map-matching`) | On birky/current the label lines up with its own image (`figures/label-map-overlay.png`). |
| `SegmentationFeatures.from_satellite_probabilities` + `label_probabilities(labels, mpp)` | built | Identity on the birky label map: 0.17 m median, 0.22 m max, for both matching modes. |
| `CorrelatorConfig.matching = "weighted-ncc"` + `class_weights` | built | `cosine` replays 12 draws of `001.3` to 2e-10 m. Identity is < 1 m wherever a weighted class is present at the true spot (0.1–0.2 m); where none is present it keeps the prior by design. |
| `uavloc.features.factory`: build features and correlator config from the logged run config | built | Used by both experiments' `train.ipynb`. |
| Orthophoto ↔ label-imagery offset | moot: labels drawn on the orthophoto | Phase correlation of the label location's imagery against the orthophoto over the box, peak-ratio guarded. The number is reported, not corrected for. |
| `position-correction-harness`, round-3 BN checkpoints | exists | — |

**What only you can provide.**

- [x] **The Gonchary satellite labels** (arrived 2026-09-27: `ua-ortho-annotated`, `gonchary-2025` / `ortho`), done in the annotator and exported
  (`python -m tools.annotator export --new`). They are expected in
  `data/derived/ua-village-annotated/samples.jsonl` as a record whose
  `location` starts with `gonchary`, with `bounds` and `label_file`.
  - **Coverage:** at least **top-left 49.761905, 24.167825 → bottom-right
    49.759486, 24.172788** (≈ 357 × 269 m). That is every frame plus the
    farthest a 15 m-shifted search window reaches at the highest altitude
    (58.8 m).
  - **Label only what the satellite image shows.** Don't draw a track or a
    shrub because you remember it from the drone footage. If a structure is
    only in the drone frames, it isn't in the map the drone would carry. Using
    it would leak the test into the reference.
  - **Say which date is the map.** If several dates are annotated, name the
    one to use. The default is the date closest to the 2025 orthophoto's
    appearance.

**Amended 2026-09-27, when the labels arrived; the criterion is unchanged.**
- **The labels were drawn on the 2025 orthophoto itself** (annotator `ortho`
  collection → `ua-ortho-annotated/gonchary-2025/label.png`, 2731 × 2009 at
  ~0.2 m/px, date `ortho`), not on the tile imagery. The map therefore has
  exactly the georeference VGG11 matches against: same corners
  (`builds/perfect-map-matching`, check *Gonchary labels*). The
  orthophoto ↔ label-imagery offset row is moot, since it is zero by
  construction, and the *different map for each method* confound no longer
  applies.
- **What is labelled:** track 1.2% and woody 4.0% of the required box,
  covering the pasture track along the hedge line plus the shrubs and tree
  clumps beside it (`figures/gonchary-labels.png` in the build). **Unlabelled
  inside the box:** the tarmac road on the west edge, the houses in the
  south-east (so building has weight 1.0 but no labels), and most of the
  dense tree cover to the south. The frames all sit over the field north of
  the tree line, so these gaps fall at the edges of the search windows, not
  under the frames. The runs use the map as delivered.

> Building must not move the success criterion above. If the build reveals the
> metric is unmeasurable as promised, amend this file with a dated note saying
> what changed and why. Do not quietly swap it.

## Setup

| | |
|---|---|
| Frames | `drone-2025-easy`, all 44. Same as 001. |
| Noise | 15 m primary, 5 and 10 m logged. 8 directions per frame, seed 42. Same draws as 001. |
| Correlator geometry | 2025: scale 1.7, FOV 60°, 3% satellite-border crop |
| Drone side | `model-round3-bn-drone.pt` at 0.10 m/px, probabilities on the 0.385 m/px grid |
| Satellite side | the hand labels, one-hot, area-averaged to 0.385 m/px |
| Hardware | MPS |

**Diagnostics** (never the criterion): the same per-subset means as 001
(2025 train/val folders, seg held-out 16), share of draws improved, error at
5/10 m, mean `peak_z`, and mean bias east/north.

## Known confounds, accepted up front

- **Different map for each method.** VGG11 matches the 2025 orthophoto; the
  labels are drawn on the annotator's tile imagery. If the two are
  georeferenced differently, one method gets a better-placed map. The offset
  is measured and reported (Requires). The 3–4 m northward bias in every 001
  run may be the orthophoto's.
- **This is an upper bound.** A result here says what a perfect map can do.
  It does not say that any model produces such a map.
- **The drone side trained on 22 of the 44 frames**, as in 001. The held-out
  subset is logged.

## What would falsify this

Both `002.1` and `002.2` end at ≥ 11.634 m. Then even a perfect satellite map
does not let class-map matching beat VGG11 here, and training a better
satellite model (e.g. on Gonchary) is not worth doing for this correlator.
The bottleneck would be the drone side or the matching itself.

## Related

- `001-seg-map-correction`: baseline `001.1`; the round-3 failure analysis in
  `mistakes.ipynb` and the report's *Where round 3 goes wrong*. `001.4`
  (weighted NCC on round-3 satellite maps) is the pre-registered 001 variant
  run with the same new matching code.
- `builds/position-correction-harness`: the equipment this extends.
