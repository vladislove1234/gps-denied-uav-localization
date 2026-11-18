# 020-building-centroid-matching — proposal

> Written **before** the first run. The success criterion below does not move
> once runs begin. If nine runs all miss it, the honest report says the
> approach failed.

## Question

Does matching building centroids (one point per building, drone frame against
satellite map, translation voted over point pairs) and falling back to `007.1`'s
weighted NCC when it finds no unique match lower the mean position error of the
`007.1` correction test at 10 m injected error?

## Hypothesis

Weighted NCC scores how well class *edges* line up. `018` showed that edge
quality does not reach the error: sharper masks on both sides moved it by 0.00 m.
`004`/`005` showed that a straight track gives a ridge, not a peak.

A centroid is the mean of every pixel of a building. Boundary noise from the
stride-8 head averages out, so a building roughly 10 m across should give a
centroid good to about 1 m even when its edges are off by 1–2 m. Two or three
buildings pin the translation in both axes, with no ridge. Point matching also
ignores what is between the buildings (season, field texture), which is where
the dates differ most.

The oracle (hand labels on both sides) separates the idea from the model: if
GT centroids do not beat NCC, no segmenter will.

Stated limits:
- **Needs buildings in the frame.** A 46 × 26 m footprint over bushcha's woody
  ground may hold none. Such draws fall back to NCC, so they neither help nor hurt.
- **Merged and missed buildings move or drop centroids.** LR-ASPP merges ~29 %
  and misses ~25 % of val buildings (`pocs/building-merge-stride`). A merged
  blob's centroid sits between two real ones. Voting tolerates missing points,
  not systematically wrong ones.
- **Regular house rows are look-alikes.** A translation of one house spacing can
  collect as many inliers as the true one. The uniqueness rule below rejects
  these instead of guessing.
- **Translation only.** The frames here have exact heading and scale. The 2025
  flight's heading and scale errors (`pocs/pose-reregistration`) would need a
  similarity fit. That is out of scope here.

## Success criterion

**Metric:** `mean_position_error_m` @ 10 m injected, all 40 frames × 8
directions.

**Baseline:** `020.1`, `007.1`'s correlator through this notebook. It must
reproduce `007.1` (5.861 m) to within 0.05 m. Otherwise the harness moved, and
that is reported first.

**Bar:** primary run `020.4` (model centroids + NCC fallback), paired with
`020.1` over frames: mean difference ≤ −0.5 m **and** 95 % frame-bootstrap CI
upper bound < 0.

**Guard:** at 5 m and 15 m injected, the CI lower bound of the paired difference
must not be above +0.5 m.

**Also reported, not part of the bar:**
- **Viability:** non-border building instances per drone frame, from GT and
  from the model (median, share of frames with ≥ 2), and the share of draws
  where centroid matching fires.
- **Mechanism:** the draws where the centroid matcher fires, split out. On those
  draws, centroid error vs NCC error, paired; harm share on fired draws.
- bushcha vs birky split, harm share @ 10 m, `ms_per_correction` on CPU.

## Method

All geometry is in the heading-up patch centred on the prior, in metres, as in
`SearchSurface.offset_m`.

1. **Instances.** Building components of the class map (`building` argmax, or
   the label), 4-connected, ≥ 16 px (`uavloc.segmentation.instances.components`).
   Centroid = mean pixel position of the component.
2. **Drone side.** Drop components that touch the frame border (≤ 1 px). Their
   centroid moves with the frame, not with the ground.
3. **Map side.** Compute components on a crop one frame half-diagonal larger
   than the `007.1` search patch, so buildings at the patch edge are whole.
4. **Hypotheses.** Every pair (drone building i, map building j) gives a
   drone-centre offset t = c_map(j) − c_drone(i). Pairs with t outside the
   `007.1` search range (`search_half_range_m`, rect ×1.7) are dropped, so the
   reach is the baseline's.
5. **Score.** For each t, inliers = drone centroids that have a map centroid
   within `INLIER_TOL_M = 2.0` after the shift (one-to-one, nearest first).
6. **Decision.** Best t = most inliers, ties broken by the smaller sum of
   residuals. The final t is the mean of (c_map − c_drone) over its inliers.
   **Fire** only if inliers ≥ `MIN_INLIERS = 2` **and** the best count is
   strictly greater than the best count among hypotheses more than
   2 × `INLIER_TOL_M` away. Otherwise the draw falls back: the prior for
   centroid-only runs, `007.1` NCC for hybrid runs.

Constants are fixed here and are not tuned on these frames.

## Registered runs

| Run | Drone centroids | Map centroids | Fallback | Role |
|---|---|---|---|---|
| `020.1` | — | — | — | NCC only (= `007.1`), **baseline** |
| `020.2` | GT (`current`) | GT (map date) | prior | oracle, centroid only |
| `020.3` | LR-ASPP | LR-ASPP | prior | model, centroid only |
| `020.4` | LR-ASPP | LR-ASPP | NCC | **primary** |
| `020.5` | GT (`current`) | GT (map date) | NCC | ceiling of the primary |

GT = `ua-village-annotated` hand labels on the same date as the imagery on each
side: `current` for the frames, the map date for the map (birky 2025-04-24,
bushcha 2024-06-06). Both dates exist.

## Requires

| Equipment | Status | Acceptance check |
|---|---|---|
| `uavloc.matching.centroids`: instance centroids, pair voting, uniqueness rule, output as a `Correction` | to build | **identity:** a map-label crop at the true pose fed as the drone map gives < 0.5 m error on every frame with ≥ 2 non-border buildings; figure of drone and map centroids with matched pairs on 4 frames (2 per place, including one reject) |
| GT class crops for frames and map (`006`'s `gt_query.py`) | exists, experiment-local | second use: move to `uavloc/data/annotated.py` before the runs (promotion rule). `006.14` must still reproduce 8.357 m |
| hybrid corrector (centroid, else NCC) | to build, glue in this folder | on a draw where centroid does not fire, result is bit-identical to `020.1` |
| `village-crossdate-frames`, `007.1` correlator, LR-ASPP satellite BN checkpoint | exist | `020.1` reproduces `007.1` |

**What only you can provide.** A go-ahead on the bar, the guard and the two
constants (`INLIER_TOL_M = 2.0`, `MIN_INLIERS = 2`).

## Setup

| | |
|---|---|
| Dataset | `village-crossdate-frames`: 40 frames on bushcha and the birky val strip, map = newest non-current date |
| Correlator (baseline and fallback) | `007.1`: rect window ×1.7, flat prior, no gate, weighted NCC, class weights (0, 1, 1, 0.5, 0, 0.5) |
| Segmenter | LR-ASPP satellite BN (`pocs/mobilenet-lraspp-speed`), both sides, 0.385 m/px |
| Draws | noise 5 / 10 / 15 m × 8 directions, seed 42 |
| Training | none |
| Hardware | MPS for segmentation, CPU for timing |

## What would falsify this

- **Oracle `020.2`: on fired draws, not ≥ 1 m better than NCC on the same draws
  @ 10 m** → building centroids carry no more position than edges on this
  ground. The direction is dropped, not tuned (no tolerance or inlier sweep).
- **Oracle works, model `020.3` does not** → the idea holds and the segmenter's
  instances are the blocker (merges, misses). Next step goes to instance quality
  (`019`), not to the matcher.
- **Fires on < 20 % of draws, even with GT** → not applicable to ground like
  this. Reported as a coverage limit: the bar cannot be met through the fallback
  alone.

## Threats known in advance

- Simulated frames: satellite crops, exact heading and scale, nadir. Real drone
  roofs lean with perspective (parallax), which moves centroids by metres at the
  frame edge.
- At 10 m only 55 % of draws can reach the truth with the rect ×1.7 window
  (`007.1` `reachable_share_at_10m`). Centroids use the same reach on purpose,
  so a gain is attributable to the matcher. A wider window is cheap for points,
  but it is a separate question.
- bushcha's hand labels are 100 % from source sublayers (OSM, MS footprints), not
  hand-drawn. GT buildings there may be offset or incomplete.
- The birky strip's 20 frames overlap, so birky draws are not independent.
  Resampling is by frame, as in `007`.

## Related

- `007-search-prior-and-gate`: baseline correlator (`007.1`). Across-only moves
  handle ridges by giving up the along-track position; centroids try to resolve
  it instead.
- `018-segfix-correction`: sharper edges do not reach the error.
- `006-segmentation-ceiling`: GT class maps, `gt_query.py`.
- `019-building-gap-loss`: merges cut 35–42 %, at a missed-building cost. Centroids
  are the first matcher that cares about separate buildings.
- `pocs/building-merge-stride`: instance definitions.
