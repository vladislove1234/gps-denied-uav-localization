# 006-segmentation-ceiling — proposal

> Written **before** the first run. The success criterion below does not move
> once runs begin. If nine runs all miss it, the honest report says the
> approach failed.

## Question

At 15 m injected error, how much localization error does imperfect onboard
segmentation cost? Put a number on it by replacing the "drone" class map with
ground truth in the `005.5` setup. If the cost is large, is it mostly **edges**
(blurred boundaries, rounded corners) or **semantics** (missed or wrong objects)?

## Hypothesis

Segmentation is not the main bottleneck. In 005 the large failures come from the
matcher and the scene: on birky-15 a straight road 34 m away outscores the true
spot, 0.87 vs 0.48, and 27.5% of draws held too little structure to move.
Perfect masks cannot fix either one. We expect a GT query to gain under 2 m.

If it gains more, we expect most of the gain to come from edges. LR-ASPP predicts
at stride 8 (about 3 m cells at 0.385 m/px) and upsamples bilinearly, which is
what rounds building corners. Degrading the GT query to stride 8 should then
reproduce most of the model's error.

This experiment decides whether ideas 12–14 get built: boundary loss, a
boundary head, FFT loss. It trains nothing.

## Success criterion

The question is diagnostic, so the bar is two pre-registered decision rules
rather than one number to beat. All numbers are `mean_position_error_m` @ 15 m
injected, over 40 frames × 8 directions, seed 42. Report @ 5 m and @ 10 m too.
Confidence intervals are 95% frame-bootstrap intervals of the paired
difference, as in 005.

**Metric:** `mean_position_error_m` (@ 15 m injected)
**Baseline:** `006.1`, a re-run of `005.5` (9.29 m) on the current code

**Rule 1: is segmentation worth improving?**
Yes if `006.3` (GT query) ≤ `006.1` − **2.0 m** and the CI of `006.3 − 006.1`
is clear of 0. The 2 m threshold matches the gain a fresher map bought in 005
(2.0–2.6 m on birky), so segmentation work has to be worth at least that much.

**Rule 2: are edges the cause?** Evaluated only if Rule 1 says yes.
Let `G = err(006.1) − err(006.3)`. Edges are the cause if the stride-8 degraded
GT (`006.8`) **or** the r = 2 px boundary-damaged GT (`006.11`) has error
≥ `err(006.3) + 0.5·G`. In words: degrading only the edges of perfect masks
brings back at least half of the model's cost.

## Pre-registered runs

Everything not listed is fixed at `005.5`'s values: `map_date: newest`,
`search_scale: 1.7`, weighted NCC, weights `[0, 1, 1, 0.5, 0, 0.5]`, FOV 60°,
3% satellite border crop, and `model-lraspp-bn-satellite.pt` wherever a side
is "model".

| Run | Query ("drone") class map | Map class map | Asks |
|---|---|---|---|
| `006.1` | model | model | baseline, reproduces `005.5` |
| `006.2` | model | **GT** | does a perfect map help? (same question as `002`, but on this ground) |
| `006.3` | **GT** | model | ceiling of onboard segmentation (Rule 1) |
| `006.4` | model, with the **building** channel from GT | model | which class costs most |
| `006.5` | model, with the **track** channel from GT | model | which class costs most (see confounds) |
| `006.6`–`006.9` | GT degraded to **stride** s = 2, 4, 8, 16 | model | resolution / edge-rounding curve (Rule 2) |
| `006.10`–`006.13` | GT degraded by **boundary damage** r = 1, 2, 4, 8 px | model | boundary-precision curve (Rule 2) |

Definitions of the damage:

- **Stride s:** one-hot GT → area-average downsample by s → bilinear upsample
  back → argmax. This imitates a head that predicts at 1/s resolution.
- **Boundary damage r:** per class, a morphological opening and then a closing
  with a disk of radius r px. This rounds convex and concave corners and
  removes structures narrower than 2r. r = 1, 2, 4, 8 px ≈ 0.4, 0.8, 1.5, 3.1 m.
- GT is fed to the matcher as one-hot channels, in the model's class order
  (background, building, track, woody, field, water).

New config keys, so that `log_run` records them in `changed`: `query_source`
(`model` | `gt`), `map_source`, `gt_classes` (all or a list), `damage`
(`none` | `stride` | `boundary`), `damage_px`.

## Requires

| Equipment | Status | Acceptance check |
|---|---|---|
| `village-crossdate-frames` (40 frames, poses) | exists | — |
| `ua-village-multidate` `label.png` for bushcha and birky | exists | one label per location, shared by all dates and on the same pixel grid as the imagery |
| GT query crop: the label cut at each frame's pose with the same rotation and crop as the frame image | to build (glue in this folder) | overlaying the argmax on the frame image matches visually on 3 frames per place (figure in `figures/`); at 0 m noise, GT query vs GT map gives NCC ≈ 1 at the true pose |
| GT map: `label.png` as one-hot map channels | to build (glue) | same shape and georeference as the `newest` map mosaic |
| Damage functions (stride, boundary) | to build (glue; move to `uavloc/` only if a second experiment needs them) | s = 1 / r = 0 is the identity; a figure of every damage level on one frame |
| Class-channel substitution (one channel from GT, the rest from the model) | to build (glue) | a substituted channel equals the GT one-hot channel exactly |

Start from `005-village-crossdate/train.ipynb`. The correlator, metrics and
`log_run` path must be the same code, so that `006.1` reproduces 9.29 m. If it
does not, stop and report before running the rest.

**What only you can provide.** Nothing.

> Building must not move the success criterion above. If the build reveals the
> metric is unmeasurable as promised, amend this file with a dated note saying
> what changed and why. Do not quietly swap it.

## Setup

| | |
|---|---|
| Dataset | `village-crossdate-frames` (query), `ua-village-multidate` newest date + `label.png` (map) |
| Split | test only; bushcha + birky strip are unseen by the model |
| Seed | 42; noise 5 / 10 / 15 m × 8 directions |
| Model | MobileNetV3-Large LR-ASPP, `model-lraspp-bn-satellite.pt`, both sides |
| Loss | none, no training |
| Hardware | CPU |

## Known confounds, accepted up front

- **Both sides share one label.** That is why there is no GT-vs-GT run: it
  would be the identical mask on both sides, a trivial 0 m.
- **OSM road widths are guesses, 2–3× the real track** (`ua-village-multidate`
  card). A GT track channel is wider than the model's track on the map side, so
  `006.3` and `006.5` understate how much a *correct* track mask would help.
  Note this when reading the track numbers. It is a matching-side problem that
  chamfer / skeleton matching would address.
- **OSM is sparse.** Unmapped trees and buildings are background in GT. A GT
  query can therefore hold *less* structure than the model's prediction. Log the
  per-frame share of draws where the prior was kept, for both query sources.
- **OSM describes today.** The query is `current` imagery, so the GT query
  agrees with it. The map side (`006.2`) is a newest-date mosaic, up to ~1 year
  older.
- **Not drone imagery; two places; the birky frames overlap.** These are the same
  threats as in 005. The bootstrap interval is too narrow for them.
- Many runs share `006.3`'s GT query. The damage curves are only interpretable
  relative to `006.3`, not relative to `006.1`.

## What would falsify this

- **`006.3` is not ≥ 2 m better than `006.1`:** mask quality is not the
  bottleneck at this GSD. Drop ideas 12–14 (boundary loss, boundary head, FFT
  loss) and move effort to matching (ambiguity detector, chamfer on skeletons).
- **Rule 1 yes, Rule 2 no:** the cost is semantic (missed or false objects), not
  edges. Work on recall and class confusion, not boundary losses. An FFT loss
  is off the table in this case too.
- **Both yes:** fix resolution first (a higher-resolution decoder path), then a
  boundary loss or boundary head. FFT loss only as an ablation against them.

## Related

- `005-village-crossdate`: setup and baseline (`005.5`, 9.29 m @ 15 m); the
  birky-15 road-slide failure.
- `002-perfect-satellite-map`: map-side oracle on the 2025 flight (12.11 m,
  a tie with VGG11). `006.2` asks the same question on exact-truth ground.
- `pocs/mobilenet-lraspp-speed`: the model and its val ground.
