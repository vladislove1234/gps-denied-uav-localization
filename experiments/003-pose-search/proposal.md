# 003-pose-search — proposal

> Written **before** the first run. The success criterion below does not move
> once runs begin. If nine runs all miss it, the honest report says the
> approach failed.

## Question

Does letting the correlator search heading (and footprint scale), instead of
trusting the logged pose, lower the position error of weighted class-map
matching on the hand-labelled map at 15 m injected error?

## Hypothesis

It will. The correlator searches translation only and takes heading and
footprint size from the log. `pocs/pose-reregistration` found the log wrong on
both counts:
- Heading is off by a constant per pass: −9° in block 0, **+26.5° and +25° in
  blocks 1–2**, +8° in block 3, −3° in block 4. Median |error| is 10°.
- The footprint is about 0.6× what the logged altitude implies.

A linear feature matched at the wrong angle scores low everywhere, and the
argmax falls wherever part of it happens to overlap. That is the failure seen
in `002-perfect-satellite-map/walkthrough.ipynb`, step 7. Blocks 1–2 are
exactly where `002.2` lost to VGG11. Searching the heading lets the track line
up, and searching the scale lets its width and spacing match.

## Success criterion

**Metric:** `mean_position_error_m` (@ 15 m injected; 44 frames × 8 directions, seed 42, same draws as 001/002)
**Baseline:** `002.2` → **12.113 m** (hand-labelled map, round-3 drone side, weighted NCC, translation-only; `002-perfect-satellite-map/runs.jsonl`)
**Bar:** `003.1` **or** `003.2` reaches `mean_position_error_m` **< 12.113 m**

The bar measures the effect of the search on the method it is meant to fix,
against that method's own translation-only run. Whether class-map matching
then beats VGG11 is a separate question. It is reported below as a diagnostic
and is **not** the criterion.

### Pre-registered runs

| Run | Method | Matching | Heading search | Scale search | Role |
|---|---|---|---|---|---|
| `003.1` | `segmentation-labels` | `weighted-ncc` | yes | no (scale 1.0) | primary |
| `003.2` | `segmentation-labels` | `weighted-ncc` | yes | yes | primary |
| `003.3` | `vgg11` | `cosine` | yes | no | control |
| `003.4` | `vgg11` | `cosine` | yes | yes | control |

Everything not listed matches `002.2` and `001.1`: frames, draws, seeds,
correlator geometry (scale 1.7, FOV 60°, 3% satellite-border crop), the
round-3 drone checkpoint, the label map `ua-ortho-annotated/gonchary-2025` /
`ortho`, and the weights background 0, building 1, track 1, woody 0.5,
field 0, water 0.5.

### The search, fixed now

Per draw, around the injected prior. This is the same schedule as
`pocs/pose-reregistration`:

1. **Coarse heading:** offsets −36° … +36° in 3° steps (25 values), added to
   the logged heading, at scale 1.0. The range covers the largest per-pass
   correction the PoC found (+33°) with one step to spare.
2. **Scale** (runs `.2` and `.4` only): at the best coarse heading, footprint
   scales {0.5, 0.6, 0.7, 0.8, 0.9, 1.0}. Scale multiplies the drone footprint
   (w, h) from the logged altitude. The drone side is resampled to that
   footprint; the satellite patch does **not** change size.
3. **Fine heading:** ±3° in 1° steps around the coarse best, at the chosen
   scale.
4. **Winner:** the (heading, scale, shift) with the highest **raw** score over
   all evaluated hypotheses. `peak_z` is then computed on the winning surface,
   exactly as today.

Raw scores must be comparable across hypotheses:
- `weighted-ncc`: the weighted mean of per-class NCC before z-scoring. It is
  already scale-free.
- `cosine`: the sum of per-pixel unit-vector dot products **divided by the
  number of drone-map cells**, i.e. mean cosine. Without this, a larger kernel
  wins by being larger.

> **Amendment 2026-10-20 (search schedule for the scale runs `003.2` and `003.4` only; success criterion unchanged; approved by
> the user before any scale run existed).** Acceptance (b) failed at true scale 0.6: 31/40 draws < 1 m. The greedy order (coarse
> heading at scale 1.0, then scale, then fine heading) committed to a wrong coarse heading (−24° instead of −20°) on 8 draws and never
> evaluated the true pose, which scores higher than the chosen one in all 8 (`artifacts/checks.json`, `true_pose_beats_winner`).
> New schedule for runs with scale search:
> 1. **Joint coarse pass:** every heading offset −36° … +36° in 3° steps × every scale {0.5, 0.6, 0.7, 0.8, 0.9, 1.0} (150 hypotheses).
> 2. **Fine heading:** ±3° in 1° steps around the coarse best heading, at the coarse best scale.
> 3. **Winner:** unchanged (highest raw score over all evaluated hypotheses).
>
> Heading-only runs (`003.1`, `003.3`) keep the registered schedule. With a single scale it is identical. `003.1` was logged
> before this amendment. Acceptance (b) is re-run with the new schedule before `003.2` / `003.4`. Cost is ~4× the registered
> schedule per draw (156 vs ~38 hypotheses). The pose-reregistration PoC used the greedy order, so its scale ≈ 0.6 finding
> may carry the same bias.

> **Amendment 2026-10-20 (2) (scale runs `003.2` / `003.4` and acceptance (b); success criterion unchanged; approved by the user).**
> With the joint schedule, (b) at true scale 0.6 reached 34/40 (`checks.json` → `identity_joint_schedule`). 4 draws: the true pose
> was never evaluated by the 3° coarse grid but scores higher than the winner (search miss). 2 draws: the true pose was evaluated
> and a pose 8–11 m along the same straight track scores marginally higher (map ambiguity, not a search fault).
> Changes:
> 1. **Fine pass around the top 3 coarse (heading, scale) pairs**, not only the best: ±3° in 1° steps at each pair's scale.
>    The winner is still the highest raw score over every evaluated hypothesis.
> 2. **Acceptance (b), scale cases:** pass = every draw where the true pose outscores every evaluated non-true pose is recovered
>    (heading ±1°, scale exact, position < 1 m). Draws where some other pose outscores the true pose are counted and reported as
>    map ambiguity, not as search failures. The heading-only cases keep the original wording (40/40 required).
>
> Heading-only runs (`003.1`, `003.3`) are unaffected. Registered 003.2 / 003.4 will be logged as 003.3 / 003.4 by `log_run`
> (registered 003.3 was logged as 003.2). `registered_run` in each config is authoritative.

Search area:
- **The satellite patch stays 1.7 × the logged footprint** (scale 1.0) in
  metres, rotated to the hypothesised heading. The map area available does not
  depend on the hypothesis. A smaller scale therefore gives a slightly larger
  translation range.
- **`reachable` keeps its 001 definition** (truth inside the scale-1.0
  translation range at the logged heading), so `reachable_share` stays
  comparable with 001/002.

The corrected position is the drone centre at the winning shift, rotated back
with the **winning** heading (logged + offset), not the logged one.

## Requires

| Equipment | Status | Acceptance check |
|---|---|---|
| Pose search in `uavloc.matching.correlator`: `CorrelatorConfig.heading_offsets_deg`, `CorrelatorConfig.scales`, the coarse→scale→fine schedule above; `Correction` gains `heading_offset_deg` and `scale` | to build | **(a) No regression:** with `heading_offsets_deg = (0,)` and `scales = (1.0,)`, 12 sampled draws each of `001.1` and `002.2` replay within 1e-6 m. **(b) Identity:** a label-map crop at the true position with heading + 20° as the "drone" map; the search returns an offset of −20° ± 1° and a position < 1 m off, on 5 frames × 8 directions. Also do this with a crop at scale 0.6 and check the scale is recovered. |
| Mean-cosine raw score for `cosine` matching across scales | to build | On one VGG11 draw, the raw score does not increase monotonically with kernel size when the satellite side is pure noise. |
| Drone features at a scaled footprint: both `Vgg11Features.drone` and `SegmentationFeatures.drone` take the scaled `(w, h)`. The drone-map cache must key on (frame, scale). | to build | The drone-map shape for scale 0.6 is 0.6 × the scale-1.0 shape, ±1 cell. |
| `run_benchmark` records per draw: `heading_offset_deg`, `scale` | to build | Present in `records-003.*.csv`. |
| `uavloc.features.factory.correlator_config` reads `heading_offsets_deg` and `scales` from the run config | to build | 003's `train.ipynb` builds everything through it; the logged config equals what ran. |
| `experiments/003-pose-search/train.ipynb` | to build | Copy of `002-perfect-satellite-map/train.ipynb` plus `HEADING_SEARCH` / `SCALE_SEARCH` params. `parameters` tag on the Params cell. `mkdir -p artifacts` before papermill. |
| `uavloc.features.factory`, the hand-labelled map, the round-3 checkpoints, `builds/position-correction-harness`, `builds/perfect-map-matching` | exists | — |

**Compute budget, measured in the PoC:** about 0.5 s per weighted-NCC
evaluation on this machine, and the schedule is ~35 evaluations per draw
(25 + 6 + 7, minus overlap). That makes one segmentation run about
**15–20 s per draw → ~5 h for 1,056 draws (3 noise levels)**. VGG11 runs add a
network forward per heading and scale. If this is too slow:
- Allowed: FFT-based correlation, or running the NCC on MPS. These change
  speed, not results; check them with acceptance (a).
- **Not allowed:** cutting noise levels, directions or search values. That is
  a config change and needs a dated amendment here, before the run.

**What only you can provide.** Nothing. All inputs are on disk.

> Building must not move the success criterion above. If the build reveals the
> metric is unmeasurable as promised, amend this file with a dated note saying
> what changed and why. Do not quietly swap it.

## Setup

| | |
|---|---|
| Frames | `drone-2025-easy`, all 44 |
| Noise | 15 m primary; 5 and 10 m logged. 8 directions per frame, seed 42. Same draws as 001/002. |
| Satellite (VGG11) | 2025 orthophoto, as `001.1` |
| Satellite (class maps) | `ua-ortho-annotated/gonchary-2025` / `ortho`, one-hot, 0.385 m/px, as `002.2` |
| Drone side (class maps) | `model-round3-bn-drone.pt` at 0.10 m/px |
| Hardware | MPS |

**Diagnostics** (logged, never the criterion):
- Everything 002 logs: the three subsets, per-block means, improved share,
  5/10 m, peak z, bias east/north.
- **Heading found vs `pocs/pose-reregistration`:** the median |difference|
  between the chosen heading offset and the PoC's per-frame correction, per
  block. It shows whether the search finds the heading the map implies, or
  just a different false peak.
- **Head-to-head:** `003.2 − 003.4` (class maps vs VGG11, both with search),
  with a frame-bootstrap 95% interval as in the 002 report.
- The distribution of chosen scales, compared with the PoC's median of 0.6.

## Known confounds, accepted up front

- **The search range is informed by the test frames.** ±36° and scales down
  to 0.5 were picked after the PoC measured corrections on these same frames.
  A range chosen with no knowledge of the flight would probably be similar
  (±30–45°), but this biases in favour of the search. State it in the
  report.
- **More hypotheses, more false peaks.** On frames with little structure, a
  wider search can find a better-scoring wrong pose. That is part of what is
  measured; the per-block breakdown should show where it happens.
- **The benchmark truth is still the logged GPS position.** The PoC could not
  check it (36 m median along-track ambiguity). Errors in the truth cap what
  any method can show.
- **The drone side trained on 22 of the 44 frames** (as 001/002). The
  held-out subset is logged.
- **The PoC corrections are not used anywhere in the runs.** They came from
  the same hand map this experiment scores against. Using them as priors or
  truth would be circular. They appear only in the diagnostic comparison
  above.

## What would falsify this

Both `003.1` and `003.2` end at ≥ 12.113 m. Then the logged heading and scale
are not what limits class-map matching here, despite the PoC's evidence that
they are wrong. The remaining error would lie in the benchmark truth, the drone
class maps, or the correlator's scoring, and a pose search is not worth its
compute cost.

If only the controls improve (`003.3`/`003.4` < 11.634 m while `003.1`/`003.2`
miss), the search helps texture features but not class maps. Report that too.

## Related

- `001-seg-map-correction`: baseline `001.1` (VGG11, 11.634 m), the harness.
- `002-perfect-satellite-map`: baseline `002.2` (12.113 m); `walkthrough.ipynb`
  steps 2b and 7 show the rotated track.
- `pocs/pose-reregistration`: per-pass heading corrections, scale ~0.6, the
  along-track ambiguity; `results.csv` per frame.
