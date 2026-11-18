# 024-class-compatibility-matching — proposal

> Written **before** the first run. The success criterion below does not move
> once runs begin. If nine runs all miss it, the honest report says the
> approach failed.

## Question

Does scoring an offset by a class×class compatibility table counted from aligned cross-date pairs (024.2), or by the mutual
information of the two class maps (024.3), place the drone closer to the truth than `007.1`'s hand-weighted per-class NCC
on the same maps, frames and window?

## Hypothesis

Weighted NCC correlates each class with itself and sums with hand weights `0/1/1/0.5/0/0.5` set in the 002 proposal and never
tuned. It ignores cross-class evidence: drone "woody" over old-date "field", the field↔background edge, a building that one
date calls background. On the 25 % of draws whose query holds no weighted class it keeps the prior.

Mechanism, after Chan et al. (CVPR 2003; KL to an expected joint class histogram): count, on aligned pairs, how often drone
class i lands on satellite class j, against chance. The pointwise mutual information `W_ij = log P(i,j) / (P(i) P(j))` is the
per-pixel log-likelihood ratio of "aligned" vs "unrelated". Summed over the drone footprint at an offset, it scores the
offset with every class pair weighted by the data, at the cost of six convolutions (the same as now). Mutual information
(024.3) needs no counting at all: the joint class distribution at each offset is 36 convolutions of soft maps.

Both arms drop `class_weights`, `MIN_PRESENT_SHARE` and `MIN_WINDOW_STD`.

## Success criterion

**Metric:** `mean_position_error_m` (@ 10 m injected)
**Baseline:** `007.1` → 5.861 m (rect ×1.7, flat prior, no gate, weighted-ncc; `007-search-prior-and-gate/runs.jsonl`)
**Bar:** `024.2` **or** `024.3` reaches mean < 5.861 m **and** the paired frame-bootstrap 95 % CI of (arm − `024.1`) at 10 m
lies entirely below 0 (10,000 resamples, seed 42, `paired_frame_bootstrap`).

`024.1` must reproduce `007.1` (5.861 m ± 0.001) before any arm is read. If it does not, stop and report.

### Pre-registered runs

| Run | Matching | Role |
|---|---|---|
| `024.1` | `weighted-ncc`, weights 0/1/1/0.5/0/0.5 | control; must reproduce 007.1 |
| `024.2` | `pmi`: 6×6 W counted on the 46 train villages of 023's `village_split` (seed 42; none of birky / bushcha / dubrova) | primary |
| `024.3` | `mi`: mutual information of the soft class maps per offset, no data | primary |

Everything else equals `007.1`: model `pocs/mobilenet-lraspp-speed/model-lraspp-bn-satellite.pt` on both sides,
`village-crossdate-frames` (40 frames, 2 places), noise 5/10/15 m, 8 directions, seed 42, rect ×1.7, FOV 60°, 3 % satellite
crop, 0.385 m/px, `map_date` newest, ungated, flat prior.

### Fixed now

- **W estimation (`024.2`).** On `PairBank` aligned pairs from the 46 train villages (query `current`, map another date, as
  023), at the **true** offset: `P(i,j)` = mean over drone cells of `p_drone_i · p_sat_j` (soft), marginals from the same
  pairs. Additive smoothing 1e-4 on `P(i,j)` before the log. W is computed once, logged in the run config (36 numbers),
  and never fitted to the bench places.
- **Score (`024.2`).** `Σ_ij W_ij (drone_i ⊛ sat_j)` / number of drone cells, then z-scored as today.
- **Score (`024.3`).** `p_ij(o)` = mean over the drone window of `drone_i · sat_j(·+o)`; `q_j(o)` its satellite marginal;
  `p_i` the drone marginal; MI = `Σ p_ij log(p_ij / (p_i q_j))`, with 1e-8 floors; then z-scored.
- `coverage` is reported as NaN for both arms (no weighted classes exist).

## Requires

| Equipment | Status | Acceptance check |
|---|---|---|
| `uavloc.matching.compatibility`: `estimate_pmi`, `pmi_heatmap`, `mi_heatmap`, registered via `register_matching("pmi"/"mi")` | to build | (a) identity: satellite crop as its own query at 5 frames × 8 directions → argmax within 1 cell for both modes; (b) a W counted from pairs where the map is the *same* date has a clearly dominant diagonal; (c) `024.1` reproduces 007.1 |
| `uavloc.matching.correlator.register_matching` | exists (added 2026-10-19) | a registered copy of weighted-ncc gives an identical heatmap (checked) |
| `uavloc.data.match_pairs` (`village_split`, `PairBank`), `village-crossdate-holdout` | exists (023) | — |

**What only you can provide.** Nothing.

## Setup

| | |
|---|---|
| Dataset | `village-crossdate-frames` (primary), `village-crossdate-holdout` (diagnostic, `holdout_` prefix) |
| Split | W counted on 023's 46 train villages; test places excluded |
| Seed | 42 |
| Model | LR-ASPP satellite checkpoint, as 007.1 |
| Hardware | CPU, torch 2 threads |

**Diagnostics** (logged, never the criterion): 5 / 15 m, harm share, per place, the `blind` subset (no weighted class
under the hand weights) and `covered` subset, reachable-only mean, ms per correction, the W matrix as a figure, and the same
three runs on `village-crossdate-holdout`.

## What would falsify this

Both arms at ≥ 5.861 m at 10 m, or a CI touching 0. Then hand weights are not what limits matching on model masks, and a
learned matcher (023) has to beat W, not hand weights, to claim value from learning. If `024.2` wins only on `blind`
draws, report that separately: it is a coverage gain, not a better score.

## Related

- `007-search-prior-and-gate` (`007.1` baseline), `006` (GT masks: matching modes indistinguishable),
  `023-learned-correlation-head` (its arm W is the learned version of this).
- Research: `~/deep-research-output/semantic-mask-matching/phase5_synthesis/synthesis.md`, recommendation 1.
