# 025-a-contrario-gate — proposal

> Written **before** the first run. The success criterion below does not move
> once runs begin. If nine runs all miss it, the honest report says the
> approach failed.

## Question

Can an a contrario acceptance rule (apply the argmax only when its expected number of false alarms is below 1) match the
LOPO-tuned static accept-reject gate `022.2` on harm, applied share and mean error at 10 m, **with no parameter tuned on
any place**?

## Hypothesis

`022.2` reaches harm 0.00625 by tuning temperature, `s_max` and `c_min` leave-one-place-out, and the tuned values differ
between places (s_max 3.0 m bushcha vs 5.0 m birky). 022 showed per-frame learned gates do not transfer. The z-score is
per-surface, so it carries no absolute evidence: a uniform-woody surface and a crisp one both get unit variance.

A contrario (Desolneux, Moisan & Morel; ORSA, Moisan & Stival IJCV 2004) asks each frame's own null: how high does this
drone map score against satellite ground where it certainly is not? With `N_tests` = number of cells searched,
`NFA = N_tests · P_null(raw score ≥ observed peak)`. Accept if `NFA < ε = 1`, else keep the prior. A flat scene scores high
against the null too, so it is rejected automatically. Nothing is fitted, so nothing has to transfer.

## Success criterion

**Metric:** `harm_share`, `accept_share`, `mean_position_error_m` (all @ 10 m injected)
**Baseline:** `022.2` → harm 0.00625 (2 / 320 draws), accept share 0.55625, mean 5.99884 m (`022-adaptive-gate/runs.jsonl`)
**Bar:** `025.2` **or** `025.3` meets **all three**: harm ≤ 0.00625, accept share ≥ 0.55625, mean ≤ 5.99884 m, with ε = 1 and no
LOPO tuning.

`025.1` must reproduce `022.2` (5.99884 m ± 0.001, harm 0.00625) first.

### Pre-registered runs

| Run | Decision | Null windows | Role |
|---|---|---|---|
| `025.1` | static accept-reject, LOPO-tuned as 022.2 | — | control |
| `025.2` | NFA < 1 | M = 200 satellite windows from the 46 `village_split` train villages (023), random pose, newest date | primary |
| `025.3` | NFA < 1 | M = 200 windows from the test place's own map, centres ≥ 60 m from the prior | primary |

Surfaces, model and frames as `022.2` (weighted-ncc, LR-ASPP satellite checkpoint, `village-crossdate-frames`, 5/10/15 m,
8 directions, seed 42, rect ×1.7).

### Fixed now

- **Raw score:** the weighted mean of per-class NCC **before** z-scoring (`class_ncc_maps` weighted as
  `weighted_ncc_heatmap`), on the actual surface and on every null window, with the same drone map and the same window size.
- **Null distribution:** per frame, pool every cell of the M null surfaces; `P_null(s)` = share of pooled cells ≥ s.
- **N_tests:** number of finite cells in the actual surface.
- **Decision:** NFA < 1 → apply the argmax move; else keep the prior (`reject`). No across-only.
- **ε = 1 is the run.** An ε sweep {0.01, 0.1, 1, 10} is logged as a diagnostic curve only, never chosen from.

## Requires

| Equipment | Status | Acceptance check |
|---|---|---|
| `uavloc.matching.acontrario`: raw-score surface, null-window sampler (both kinds), `nfa`, `score_nfa_draws` (rows in `score_draws`' format plus `nfa`) | to build | (a) identity bench (satellite as own query, ungated): NFA < 1 on ≥ 95 % of reachable draws; (b) drone map scored against a window from *another* village as the "true" one: NFA < 1 on ≤ 5 % of draws (false-alarm check of the null); (c) `025.1` reproduces 022.2 |
| `uavloc.matching.benchmark`, `uavloc.matching.tuning`, `uavloc.data.match_pairs.village_split` | exists | — |

> **Amendment 2026-10-20 (acceptance check (a) only; success criterion unchanged).** Check (a) as first written counted
> all reachable identity draws. 102 of 528 are "blind": the query holds no weighted class, the raw surface is all zeros, and
> NFA = N_tests, so no acceptance rule can accept them (`decide()` rejects them too). That was an error in how the check was
> specified, not in the equipment. Check (a) now reads: **NFA < 1 on ≥ 95 % of reachable non-blind identity draws**.
> Measured before this amendment: 426 / 426 = 1.0 for both nulls; all-reachable share 0.807 (`artifacts/checks.json`).
> Blind draws stay in every run's metrics as rejects.

**What only you can provide.** Nothing.

## Setup

| | |
|---|---|
| Dataset | `village-crossdate-frames`; `village-crossdate-holdout` as a diagnostic (`holdout_` prefix) |
| Seed | 42 (null sampling seed 42 as well) |
| Model | LR-ASPP satellite checkpoint, as 022.2 |
| Hardware | CPU, torch 2 threads |

**Diagnostics:** accept share and harm **per place** (birky vs bushcha; the transfer question), 5 / 15 m, p90 / max error,
the ε curve, NFA distribution of harmful vs helpful draws, ms per correction including the null.

## What would falsify this

Both arms miss at least one bar at ε = 1. If they miss only on accept share, the null is too strict for this score
(weighted-NCC surfaces are spatially correlated, so `N_tests` over-counts independent tests); report that, do not retune ε.
If per-place accept shares still differ as much as 022's tuned thresholds did, scene dependence is not removed by a
frame-own null either.

## Related

- `022-adaptive-gate` (`022.2` baseline; risk gate falsified), `007` (static gate, τ per village).
- Research: `~/deep-research-output/semantic-mask-matching/phase5_synthesis/synthesis.md`, recommendation 2.
