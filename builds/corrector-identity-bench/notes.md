# corrector-identity-bench

**Status:** ready for your review · **Unblocks:** any change to `PositionCorrelator`

> Status is one of: `planning`, `blocked`, `ready for your review`,
> `accepted`, `parked`. It is parsed from this line — keep the format.

## What this is

The corrector matched against itself: the "drone" map is the satellite's own
feature map, cut heading-up at the frame's true pose, correlated against the
same raster. Segmentation error, date gap and sensor gap are all removed by
construction, so whatever error is left belongs to the corrector — window reach,
rotation resampling, match-grid quantisation, peak choice, gate.

It is the regression floor for corrector work: these numbers should not get
worse when the corrector changes, and a new matcher that cannot reproduce them
is broken before it reaches real imagery.

Same test as the models benchmark, deliberately: `village-crossdate-frames`
(40 frames, 20 per place), 8 bearings per frame, 5 / 10 / 15 m injected,
`weighted-ncc`, the 006.1 / 007.1 / 021 correlator setup. Keeping the frames
identical is what makes the floor comparable to `006.1` (4.59 / 5.86 / 9.29 m).

Each draw is scored twice from one set of surfaces, via `score_draws`:
**ungated**, and **gated** with the 007.8 values (`gate: full`, `s_max 3 m`,
`c_min 0.01`, `T 0.25`).

## What I needed from you

Nothing. `SatelliteAsDroneCorrelator` went into `uavloc/matching/identity.py`
beside `SatelliteAsDrone`, which the position-correction harness already used —
the harness version bakes one frame's pose into the extractor, so it cannot run
over a frame list. The subclass overrides `drone_map` instead and works with
`benchmark_draws`.

## Does it work

Run: `conda activate uavloc && $CONDA_PREFIX/bin/papermill builds/corrector-identity-bench/bench.ipynb builds/corrector-identity-bench/runs/bench.ipynb --cwd .`
(about 3 min on MPS, from `src/`). Bare `papermill` is the Python 3.12 copy and
dies on NumPy 2 — use `$CONDA_PREFIX/bin/papermill`.

**The corrector's geometry is sound.** Ungated, on the 32 frames that have
matchable content, when the truth is inside the search window:

| injected | reachable | median | mean | p90 | max |
|---|---|---|---|---|---|
| 5 m | 100 % | 0.167 | 0.162 | 0.237 | 0.424 |
| 10 m | 56 % | 0.155 | 0.157 | 0.230 | 0.501 |
| 15 m | 10 % | 0.173 | 0.179 | 0.325 | 0.392 |

Sub-pixel: the match grid is 0.385 m/px, so 0.17 m is under half a cell, flat
across injected levels, and the worst draw in 960 is 0.50 m. This agrees with
the position-correction harness's identity check (0.14 m median, max 0.27, on 5
frames). No correction harms the prior at 5 or 10 m.

Three things bound the result, and none of them is the matching:

- **Window reach.** The truth is inside the window on 100 % of draws at 5 m,
  **56 % at 10 m and 10 % at 15 m**. Unreachable draws are most of the error at
  15 m (ungated median 3.28 m over all draws, harm 4.3 %). This is the 005/006
  window bound again, now measured with everything else perfect.
- **8 of 40 frames are blind.** `birky-14` and 7 bushcha frames
  (`bushcha-02/04/05/08/10/13/17`) hold no building, track, woody or water at
  all, so no weighted class survives `MIN_PRESENT_SHARE` / `MIN_WINDOW_STD`,
  `peak_z` is 0 on every draw and the prior is kept. That is the dataset card's
  known problem, not a corrector fault, but it puts 20 % of draws at exactly the
  injected error and inflates every mean that includes them. `blind_frames()`
  splits them out; headline numbers are the content frames only.
- **The 007.8 gate costs 1–2.5 m on a match it should accept trivially.**
  Mean error when reachable, gated vs ungated: **1.18 vs 0.16 m** at 5 m,
  **2.35 vs 0.16** at 10 m, **2.72 vs 0.18** at 15 m. The gate rejects ~21 % of
  draws and sends ~27 % "across" (ridge-constrained) on a *self-match*, where
  the correct answer is sitting at a sub-pixel peak. It was tuned to suppress
  confident wrong peaks on cross-date imagery and it cannot tell this case
  apart.

## Follow-up: is the gate worth it? (`gate-roc.ipynb`)

The bench prices only the gate's cost, so this pairs it with real cross-date draws —
7,485 decidable draws from 021 (9 models), gated rows, every draw scored under each
policy (accepted takes the argmax, blocked keeps the prior).

| policy | mean | median | p90 | max | harm | applied |
|---|---|---|---|---|---|---|
| accept all (no gate) | **5.56** | 3.14 | 14.48 | 31.01 | 16.0 % | 100 % |
| s1 gate (007.8) | **6.56** | 5.00 | 15.00 | 31.01 | **4.9 %** | 58 % |
| keep prior always | 10.00 | 10.00 | 15.00 | 15.00 | 0 % | 0 % |

**The gate costs 1.0 m of mean error and cuts harm 3x.** Which one is "better" is
entirely a choice of metric — and `p90` and `max` are *identical* with and without it.
It reduces how often harm happens, never how bad it gets, because the worst draws are
confident wrong peaks with small `s1`. That is 007's own finding ("no per-frame gate
can see them"), now measured.

**Do not score a gate on the mean over its accepted draws.** That conditional mean is
4.08 m here, which flatters it by dropping the hard draws it refused instead of
charging it for keeping their prior. `policy_stats()` scores every draw.

### Which statistic should the gate use

AUC for detecting a harmful argmax, same draws:

| statistic | AUC | false alarm at 007.8's detection |
|---|---|---|
| **combined** (5 feat, logistic, out-of-fold, grouped by frame) | **0.767** | **30.2 %** |
| `s1_m` | 0.717 | 36.5 % |
| `second_peak_ratio` | 0.705 | 37.1 % |
| `s2_m` | 0.679 | 44.3 % |
| `-peak_z` | 0.658 | 50.2 % |
| `-coverage` | 0.456 | 92.1 % |

- **`s1` is already the best single statistic.** Swapping it for `second_peak_ratio`
  is a wash (36.5 % vs 37.1 % false alarm) — that idea is dead.
- **`coverage` is below chance (0.456)**, which is why its condition never fires on
  the identity bench and never fired here.
- **The threshold is near-optimal too:** Youden on the ROC gives 2.82 m against the
  tuned 3.0 m. Retuning buys nothing.
- **Combining is the only real lead:** 0.717 -> 0.767, false alarm 36.5 % -> 30.2 %
  at matched detection.

**The identity bench cross-validates this.** It measured a 35 % false-alarm rate with
zero wrong peaks present; real data gives 36.5 %. The bench's estimate is sound, which
is the argument for keeping it.

**Gotcha:** `s1_m`/`s2_m` depend on `config.temperature`, so they are comparable only
within one rule — `ungated` rows are logged at the default T = 1.0 and `gated` at
T = 0.25. Applying the 3 m threshold to ungated `s1` makes the gate look like it
accepts 0.3 % of draws.

## Follow-up: per-class NCC vs one whole-mask correlation (`matching-ab.ipynb`)

`weighted-ncc` was inherited from 002, where it beat `cosine` on a hand-labelled map
(12.11 vs 14.49) — but in 001 it lost on model maps (19.89 vs 15.04), and the current
stack never re-tested it. This is that A/B: 006.1 setup, ungated, same model, same
frames, only `matching` differs.

| injected | matching | mean | median | p90 | max | harm |
|---|---|---|---|---|---|---|
| 5 m | **weighted-ncc** | **4.59** | 4.50 | **10.96** | 18.57 | **20.0 %** |
| 5 m | cosine | 6.45 | 5.27 | 14.59 | 21.03 | 37.2 % |
| 10 m | **weighted-ncc** | **5.86** | 3.83 | **10.00** | 19.27 | **8.4 %** |
| 10 m | cosine | 6.90 | 4.44 | 18.56 | 25.94 | 18.8 % |
| 15 m | **weighted-ncc** | **9.29** | 7.48 | **16.81** | 30.80 | **9.7 %** |
| 15 m | cosine | 9.76 | 6.62 | 23.32 | 30.80 | 22.8 % |

**Keep per-class NCC.** It wins in every place x noise cell. The mean difference is only
CI-clear at 5 m (cosine +1.87 m, CI 0.71 to 3.10; at 10 and 15 m the CI crosses 0), but
**harm roughly doubles with cosine at every level** and p90 is 3-8 m worse — consistent
across all six cells.

Mechanism: per-class normalisation lets a building corner outvote a treeline covering
about six times more pixels. A single joint correlation is dominated by woody (~64 % of
pixels), so the peak slides along vegetation edges. That is why the gap appears in the
tail and the harm rate rather than the median.

**Control reproduces exactly:** weighted-ncc 9.292 m @15 m = `006.1`, 5.861 m @10 m =
`007`'s best.

### On GT masks the advantage disappears

`MASK_SOURCE` switches both sides between the segmenter and the hand labels
(`annotated`). cosine minus weighted-ncc, mean error with a 95 % paired frame bootstrap:

| masks | 5 m | 10 m | 15 m |
|---|---|---|---|
| model | **+1.87** (0.71, 3.10) | +1.04 (-0.13, 2.22) | +0.47 (-0.88, 1.81) |
| GT (`annotated`) | **-0.13** (-1.02, 0.63) | **-0.26** (-1.09, 0.28) | **-0.41** (-1.25, 0.32) |

On GT the sign flips to cosine and every CI crosses 0: **the two modes are
indistinguishable on clean masks.** So per-class NCC is absorbing model noise in sparse
classes, not exploiting class structure.

- **Keep weighted-ncc today** — we run on model masks, where the margin is real.
- **Re-test after any real segmentation gain**, because the margin shrinks toward zero.
- **Most harm is segmentation error, not matching:** harm falls from 20/8.4/9.7 % to
  2.5/1.9/7.8 % at 5/10/15 m when the masks are clean.

Caveat: GT on both sides is near a self-match (the query is a rotated crop of the raster
the map side uses), so it is the clean extreme rather than an operating point — reality
sits between the two arms. Its median error at 5 m, 0.197 m, agrees with the identity
bench floor of 0.17 m.

## Your call

The gate result is the one worth acting on. On this bench the gate is pure
cost — it can only throw away correct answers — so the bench cannot say the gate
is wrong in the field, where 007.8 cut harm 0.084 → 0.022. But it does say the
gate's accept condition is far too strict for high-confidence matches, and that
there is room for a confidence band that passes them straight through.

Two follow-ups this bench does not cover, in the order I would do them:

1. **Widen the ground.** 40 frames over 2 places, 8 of them blind, and birky's
   20 overlap on a ~270 m strip. `ua-village-multidate` has 59 villages and the
   seg models only trained on dubrova + birky below row 704, so ~57 are free.
   Rebuilding `village-crossdate-frames` over them would give a few hundred
   non-overlapping frames — at the cost of comparability with every existing
   run, which is why I did not do it here.
2. **Add the damage ladder.** 006's `damaged()` (stride 2–16, boundary r 1–8)
   turns this from a pass/fail floor into a difficulty curve, which is what
   would actually rank two correctors against each other. A self-match has no
   appearance gap, so on its own it catches regressions, not quality.
