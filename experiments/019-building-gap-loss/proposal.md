# 019-building-gap-loss — proposal

> Written **before** the first run. The success criterion below does not move
> once runs begin. If nine runs all miss it, the honest report says the
> approach failed.

## Question

Does up-weighting the cross-entropy on background pixels that separate two nearby
buildings (a U-Net-style gap weight) during an LR-ASPP fine-tune cut the share of
buildings the model merges, among those a perfect stride-8 map keeps separate,
by at least 25 % relative to a control fine-tune?

## Hypothesis

LR-ASPP merges 29.7 % of the val buildings (`pocs/building-merge-stride`). About
half of those merges (263 of 595) are a resolution limit: the gaps are under
~2 m, and even the label averaged into 8 x 8 cells merges them. The other half
(329) are model errors. The perfect stride-8 map separates every building more
than 3 m apart, yet LR-ASPP merges 45 % of buildings with a 3–4 m gap and 17 %
with a 4–6 m gap.

The reason is that plain CE barely notices them. A 3 m gap between two roofs is
a few hundred background pixels next to thousands of roof pixels, so filling it
costs almost nothing. Weighting exactly those pixels, as Ronneberger et al.
(2015) did for touching cells, makes a filled gap expensive. It targets the
separable merges and leaves roof interiors and building edges against open
background alone. That is the difference from 016, which up-weighted every
class edge equally and gained +0.005 Boundary IoU.

Stated limits:
- **The loss cannot split sub-cell gaps.** The 44 % of merges that are
  resolution-bound are excluded from the primary metric, not counted as a
  failure.
- **It may shrink all buildings to open the gaps.** The *missed* share and
  building IoU are guarded for that reason.
- **It does not find missing buildings** (24.8 % of val buildings).

## Success criterion

**Metric:** `separable_merge_rate_mean`. For each val set, take the labelled
buildings that the perfect stride-8 map (bilinear, `label_at_stride`) keeps
separate. The metric is the share of those that the model merges, as the
unweighted mean over `birky-strip`, `birky-strip-google` and `bushcha`. The
building definitions are the PoC's, moved to `uavloc`: 4-connected components
of at least 16 px. A building is covered when a component overlaps at least
20 % of it, and merged when a component that covers it also covers another
building.

**Baseline:** the **w₀ = 0 control run** of this experiment: same
initialisation, steps, schedule, sampler stream and seed. It is not the original
checkpoint, because in 015/016 fine-tuning alone moved the boundary numbers by
+0.004. Run `019.1` re-scores the original checkpoint as a reference. The PoC
estimate for it, pooled over the sets, is 329 / 1,467 ≈ 0.22. That is an
estimate, not the baseline.

**Bar:** the best w₀ run at seed 42 has `separable_merge_rate_mean` ≤
0.75 × control.

**Guards.** The run must also pass all three, or it does not count:
- `miou_mean` ≥ control − 0.01
- `building_missed_share_mean` ≤ control + 0.02 (it must not open gaps by
  dropping buildings)
- building IoU (mean over sets) ≥ control − 0.02 (it must not shrink every roof)

**Replication:** the control and the best w₀ are re-run at seed 7. The claim is
"confirmed" only if the seed-7 relative reduction is also at least 10 %. w₀ is
picked on the val sets from two values, so a single-seed pass carries selection
bias.

**Also reported:**
- the full merged / missed / separate shares, the resolution-bound merge rate
  (expected flat) and merged share by gap bin
- Boundary IoU, BF1 and mean Dice
- the gap-band share of labelled training pixels
- model latency, unchanged by construction

## Registered runs

| Run | What | Role |
|---|---|---|
| `019.1` | original checkpoint, no fine-tune | reference |
| `019.2` | fine-tune, w₀ = 0, seed 42 | **control / baseline** |
| `019.3` | w₀ = 3, seed 42 | |
| `019.4` | w₀ = 10, seed 42 | |
| `019.5` | control, seed 7 | replication |
| `019.6` | best of 019.3 / 019.4, seed 7 | replication |

## Requires

| Equipment | Status | Acceptance check |
|---|---|---|
| `uavloc.segmentation.instances` (building components, overlap, separate/merged/missed status, nearest gap), moved from the `pocs/building-merge-stride` notebook on second use | to build | re-running the PoC notebook on the module reproduces its tables exactly (595 merged, 263 / 329 split) |
| `uavloc.segmentation.losses.gap_band`, `gap_weighted_ce` | to build | on real training crops: w₀ = 0 gives exactly `ce_dice_loss`; the band covers background pixels only, lies between two different buildings, and is a small share of labelled pixels (reported); figure of band overlays on 4 crops |
| `builds/seg-boundary-eval`, `pocs/village-seg-baseline/village_seg.py`, `model-lraspp-bn-satellite.pt` | exist | `019.1` reproduces 0.4758 mIoU / 0.1232 BIoU |

**What only you can provide.** A go-ahead on the bar and the guards. Then you
run the notebook.

## Setup

| | |
|---|---|
| Dataset | `ua-village-annotated`: train = birky (rows below the strip plus a 64 px gap, 5 dates) and dubrova (3 dates); val = the harness's three sets. The same as 016 |
| Split | as `probe.ipynb` / round 3 (`vs.ban_holdouts`), unchanged |
| Seed | 42 (sampler RNG and torch); replication seed 7 |
| Model | LR-ASPP MobileNetV3-L from `model-lraspp-bn-satellite.pt`, satellite-only fine-tune. BN is re-estimated at the end as in 016 |
| Loss | `mean(w · CE) + Dice`, with `w = 1 + w₀ · gap_band`. `gap_band` = labelled non-building pixels within 10 px (chessboard, ~3.9 m) of **two different** building components, which targets gaps up to ~7.7 m, the range where merges are model errors. Components are computed per training crop, so the band follows the augmentation. w₀ ∈ {0, 3, 10} |
| Training | the same as 016: 400 steps, batch 4 × 512 px, `SatelliteSampler(rare_p = 0.3)`, AdamW lr 1e-4, OneCycle with 5 % warm-up, wd 1e-4 |
| Hardware | training on MPS; evaluation and latency on CPU |

## What would falsify this

If neither w₀ cuts the separable merge rate by at least 10 % relative to the
control at seed 42, a pixel-weighted CE does not change where LR-ASPP puts
building boundaries between neighbours. The direction is then dropped rather
than widened (a larger radius, larger w₀). The separable merges would then need
a model change, such as instance or contact outputs, not a loss. If the merge
rate drops but the missed share or building IoU guard fails, the loss separates
buildings by erasing them, and that is reported as the result.

## Related

- `pocs/building-merge-stride`: the merge/resolution split this experiment targets.
- `016-boundary-loss`: equal weight on all class edges, +0.005 BIoU. The same recipe and harness.
- `015-boundary-head`: a boundary output head; no effect on the semantic edges.
- `017-gray-logit-refiner`: a post-hoc refiner; no gain (report pending).
