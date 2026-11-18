# 035-loss-arms-rerun — proposal

> Written **before** the first run. The success criterion below does not move
> once runs begin.

## Question

Are the `033` system results attributed to the right loss arm? Re-run from the saved `033`
artifacts, with each arm's segmenter, head and T named explicitly in the run config: does each
arm reproduce its recorded `033` end error, flight by flight?

## Hypothesis

The concern is that the `033` numbers are swapped between arms (that the class-specific loss in
fact gave the lower end error). The records are internally consistent, so a swap could only come
from code: the arm → checkpoint / head / T mapping in `run_all.sh`, `tune.ipynb`, `train.ipynb`,
`train_head.ipynb`, or the flights CSV / `runs.jsonl` attribution.

Expectation: no swap. `run_flights` is deterministic per route (INS and particle seeds come from
the route), and `034.2` / `034.3` already re-ran `033.1` / `033.4` from the same artifacts and
matched them to the third decimal. The building and structure arms were never re-run; this
experiment covers them.

## Success criterion

This is a verification, not an improvement experiment. No retraining.

**Metric:** `mean_end_error_m` over the 94 test flights of `sim-flights`, normal start, filter A
(particles) + the arm's own head, T = the arm's tuned value.

**Reproduction (each arm against its `033` run):**
- `035.1` control vs `033.1` (9.256 m), `035.2` building vs `033.2` (11.218 m),
  `035.3` structure vs `033.3` (11.573 m).
- **Reproduced** if |Δ mean| ≤ 0.01 m and the per-flight end errors match (max |Δ| ≤ 0.01 m).
- **Swap** if an arm's per-flight end errors match a *different* `033` arm's.
- Anything else is a non-determinism or a code change since `033`; it is explained in the report,
  not read as a swap.

**Independent segmenter check (audit notebook).** Each saved
`segmenter-<arm>/model-bn-satellite.pt` is loaded and scored on the round-3 satellite val sets
with `uavloc.segmentation.village_seg.val_metrics` (the `021` metric code). The loss leaves a
fingerprint: building and structure lower `building_missed_share_mean` by > 0.03 against
control, structure raises `track_skeleton_recall_mean` by > 0.03. The checkpoint that shows the
fingerprint is the one trained with that loss, whatever its folder name.

**Reported:** median end error per arm, paired village bootstrap (10,000 resamples, seed 42)
building − control on the `035` flights, as in `033`.

## Requires

| Equipment | Status | Acceptance check |
|---|---|---|
| `033` artifacts: `segmenter-<arm>/model-bn-satellite.pt`, `head-<arm>.pt`, `tuning-<arm>/` | exist | audit: files distinct (SHA-256), metrics match `segmenter-<arm>/metrics.json` (±0.005, CPU vs MPS) |
| `uavloc.simulation.arms`, `uavloc.simulation.sequence` | exist | `035.1` reproduces `033.1` |
| `uavloc.segmentation.village_seg.val_metrics`, `uavloc.segmentation.eval.load_val_sets` | exist | — |
| `sim-flights` | exists | `dataset_ref` = `sim-flights@35a367`, as `033` / `027.12` |

## Setup

| | |
|---|---|
| Dataset | `sim-flights`, split `test` (94 flights, 47 villages), INS config from the set manifest |
| Seed | 42 (process), per-route INS and particle seeds from `routes.json`, as `033` |
| Model | the `033` arm's saved satellite-BN segmenter (both correlator sides) + its `head-<arm>.pt` |
| Filter | particles, 2,000, q = 2.0, T from the arm's `033` tuning (audit verifies 0.5) |
| Hardware | CPU only (bug-121), 3 lanes x 2 threads (the `033` test runs used 2 threads) |

Run order: audit → `035.1`, `035.2`, `035.3` in parallel, logged in that order → analysis.

## What would falsify "no swap"

- An arm's `035` flights match another arm's `033` flights.
- A checkpoint in `segmenter-control/` shows the building or structure fingerprint, or the
  reverse.
- The `035` building mean is below control while `033` had it above.

## Related

- `033-class-loss-in-system`: the runs under audit.
- `034-structure-blind-rule`: `034.2` / `034.3` re-ran `033.1` / `033.4` with no rule.
- `027-sequence-filters`: `027.12`, the thesis reference system.
