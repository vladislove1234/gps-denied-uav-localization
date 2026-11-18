# NNN-slug — proposal

> Written **before** the first run. The success criterion below does not move
> once runs begin. If nine runs all miss it, the honest report says the
> approach failed.

## Question

_One sentence. What are we trying to find out? Must be answerable yes/no or
by a number._

## Hypothesis

_What we expect to happen, and why. State the mechanism, not just the outcome._

## Success criterion

_A specific number on a specific metric against a specific baseline._

The metric that matters here is **metres of position error**, because the
system corrects an inertial prior rather than retrieving from a global map.
Retrieval scores are useful diagnostics, never the criterion on their own.
State the injected-error magnitude too — 5.4 m at 10 m injected and 10.8 m at
15 m are different claims about the same mechanism.

> Example: beat `002`'s best mean position error of 5.4 m on the held-out
> route at 10 m injected error, same split and seed 42.

**Metric:** `mean_position_error_m` (@ 10 m injected)
**Baseline:** `002.4` → 5.4 m
**Bar:** < 4.5 m

## Requires

_What has to exist before this experiment can run. Filled in **before**
building any of it — the question scopes the build, not the other way round.
Leave the table empty if everything already exists._

Equipment is anything reusable with no metric of its own: a dataset, a metric
implementation, a model, an eval harness. Code equipment lives in `uavloc/`
and is versioned by git. Data equipment lives in `datasets/<name>/` and is
versioned by its manifest hash. One-off glue stays in this folder.

| Equipment | Status | Acceptance check |
|---|---|---|
| `builds/<slug>` | to build | _how we will know it is sound, not just finished_ |
| `datasets/<name>` | to build | |
| `uavloc.<...>` | exists | — |

**What only you can provide.** Inputs I cannot make myself — a flight at a new
site, satellite tiles, a decision about which data to use. Each row names the
**expected path**, so a session can check whether it arrived instead of asking.

- [ ] _what_ → `data/flights/logs/<file>.bin`

> Building must not move the success criterion above. If the build reveals the
> metric is unmeasurable as promised, amend this file with a dated note saying
> what changed and why — do not quietly swap it.

## Setup

| | |
|---|---|
| Dataset | |
| Split | |
| Seed | |
| Model | |
| Loss | |
| Hardware | |

## What would falsify this

_What result would make us abandon this direction rather than keep tuning?
Write it down now, while it's still cheap to be honest._

## Related

_Prior experiments this builds on or contradicts. Link them._
