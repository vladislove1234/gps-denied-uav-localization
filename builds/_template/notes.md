# <slug>

**Status:** planning · **Unblocks:** `<experiment or build this enables>`

> Status is one of: `planning`, `blocked`, `ready for your review`,
> `accepted`, `parked`. It is parsed from this line — keep the format.

## What this is

_One or two sentences. What this piece of work does, and why it is worth
building once rather than inlining into a notebook._

## What I needed from you

_Inputs only you can supply: a flight, satellite tiles, a decision about
which site. Each row names the **expected path**, so a session can check
whether it has arrived without asking you._

- [ ] _what_ → `data/flights/logs/<file>.bin`

## What I did

_The implementation, in prose. Where the reusable code landed — that is
`uavloc/…`, not this folder. This folder holds the record and the evidence._

## Does it work

_Run it on the real data and show something judgeable. Not assertions against
synthetic input — for anything touching flight data, a plotted track that lands
in the wrong field catches more than a passing unit test ever would._

- `figures/<name>.png` — _what it shows, and why it looks right_

_State what is still wrong or unverified. A silent gap reads as a pass six
months from now._

## Your call

_What you should look at, and any decision that is genuinely yours. This is the
handoff — after this we either keep working on it or leave it as is._
