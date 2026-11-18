# <slug>

**Status:** exploring · **Question:** `<the one thing this is meant to settle>`

> Status is one of: `exploring`, `worked`, `failed`, `parked`. Nothing parses
> this yet — PoCs are deliberately absent from `INDEX.md` — but it matches the
> build format so a parser can pick it up if that ever changes. A PoC has no
> success criterion and no bar; it is allowed to come back `failed`, and a
> `failed` folder stays exactly where it is.

## The question

_One sentence, answerable yes or no. "Does Delineate Anything produce usable
field boundaries on our Google Earth imagery" is a question. "Explore
segmentation" is not — if it cannot come back no, it is not a PoC._

## Verdict

_Fill this in last. Two or three sentences, written so that someone who never
saw the code knows whether to build on it. If it failed, say what specifically
failed — the model, the data, the assumption — because those have different
next steps._

## What I ran

_Prose, brief. Which files in this folder do what, what data they touched, and
anything a re-run would need to know (weights, cache paths, an endpoint that
rate-limits). The code lives HERE, not in `uavloc/` — that move happens only on
graduation._

## Evidence

- `figures/<name>.png` — _what it shows, and what in it supports the verdict_

_State what is unverified or what the figure does not prove. A PoC earns its
keep by being honest about the edge of what it checked._

## Cost

_Roughly what it took — wall-clock, whether a download or a fit dominated,
anything that would make a repeat expensive. This is what tells you next time
whether to bother._

## If it graduates

_What would move into `uavloc/`, and under which module. Leave this empty while
exploring. On graduation, replace this section with a single line naming where
the code went, and keep the rest of the file as the record._
