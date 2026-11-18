# Annotator

Local app for correcting segmentation labels over `ua-village-multidate`
(satellite, seeded from references) and for drawing labels on the 44 drone
frames of `drone-2025-easy` and the 2025 Gonchary ortho `gonchary-2025-ortho`
(both pixel-only, no references).
Spec: [SPEC.md](SPEC.md) · plan: [PLAN.md](PLAN.md).

## Run

From `src/`, in the `uavloc` env:

```bash
# 1. reference sources (once; resumable; --force to redo)
python -m tools.annotator.refs run --source all --locations all
python -m tools.annotator.refs list

# 2. the app
python -m tools.annotator serve            # http://127.0.0.1:8765

# 3. export done samples: satellite -> ua-village-annotated, drone -> ua-drone-annotated,
#    ortho -> ua-ortho-annotated
python -m tools.annotator export --new     # or --stale / --all
python -m tools.annotator status
```

## Using it

- **Left**: locations with status, grouped *Satellite* / *Drone frames*; the
  second filter picks a collection. **Centre**: map. **Right**: dates and
  compare, edit target, tool options, layers, reference overlays.
- A class is a group of **sublayers**, one per seeded source plus `manual`.
  Click a sublayer to make it active; every tool edits the active sublayer.
  `S` solos a sublayer, `⋯` re-seeds, merges or deletes it, `+ source` seeds
  another reference into the class.
- **Edit target** (`G`): *BASE* changes every date; *only <date>* records a
  per-date change (a building built later, a forest cut). The map gets an
  orange frame when you are editing a date only.
- **Align a date** (`T` with target *only <date>*): drag to move that date's
  image under the layers (= moving every layer on that date only). Stored as
  `offset_px` in `state.json`; export shifts that date's label to match. The
  JPEG is never changed; Reset in the Tool panel undoes it.
- **Compare**: opacity, swipe (drag the divider), blink (hold `X`). `,` `.`
  step through dates.
- **Reference overlays** are read-only. Turn one on and use **Adopt** (`A`) to
  copy a shape or blob into a sublayer.
- **Magic wand** (`W`): hover to preview, click to fill, `Alt`+click to erase.
  *Tolerance* = colour range, *Edges* = how strongly image edges stop the fill,
  *Reach* = the window it can spread in.
- Autosaves 1.5 s after the last edit. Editing a `done` date sets it back to
  `in_progress`. `⌘Enter` marks the viewed date done. `?` lists all shortcuts.

## Checks

```bash
node tools/annotator/checks/web_selftest.mjs                 # geometry, bit packing, wand
python tools/annotator/checks/export_check.py                 # uavloc env; export bookkeeping
/Users/vladyslavtaraban/miniconda3/bin/python tools/annotator/checks/e2e.py   # headless browser, real data
papermill tools/annotator/checks/m1_refs.ipynb tools/annotator/checks/runs/m1_refs.ipynb
```

`e2e.py` and `export_check.py` run on a scratch workspace and never touch
`data/derived/annotation/work` or the real export.

## Layout

| path | what |
|---|---|
| `classes.json` | class registry: ids, colours, geometry, tools, export priority, seed sources. Adding a class = one entry + bump `version` |
| `refs/` | adapter contract (`base.py`) + one module per source |
| `server/` | FastAPI app, workspace storage, export |
| `web/` | vanilla JS frontend, no build step, no external libraries |
| `checks/` | the checks above and their figures |

Data: `data/derived/annotation/{refs,work}/` (gitignored with `data/`).
