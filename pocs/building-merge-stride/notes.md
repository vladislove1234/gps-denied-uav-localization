# building-merge-stride

**Status:** worked · **Question:** `Are LR-ASPP's merged buildings a resolution limit of its stride-8 output, or a model error a loss could fix?`

## The question

When LR-ASPP merges neighbouring buildings into one blob, would a perfect stride-8 prediction (label averaged into
8 x 8 cells, upsampled like the model) merge them too?

## Verdict

**Both, roughly half each.** Of the 595 labelled buildings LR-ASPP merges on the val sets, 263 (44 %) are merged by
the perfect stride-8 map as well, so no loss can separate them at stride 8. 329 (55 %) the perfect stride-8 map
keeps separate, so they are model errors. The split is by gap width: below ~2 m the stride-8 map itself merges
66-95 % of buildings; at 3-4 m it merges none while LR-ASPP merges 45 %, and 17 % at 4-6 m. A gap-weighted loss can
target the 3-8 m gaps; the <2 m gaps need a finer output (stride 4). Missed buildings (24.8 % of all, vs <1 % for
the stride-8 map) are as large a problem as merges (29.7 %).

**Stride 4 (added 2026-09-29):** a perfect stride-4 map (bilinear) merges 14.6 % of buildings vs 25.7 % at
stride 8. Of LR-ASPP's 595 merges, only 149 (25 %) stay resolution-bound at stride 4, vs 263 (44 %) at stride 8:
a stride-4 output would make 114 more merges fixable. It separates every gap >= 2 m (1-2 m: 31 % merged vs 66 %),
but gaps < 1 m stay 90 % merged; only near full resolution splits those. The stride-8 numbers reproduced exactly
after `label_at_stride` gained its `stride` argument.

## What I ran

`merge.ipynb` (executed copy with every table: `runs/merge.ipynb`). Val sets of the seg-boundary-eval harness
(birky-strip, birky-strip-google, bushcha; 7 images, 2,001 labelled buildings). Buildings = 4-connected components
of the building class, >= 16 px (2.4 m²). A labelled building is *missed* if no predicted component covers >= 20 %
of it, *merged* if a component covering it also covers another building, else *separate*. Gap = distance to the
nearest other labelled building, capped at 40 px (15.4 m). 0.385 m/px, one LR-ASPP cell = 3.08 m.

## Evidence

- `figures/merged-by-gap.png` — share merged vs gap; the stride-8 maps drop to 0 at 3 m, LR-ASPP stays at 45 % to
  4 m. Below 2 m LR-ASPP merges *less* than the stride-8 map only because it misses many of those buildings.
- `figures/merged-by-gap.png` also shows the stride-4 oracle: zero merges from 2 m up.
- `figures/status-shares.png` — separate / merged / missed per map and val set.
- `figures/example-bushcha.png` — the crop from the user's screenshot, buildings coloured by status.

Not checked: sensitivity to MATCH_FRAC (0.2), MIN_BUILDING_PX (16) and 4- vs 8-connectivity. 45 % of buildings sit
< 3.08 m from a neighbour, so the resolution share depends on the village density.

## Cost

~5 min, mostly LR-ASPP on the val images (CPU).
