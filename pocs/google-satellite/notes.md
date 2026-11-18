# google-satellite

**Status:** worked · **Question:** Does Google satellite imagery from the Map Tiles API add a sharper, co-registrable, visually distinct look to `ua-village-multidate`?

## The question

Can Google imagery be fetched on the dataset's exact z18 tile grid, shifted onto
the Esri reference by the same phase correlation the build uses, and still look
different from every Esri date already stored? If it can't be registered, or if
it's the same capture Esri already has, it adds nothing.

## Verdict

**Worked.** On 8 villages Google imagery covers 99.98–100% of each grid. It sits
1.6–4.5 m off the Esri reference (median 2.5 m), a shift phase correlation
recovers with a strong peak in 7/8 villages. Its best correlation with *any*
stored Esri date is 0.12–0.43, far below the build's 0.97 duplicate threshold, so
it's a new look in every village. Native resolution is **z19 (0.19 m/px)**, twice
the dataset's z18. Even at z18 it's visibly sharper than Esri in most villages.

## What I ran

`probe.ipynb`, run headless with papermill into `runs/` (gitignored). It reads
the villages' `meta.json` grids from `data/derived/ua-village-multidate/`, pulls
Google tiles for the same grid, and compares them with the stored Esri mosaics.

- API key: `GOOGLE_MAPS_API_KEY` in `src/.env` (gitignored). One session per run
  (`POST /v1/createSession`, `mapType: satellite`), tiles from `/v1/2dtiles/z/x/y`.
- Tile cache: `data/derived/tile-cache/google-satellite/{z}/{x}/{y}.jpg`. A
  re-run costs zero requests. `MAX_NEW_TILES` stops a run that would fetch more.
- Co-registration uses `uavloc.matching.registration.phase_shift`, promoted
  from the dataset build for this, so both measure shift identically.

## Evidence

- `figures/esri-vs-google.png`: Esri reference vs Google z18 (shifted) and z19,
  over the central ~200 m of 4 villages. Same buildings and roads, different
  season and sun (Birky: low-sun late-season image against Esri's green summer).
- `figures/native-pixels.png`: ~50 m at each source's native pixels, no
  smoothing. z19 resolves roof edges and greenhouse stripes z18 doesn't. z20 is
  the z19 picture upsampled, which matches the viewport API's `maxZoom: 19`.
- `figures/detail-by-zoom.png`: share of variance lost by 2x down/up. It agrees
  with the pictures except Lapaivka, where Esri's blocky JPEG scores as detail.
  **Trust the pictures, not this metric.**
- `results.json`: per-village shift, peak ratio, correlation with every Esri date.

Not verified:
- **Capture date.** The API gives none, and the viewport `copyright` string came
  back empty. A Google sample can be labelled only "fetched 2026-09-27", not
  dated. Anything that orders looks by date can't use it.
- **Label alignment.** Measured against the Esri reference, not against OSM
  directly. The build aligns every date to that reference the same way, so the
  label offset should carry over, but it wasn't measured on Google itself.
- Zlynets' shift peak (14.1) is under the build's 15.0 cut-off, so the build
  would drop Google there. Expect roughly 1 village in 8 to lose it.
- 8 villages of 60. Coverage and freshness elsewhere in Ukraine are assumed.

## Cost

About 1,370 tile requests (1,050 at z18, 64 at z19, 256 at z20) and ~1.5 min
wall-clock on the first run. Full-dataset z18 for 60 villages is ~8,000 tiles;
z19 is 4x that.

## If it graduates

_Empty while the user decides._ The obvious route: a Google source in the
dataset build, added as one more candidate look beside Esri current and Wayback.
The open choice is z18 (fits the current dataset) vs z19 (Google's real
resolution, but the Esri dates only exist at z18).
