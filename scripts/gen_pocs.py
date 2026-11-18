"""Generate the three PoC notebooks.

Notebooks are JSON; patching them by cell index goes wrong (bug-020). This is
the generator — edit here and re-run, never hand-edit the .ipynb.

    python gen_pocs.py
"""
from __future__ import annotations

import json
from pathlib import Path

POCS = Path("/Users/vladyslavtaraban/Documents/Projects/UAV_Localization/src/pocs")


def nb(cells):
    """Build the notebook JSON.

    The first code cell gets `"tags": ["parameters"]`. Without that tag papermill
    injects its -p overrides at the TOP of the notebook, and the defaults cell
    further down then reassigns them back — so `-p train True` is silently
    ignored and the run does the wrong thing while exiting 0.
    """
    out = []
    tagged = False
    for kind, src in cells:
        lines = src.strip("\n").split("\n")
        lines = [l + "\n" for l in lines[:-1]] + [lines[-1]]
        meta = {}
        if kind == "code" and not tagged:
            meta = {"tags": ["parameters"]}
            tagged = True
        cell = {"cell_type": kind, "metadata": meta, "source": lines}
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        out.append(cell)
    return {
        "cells": out,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.16"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write(slug: str, cells):
    d = POCS / slug
    (d / "figures").mkdir(parents=True, exist_ok=True)
    (d / "probe.ipynb").write_text(json.dumps(nb(cells), indent=1) + "\n")
    print(f"wrote {d/'probe.ipynb'}  ({len(cells)} cells)")


# --------------------------------------------------------------------------
# shared preamble — every probe loads locations the same way
# --------------------------------------------------------------------------

SETUP = '''
import gzip, io, json, sys, time, urllib.request, warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = None

assert Path("datasets").is_dir(), "run from src/ — cwd is " + str(Path.cwd())

sys.path.insert(0, str(Path.cwd() / "pocs" / "map-derived-labels"))
import align, osm                              # noqa: E402
from tiles import TileGrid                     # noqa: E402

ROOT = Path("data/derived/ua-village-multidate")
HERE = Path("pocs") / SLUG
FIGS = HERE / "figures"
FIGS.mkdir(parents=True, exist_ok=True)


def locations():
    return json.loads((ROOT / "index.json").read_text())["built"]


def load_location(name, date="current"):
    """(image, grid, meta) at full resolution — same path build.py rasterised through."""
    meta = json.loads((ROOT / name / "meta.json").read_text())
    img = Image.open(ROOT / name / f"{date}.jpg").convert("RGB")
    return img, TileGrid(**meta["grid"]), meta


def osm_mask(name, cls):
    """One class from the dataset's own label.png, shifted by its measured offset.

    The build recorded `label_offset` per location. Charging the dataset's known
    2.84 m label error to the thing under test would be unfair, so undo it here.
    """
    meta = json.loads((ROOT / name / "meta.json").read_text())
    lab = np.array(Image.open(ROOT / name / "label.png"))
    m = lab == osm.CLASSES[cls]
    off = meta.get("label_offset") or {}
    dx, dy = int(off.get("dx_px", 0)), int(off.get("dy_px", 0))
    if dx or dy:
        m = np.roll(np.roll(m, dy, axis=0), dx, axis=1)
    return m


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else float("nan")


def null_iou(pred, truth, k=20, seed=0):
    """Same mask, position destroyed. Anything not beating this is an artifact
    of how much ground got painted (bug-016)."""
    rng = np.random.default_rng(seed)
    h, w = pred.shape
    return float(np.median([
        iou(np.roll(np.roll(pred, rng.integers(h // 8, h - h // 8), 0),
                    rng.integers(w // 8, w - w // 8), 1), truth)
        for _ in range(k)
    ]))
'''


# ==========================================================================
# PoC A — Microsoft Global ML Building Footprints
# ==========================================================================

A = [
("markdown", """
# ms-building-footprints — does an independent building source agree with OSM?

**Status:** `exploring`

`building` is one of only two classes measured good, and the whole quality story
rests on a single source. Microsoft's Global ML Building Footprints is derived
from Bing aerial imagery by a model — **independent of OSM in both data and
method**. So it can say three things OSM alone cannot:

1. **Does OSM miss buildings?** MS-only footprints over ground OSM left blank.
2. **Is the 2.84 m label offset real?** Two independent maps disagreeing with the
   imagery by the *same* vector is confirmation; disagreeing differently is not.
3. **Can the two be merged** into a denser, better-conditioned `building` class?

### What would make this `failed`

- MS footprints agree with OSM no better than the null baseline → not an
  independent source, just a copy or noise.
- MS-vs-imagery offset is uncorrelated with OSM-vs-imagery offset → the offset is
  not a property of the maps and the per-location correction is unjustified.
- Coverage over rural Ukraine is so sparse it adds nothing OSM did not have.

Run from `src/` with the `uavloc` env.
`papermill pocs/ms-building-footprints/probe.ipynb pocs/ms-building-footprints/runs/probe-$(date +%F).ipynb -p n 59`
"""),
("code", """
# papermill parameters
SLUG = "ms-building-footprints"
n = 9              # locations; 59 = all
links_url = "https://bfppub.blob.core.windows.net/$web/2026-08-13/dataset-links.csv"
country = "Ukraine"
"""),
("markdown", "## Setup"),
("code", SETUP),
("markdown", """
## The source

MS publishes one index CSV of `(Location, QuadKey, Url, Size)`. QuadKeys are
z9 tiles; each file is `.csv.gz` whose contents are **geojsonl**, not CSV —
the extension lies, and that is the single thing most likely to break this cell.
Licence is CDLA Permissive 2.0.

Verify the column names and the payload shape before trusting anything below.
"""),
("code", '''
CACHE = Path("data/derived/ms-buildings")
CACHE.mkdir(parents=True, exist_ok=True)


def get(url, dest=None):
    if dest and dest.exists():
        return dest.read_bytes()
    req = urllib.request.Request(url, headers={"User-Agent": "uav-localization-research/0.1"})
    with urllib.request.urlopen(req, timeout=180) as r:
        blob = r.read()
    if dest:
        dest.write_bytes(blob)
    return blob


raw = get(links_url, CACHE / "dataset-links.csv").decode("utf-8", "replace")
header, *rows = raw.strip().split("\\n")
cols = [c.strip() for c in header.split(",")]
print("columns:", cols)

recs = [dict(zip(cols, r.split(","))) for r in rows]
ua = [r for r in recs if r.get("Location", "").strip() == country]
print(f"{len(recs)} rows total, {len(ua)} for {country}")
assert ua, f"no rows for {country} — check the Location spelling in `cols` above"
print("sample:", ua[0])
'''),
("code", '''
def quadkey(lat, lon, z=9):
    """Bing quadkey for the z-level tile containing (lat, lon)."""
    import math
    lat = max(min(lat, 85.05112878), -85.05112878)
    n = 1 << z
    x = int((lon + 180.0) / 360.0 * n)
    s = math.sin(math.radians(lat))
    y = int((0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)) * n)
    x, y = min(x, n - 1), min(y, n - 1)
    key = ""
    for i in range(z, 0, -1):
        d, mask = 0, 1 << (i - 1)
        if x & mask:
            d += 1
        if y & mask:
            d += 2
        key += str(d)
    return key


names = locations()[:n]

# A z9 quadkey is ~50 km across at this latitude and a location box is 1.28 km,
# so most locations sit inside one tile -- but not all. Measured: 3 of the 59
# span two quadkeys. Taking only the centre's key silently drops the buildings
# on the far side, which would look exactly like "OSM has buildings MS does not"
# -- the very thing this probe is testing. So union all four corners.
keys_for = {}
for name in names:
    meta = json.loads((ROOT / name / "meta.json").read_text())
    b = meta["bounds"]
    keys_for[name] = sorted({quadkey(la, lo)
                             for la in (b["top"], b["bottom"])
                             for lo in (b["left"], b["right"])})

need = sorted({k for ks in keys_for.values() for k in ks})
spanning = {nm: ks for nm, ks in keys_for.items() if len(ks) > 1}
print(f"{len(names)} locations -> {len(need)} z9 quadkeys")
print(f"locations spanning >1 quadkey: {len(spanning)}"
      + (f" -> {list(spanning)}" if spanning else ""))

by_key = {r["QuadKey"]: r for r in ua}
missing = [k for k in need if k not in by_key]
print("quadkeys with no MS file:", missing or "none")
assert not missing, "some ground has no MS coverage; handle before scoring"
'''),
("code", '''
def load_quadkey(key):
    """Return the list of GeoJSON features in one MS quadkey file."""
    row = by_key.get(key)
    if row is None:
        return []
    dest = CACHE / f"{key}.csv.gz"
    blob = get(row["Url"], dest)
    feats = []
    with gzip.open(io.BytesIO(blob), "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                feats.append(json.loads(line))
    return feats


t0 = time.time()
tiles_feats = {k: load_quadkey(k) for k in need}
print(f"fetched {len(tiles_feats)} quadkeys in {time.time()-t0:.0f}s")
for k, f in tiles_feats.items():
    users = [nm for nm, ks in keys_for.items() if k in ks]
    print(f"  {k}: {len(f):>8,} footprints  ({len(users)} loc)")


def features_for(name):
    """Every footprint from every quadkey this location touches."""
    out = []
    for k in keys_for[name]:
        out.extend(tiles_feats[k])
    return out
'''),
("markdown", """
## The control

OSM `building`, from the dataset's own `label.png`, shifted back by the
per-location `label_offset` the build measured. That is the trusted yardstick —
measured good, 4.09% coverage — and it is what makes this probe falsifiable.

Paired with a **null baseline** because the score is an aggregate (`bug-016`).
"""),
("code", '''
def burn(features, grid, shape):
    """Rasterise MS footprints through the tile grid's closed-form georeference.

    TileGrid.to_pixel, not Bounds.to_pixel — Bounds is linear in latitude and
    drifts across the mosaic. Using the grid is what makes these land where the
    dataset's own labels land.
    """
    canvas = Image.new("L", (shape[1], shape[0]), 0)
    draw = ImageDraw.Draw(canvas)
    drawn = 0
    for feat in features:
        geom = feat.get("geometry", {})
        if geom.get("type") != "Polygon":
            continue
        ring = geom["coordinates"][0]
        pts = [grid.to_pixel(lat, lon, shape) for lon, lat in ring]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        if max(xs) < 0 or min(xs) > shape[1] or max(ys) < 0 or min(ys) > shape[0]:
            continue
        draw.polygon(pts, fill=1)
        drawn += 1
    return np.array(canvas, bool), drawn
'''),
("markdown", "## Run"),
("code", '''
results = []
for name in names:
    img, grid, meta = load_location(name)
    shape = np.array(img).shape[:2]
    ms, drawn = burn(features_for(name), grid, shape)
    ref = osm_mask(name, "building")

    # Offset against the imagery, measured exactly the way the build measured
    # OSM's: image GRADIENT vs label OUTLINE, not raw image vs filled mask.
    # peak_z < 4 means the two layers share no structure and the offset is
    # meaningless however precise it looks (align.offset docstring).
    grey = np.array(img.convert("L"), float)
    try:
        ms_off = align.offset(align.gradient_magnitude(grey),
                              align.outline(ms), meta["mpp"])
        ms_off["trusted"] = bool(ms_off["peak_z"] >= 4 and not ms_off["at_search_limit"])
    except Exception as exc:                     # noqa: BLE001
        ms_off = {"error": repr(exc)}

    results.append({
        "location": name,
        "ms_drawn": drawn,
        "ms_frac": float(ms.mean()),
        "osm_frac": float(ref.mean()),
        "iou": iou(ms, ref),
        "null_iou": null_iou(ms, ref),
        "ms_only_frac": float((ms & ~ref).mean()),
        "osm_only_frac": float((ref & ~ms).mean()),
        "osm_recall_by_ms": float((ms & ref).sum() / ref.sum()) if ref.sum() else float("nan"),
        "ms_offset": ms_off,
        "osm_offset": meta.get("label_offset"),
        "mpp": meta["mpp"],
    })
    r = results[-1]
    print(f"{name:34s} ms {drawn:>6,}  IoU {r['iou']:.3f}  null {r['null_iou']:.3f}  "
          f"recall {r['osm_recall_by_ms']:.3f}")

(HERE / "results.json").write_text(json.dumps(results, indent=1))
'''),
("markdown", """
## Numbers

Distribution, never a single extremum (`bug-019`). Three questions, in order:

1. **Is MS independent and real?** IoU vs its own null.
2. **Does it add anything?** `ms_only_frac` against `osm_frac`.
3. **Is the offset a map property?** correlation of the two offset vectors.
"""),
("code", '''
import statistics as st


def col(k):
    return [r[k] for r in results if isinstance(r.get(k), float) and not np.isnan(r[k])]


print("             n   median     min     max")
for k in ("iou", "null_iou", "osm_recall_by_ms", "ms_only_frac", "osm_only_frac"):
    v = col(k)
    print(f"{k:>18s} {len(v):>3}  {st.median(v):.4f}  {min(v):.4f}  {max(v):.4f}")

beat = [r for r in results if r["iou"] > r["null_iou"] * 2]
print(f"\\nlocations beating 2x their null: {len(beat)}/{len(results)}")

pairs = [(r["ms_offset"], r["osm_offset"]) for r in results
         if isinstance(r["ms_offset"], dict) and r["ms_offset"].get("trusted")
         and (r["osm_offset"] or {}).get("peak_z", 0) >= 4]
print(f"\\nlocations where BOTH offsets have peak_z >= 4: {len(pairs)}/{len(results)}")
if pairs:
    mx = np.array([[p[0]["dx_px"], p[0]["dy_px"]] for p in pairs], float)
    ox = np.array([[p[1]["dx_px"], p[1]["dy_px"]] for p in pairs], float)
    print(f"\\noffset agreement over {len(pairs)} locations")
    print(f"  median |MS - OSM| = {np.median(np.hypot(*(mx - ox).T)):.2f} px")
    for ax, lab in ((0, "dx"), (1, "dy")):
        if mx[:, ax].std() > 1e-9 and ox[:, ax].std() > 1e-9:
            print(f"  corr({lab}) = {np.corrcoef(mx[:, ax], ox[:, ax])[0,1]:+.3f}")
'''),
("markdown", """
## Look at it

Full resolution before any verdict (`bug-014`). Draw **everything** MS returned,
not a selection. The zoom panel is the one that decides whether these are real
footprints on real roofs.
"""),
("code", '''
show = names[:min(4, len(names))]
fig, axes = plt.subplots(len(show), 3, figsize=(15, 5 * len(show)))
axes = np.atleast_2d(axes)
for row, name in enumerate(show):
    img, grid, meta = load_location(name)
    arr = np.array(img)
    ms, _ = burn(features_for(name), grid, arr.shape[:2])
    ref = osm_mask(name, "building")

    axes[row, 0].imshow(arr)
    axes[row, 0].set_title(f"{name}\\nimagery", fontsize=8)

    rgb = np.zeros((*ms.shape, 3))
    rgb[..., 0] = ms            # MS red
    rgb[..., 1] = ref           # OSM green -> agreement is yellow
    axes[row, 1].imshow(arr)
    axes[row, 1].imshow(rgb, alpha=0.55)
    axes[row, 1].set_title("red = MS · green = OSM · yellow = both", fontsize=8)

    ys, xs = np.nonzero(ms | ref)
    if len(ys):
        cy, cx = int(np.median(ys)), int(np.median(xs))
        h = 300
        sl = (slice(max(0, cy - h), cy + h), slice(max(0, cx - h), cx + h))
        axes[row, 2].imshow(arr[sl])
        axes[row, 2].imshow(rgb[sl], alpha=0.55)
    axes[row, 2].set_title("native-resolution zoom", fontsize=8)
    for a in axes[row]:
        a.axis("off")
plt.tight_layout()
plt.savefig(FIGS / "ms-vs-osm.png", dpi=130, bbox_inches="tight")
plt.show()
'''),
("markdown", """
## Verdict

`worked` / `failed` / `parked`, two or three lines, then copy into `notes.md`.

### What is *not* established
"""),
]


# ==========================================================================
# PoC B — FLAIR pretrained transfer
# ==========================================================================

B = [
("markdown", """
# flair-landcover-transfer — does a French land-cover model work on Ukrainian tiles?

**Status:** `exploring`

`IGNF/FLAIR-INC_rgb_15cl_resnet34-fpn` — FPN + ResNet-34, **RGB only** (Esri
tiles have no NIR and no elevation), trained by IGN on 174,700 photo-interpreted
512x512 patches of French BD ORTHO at **0.2 m**. Licence Etalab 2.0.
Reported per-class IoU: Water 83.1, Building 76.3, Brushwood 29.9, mIoU 59.3.

**This probe gates every aerial-pretrained-model route.** If a model this large
and this well-supervised cannot find buildings over Ukrainian Esri tiles, neither
will LandCover.ai, and a day of training is saved.

It is **self-falsifying**, which probe 1 was not and probe 4 was: `Building` and
`Water` are exactly the two classes where OSM labels are measured good. A model
that misses buildings OSM proves are there cannot be trusted on anything else.

### The two risks, named before running

1. **GSD: model expects 0.2 m, tiles are 0.386 m/px (1.93x).** Both ways of
   feeding it are flawed — upsampling fabricates detail (the exact failure in
   `pocs/drone-frame-boundaries`, where a 131 px crop shown at 512 px returned
   nothing), feeding native halves apparent object size. So run **both**, one
   variable, same tiles.
2. **Sensor: BD ORTHO is aircraft, Esri is a satellite mosaic.** The card says
   outright that generalisation issues can occur at other resolutions and sensors.

### What would make this `failed`

- `Building` or `Water` IoU at or below its null baseline at both GSDs.
- Masks that follow the inference grid rather than the imagery (`bug-016` —
  look at the figure before writing anything).
- Predictions disagreeing across two dates of the *same ground* more than they
  agree — a model keying on season, not structure.

Run from `src/` with the `uavloc` env.
`papermill pocs/flair-landcover-transfer/probe.ipynb pocs/flair-landcover-transfer/runs/probe-$(date +%F).ipynb -p n 9`
"""),
("code", """
# papermill parameters
SLUG = "flair-landcover-transfer"
n = 9                       # locations
gsd_modes = ["native", "upsampled"]
repo = "IGNF/FLAIR-INC_rgb_15cl_resnet34-fpn"
patch = 512
overlap = 64
flair_mpp = 0.2             # what the model was trained at
cross_date = True           # the no-ground-truth consistency check
"""),
("markdown", """
## Setup

`segmentation_models_pytorch` and `timm` are **not** in the `uavloc` env. Install
under the existing pin — torch 2.2.2 / torchvision 0.17.2 — or this quietly drags
torch forward and breaks `ultralytics`.

```
/Users/vladyslavtaraban/miniconda3/envs/uavloc/bin/pip install \\
    "segmentation-models-pytorch==0.3.4" "timm==0.9.16" \\
    -c <(printf 'torch==2.2.2\\ntorchvision==0.17.2\\n')
```
"""),
("code", SETUP),
("code", '''
import torch
import segmentation_models_pytorch as smp
from huggingface_hub import hf_hub_download

print("torch", torch.__version__, "| smp", smp.__version__)
torch.set_num_threads(6)        # 6 physical cores on this machine

# FLAIR's nomenclature is **19 classes**, not 15 — the "15cl" in the repo name
# counts the ACTIVE ones. The head emits 19 logits and four were trained with
# weight 0 (README: clear cut, mixed, ligneous, other). Verified against
# `criterion.weight` in the checkpoint, which is [1]*14 + [0,0,0,1,0].
FLAIR_CLASSES = [
    "building", "pervious surface", "impervious surface", "bare soil", "water",
    "coniferous", "deciduous", "brushwood", "vineyard", "herbaceous vegetation",
    "agricultural land", "plowed land", "swimming pool", "snow",
    "clear cut", "mixed", "ligneous",          # deactivated
    "greenhouse",
    "other",                                    # deactivated
]
DEAD = {"clear cut", "mixed", "ligneous", "other"}
DEAD_IDX = [FLAIR_CLASSES.index(c) for c in DEAD]
# how they fold onto our 5-class set
TO_OURS = {
    "building": ["building"],
    "water": ["water", "swimming pool"],
    "woody": ["coniferous", "deciduous", "brushwood"],
    "track": ["impervious surface"],          # FLAIR has no road class — see notes
}
MEAN = np.array([105.08, 110.87, 101.82], np.float32)
STD = np.array([52.17, 45.38, 44.00], np.float32)
'''),
("code", '''
from huggingface_hub import list_repo_files
files = list_repo_files(repo)
print("repo files:", files)

weights = next((f for f in files if f.endswith((".ckpt", ".pth", ".pt", ".safetensors"))), None)
assert weights, f"no weight file found in {repo}: {files}"
path = hf_hub_download(repo, filename=weights)
print("weights ->", path)

# Keys are prefixed `model.seg_model.` and the checkpoint also carries
# `criterion.weight`, which is not part of the network. Stripping only `model.`
# leaves `seg_model.*`, which matches nothing and silently yields a
# randomly-initialised model — the bug-018 shape of failure.
PREFIX = "model.seg_model."
raw = torch.load(path, map_location="cpu")
state = {k[len(PREFIX):]: v for k, v in raw.items() if k.startswith(PREFIX)}
assert state, f"prefix {PREFIX!r} matched nothing; keys look like {list(raw)[:3]}"

model = smp.FPN(encoder_name="resnet34", encoder_weights=None,
                in_channels=3, classes=len(FLAIR_CLASSES))
missing, unexpected = model.load_state_dict(state, strict=True)
print(f"loaded {len(state)} tensors into FPN/resnet34 with {len(FLAIR_CLASSES)} outputs")
if "criterion.weight" in raw:
    print("train class weights:", raw["criterion.weight"].numpy().astype(int).tolist())
model.eval()
'''),
("markdown", """
### Load check — the README hands us a free one

*"classes n° 15, 16, 17 and 19 should appear at 0 in the logits and should not be
present in the final argmax product."* So if a deactivated class ever wins the
argmax, the weights did **not** load correctly. That turns a silent failure into
a loud one, which is the whole lesson of `bug-018`.
"""),
("code", '''
with torch.no_grad():
    probe = model(torch.randn(2, 3, 512, 512))
print("logits:", tuple(probe.shape), "- expected (2, 19, 512, 512)")
assert probe.shape[1] == 19, "head is not 19-way; class mapping below would be wrong"

won = set(np.unique(probe.argmax(1).numpy()).tolist())
alive = won - set(DEAD_IDX)
print("classes winning argmax on noise:", sorted(won))
assert not (won & set(DEAD_IDX)), (
    f"deactivated classes {sorted(won & set(DEAD_IDX))} won the argmax — "
    "weights did not load, do not trust anything below")
print("load check passed —", len(alive), "active classes present")
'''),
("markdown", """
## Run — tiled inference at both GSDs

The README is explicit: *"No data augmentation method concerning scale change was
used during training."* The model is scale-brittle by construction, so the
`upsampled` arm (matching its 0.2 m training scale) is the one with the stronger
prior — even though it fabricates detail. That is exactly why both arms run.
"""),
("code", '''
@torch.no_grad()
def predict(img, mpp, mode):
    """Return a (H, W) class-index map at the ORIGINAL pixel grid.

    mode="native"    feed at the tile's own 0.386 m/px — objects look half size
    mode="upsampled" resample to FLAIR's 0.2 m/px — fabricates detail
    """
    arr = np.asarray(img, np.float32)
    h0, w0 = arr.shape[:2]
    if mode == "upsampled":
        scale = mpp / flair_mpp
        img_s = img.resize((int(w0 * scale), int(h0 * scale)), Image.BICUBIC)
        arr = np.asarray(img_s, np.float32)
    h, w = arr.shape[:2]

    x = (arr - MEAN) / STD
    logits = np.zeros((len(FLAIR_CLASSES), h, w), np.float32)
    count = np.zeros((h, w), np.float32)
    step = patch - overlap
    for top in range(0, max(h - overlap, 1), step):
        for left in range(0, max(w - overlap, 1), step):
            t, l = min(top, max(h - patch, 0)), min(left, max(w - patch, 0))
            tile = x[t:t + patch, l:l + patch]
            ph, pw = tile.shape[:2]
            if ph < patch or pw < patch:
                tile = np.pad(tile, ((0, patch - ph), (0, patch - pw), (0, 0)))
            out = model(torch.from_numpy(tile.transpose(2, 0, 1))[None])[0].numpy()
            logits[:, t:t + ph, l:l + pw] += out[:, :ph, :pw]
            count[t:t + ph, l:l + pw] += 1
    logits /= np.maximum(count, 1)
    pred = logits.argmax(0).astype(np.uint8)
    if mode == "upsampled":
        pred = np.array(Image.fromarray(pred).resize((w0, h0), Image.NEAREST))
    return pred


def fold(pred, ours):
    idx = [FLAIR_CLASSES.index(c) for c in TO_OURS[ours]]
    return np.isin(pred, idx)
'''),
("code", '''
names = locations()[:n]
results, cache = [], {}

for name in names:
    img, grid, meta = load_location(name)
    for mode in gsd_modes:
        t0 = time.time()
        pred = predict(img, meta["mpp"], mode)
        cache[(name, mode)] = pred
        row = {"location": name, "mode": mode, "seconds": round(time.time() - t0, 1)}
        for ours, osm_cls in (("building", "building"), ("water", "water"), ("woody", "wood")):
            p, t = fold(pred, ours), osm_mask(name, osm_cls)
            row[ours] = {
                "pred_frac": float(p.mean()),
                "osm_frac": float(t.mean()),
                "iou": iou(p, t),
                "null_iou": null_iou(p, t),
                "recall": float((p & t).sum() / t.sum()) if t.sum() else float("nan"),
            }
        row["track"] = {"pred_frac": float(fold(pred, "track").mean())}
        results.append(row)
        b = row["building"]
        print(f"{name:30s} {mode:>10s} {row['seconds']:>5.0f}s  "
              f"building IoU {b['iou']:.3f} (null {b['null_iou']:.3f})  "
              f"claims {b['pred_frac']:.1%} vs actual {b['osm_frac']:.1%}")

(HERE / "results.json").write_text(json.dumps(results, indent=1))
'''),
("markdown", """
## Cross-date consistency — scored with no ground truth at all

Every date of a location is co-registered to **1.5 cm median** and shares one
label. So running the model on two dates of the *same ground* gives a score that
needs no truth: **disagreement between dates is pure source error.**

This measures the property the dataset exists for — invariance to season, haze
and sensor date. A source that scores well against OSM but flips between dates is
keying on appearance, and is useless here however good its IoU looks.
"""),
("code", '''
consistency = []
if cross_date:
    for name in names[:min(5, len(names))]:
        dates = sorted(p.stem for p in (ROOT / name).glob("*.jpg") if p.stem != "current")
        if not dates:
            continue
        other = dates[0]
        img_o, _, meta = load_location(name, other)
        for mode in gsd_modes:
            a = cache[(name, mode)]
            b = predict(img_o, meta["mpp"], mode)
            row = {"location": name, "mode": mode, "date": other,
                   "pixel_agreement": float((a == b).mean())}
            for ours in ("building", "water", "woody"):
                row[ours] = iou(fold(a, ours), fold(b, ours))
            consistency.append(row)
            print(f"{name:30s} {mode:>10s} vs {other}  pixels {row['pixel_agreement']:.1%}  "
                  f"building {row['building']:.3f}  woody {row['woody']:.3f}")

(HERE / "results-consistency.json").write_text(json.dumps(consistency, indent=1))
'''),
("markdown", """
## Numbers

Distribution per class per GSD, each paired with its null (`bug-016`). The
`claims vs actual` column is the one that caught YOLOE: a mask covering half the
tile scores well on recall while carrying no positional information.
"""),
("code", '''
import statistics as st

print(f"{'mode':>10s} {'class':>9s} {'n':>3s} {'IoU':>7s} {'null':>7s} {'recall':>7s} "
      f"{'claims':>7s} {'actual':>7s}")
for mode in gsd_modes:
    for cls in ("building", "water", "woody"):
        rows = [r[cls] for r in results if r["mode"] == mode
                and not np.isnan(r[cls]["iou"])]
        if not rows:
            continue
        m = lambda k: st.median([r[k] for r in rows if not np.isnan(r[k])])   # noqa: E731
        print(f"{mode:>10s} {cls:>9s} {len(rows):>3} {m('iou'):>7.3f} {m('null_iou'):>7.3f} "
              f"{m('recall'):>7.3f} {m('pred_frac'):>7.1%} {m('osm_frac'):>7.1%}")

print("\\nbeats null (IoU > 2x null), per mode/class:")
for mode in gsd_modes:
    for cls in ("building", "water", "woody"):
        rows = [r[cls] for r in results if r["mode"] == mode]
        k = sum(1 for r in rows if r["iou"] > 2 * r["null_iou"])
        print(f"  {mode:>10s} {cls:>9s}  {k}/{len(rows)}")
'''),
("markdown", """
## Look at it

Full resolution, every class, both GSDs, **before** the verdict. The specific
thing to check: do the masks follow buildings and treelines, or the 512 px
inference grid? Grid-aligned rectangles is what YOLOE produced and it is visible
instantly.
"""),
("code", '''
palette = {"building": (1, 0, 0), "water": (0, 0.4, 1),
           "woody": (0, 0.8, 0), "track": (1, 0.85, 0)}
show = names[:min(3, len(names))]
cols = 1 + len(gsd_modes) + 1
fig, axes = plt.subplots(len(show), cols, figsize=(5 * cols, 5 * len(show)))
axes = np.atleast_2d(axes)
for row, name in enumerate(show):
    img, _, _ = load_location(name)
    arr = np.array(img)
    axes[row, 0].imshow(arr)
    axes[row, 0].set_title(f"{name}\\nimagery", fontsize=8)
    for j, mode in enumerate(gsd_modes, start=1):
        over = np.zeros((*arr.shape[:2], 3))
        for cls, colour in palette.items():
            m = fold(cache[(name, mode)], cls)
            for c in range(3):
                over[..., c] = np.where(m, colour[c], over[..., c])
        axes[row, j].imshow(arr)
        axes[row, j].imshow(over, alpha=0.5)
        axes[row, j].set_title(f"FLAIR — {mode}", fontsize=8)
    ctrl = np.zeros((*arr.shape[:2], 3))
    ctrl[..., 0] = osm_mask(name, "building")
    ctrl[..., 2] = osm_mask(name, "water")
    axes[row, -1].imshow(arr)
    axes[row, -1].imshow(ctrl, alpha=0.6)
    axes[row, -1].set_title("OSM control (building/water)", fontsize=8)
    for a in axes[row]:
        a.axis("off")
plt.tight_layout()
plt.savefig(FIGS / "flair-overview.png", dpi=130, bbox_inches="tight")
plt.show()
'''),
("markdown", """
## Verdict

`worked` / `failed` / `parked`, two or three lines, then copy into `notes.md`.

State the GSD answer separately from the transfer answer — they are different
findings and the next probe needs both.

### What is *not* established
"""),
]


# ==========================================================================
# PoC C — LandCover.ai fine-tune
# ==========================================================================

C = [
("markdown", """
# landcoverai-finetune — can a Polish land-cover model give us `track`?

**Status:** `exploring`

LandCover.ai is the closest annotated dataset that exists to this problem:
**rural Poland**, 0.25 m (33 orthophotos) and 0.5 m (8), 216.27 km², manually
fine-annotated, split into 10,674 tiles of 512x512 (7,470 train / 1,602 val /
1,602 test). Classes: **building, woodland, water, road** — four of our five,
and it is the **only candidate with a road class**, which is the class nothing
else supplies.

Class areas: woodland 72.02 km², water 13.15 km², road 3.50 km², building
1.85 km². Roads are 1.6% of the labelled area — the class we most need is the
rarest, and that has to be handled in the loss, not discovered afterwards.

**Licence: CC-BY-NC-SA 4.0.** Non-commercial *and* share-alike. Fine for offline
label generation and academic publication; the share-alike may attach to weights
trained on it. Not a blocker, but it must be in the dataset card.

### This probe is gated on `flair-landcover-transfer`

Both ask the same underlying question — *does a Western-European land-cover model
transfer to Ukrainian Esri tiles?* — and FLAIR answers it with no training at
all. **If FLAIR fails on `building` and `water` at both GSDs, stop here.**

### Compute reality, stated before starting

This machine is an **Intel i7-9750H, 6 cores, 16 GB, no CUDA, no MPS**. A full
ResNet34-UNet run over 7,470 tiles at 512x512 is days, not hours. Stage 0 below
**measures** seconds/batch on this hardware and prints the extrapolated epoch
time, so the decision to train is made against a number rather than a hope.
Training is opt-in (`train=False` by default).

### What would make this `failed`

- `road` IoU on LandCover.ai's own test split is so low the class was never
  learned — a Polish-data failure, before Ukraine is involved.
- The model transfers on building/water but `road` collapses over Ukrainian
  tiles — paved-road bias, the failure the literature already reports for
  unpaved tracks.
- Predicted tracks disagree across two dates of the same ground.

Run from `src/` with the `uavloc` env.
`papermill pocs/landcoverai-finetune/probe.ipynb pocs/landcoverai-finetune/runs/probe-$(date +%F).ipynb -p train True`
"""),
("code", """
# papermill parameters
SLUG = "landcoverai-finetune"
n = 9                  # Ukrainian locations for the transfer eval
train = False          # stage 1 is opt-in — measure the budget first
crop = 256             # smaller than LandCover.ai's 512: 4x cheaper per sample
batch = 8
epochs = 6
max_train_tiles = 2000 # subset; a probe does not need the full 7,470
lr = 3e-4
encoder = "resnet34"
data_url = "https://landcover.ai.linuxpolska.com/download/landcover.ai.v1.zip"
"""),
("markdown", "## Setup"),
("code", SETUP),
("code", '''
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

torch.set_num_threads(6)
print("torch", torch.__version__, "| smp", smp.__version__)

DATA = Path("data/derived/landcoverai")
DATA.mkdir(parents=True, exist_ok=True)

# LandCover.ai mask values -> our names. Index 0 is unlabelled/other.
LCAI = {0: "background", 1: "building", 2: "woodland", 3: "water", 4: "road"}
OURS = ["background", "building", "track", "woody", "water"]
LCAI_TO_OURS = {0: 0, 1: 1, 2: 3, 3: 4, 4: 2}
'''),
("markdown", """
## Stage 0a — get the data

~1.4-2.9 GB zipped. **29 GB free on this disk**, and extracted tiles add several
more, so check before unpacking. `split.py` ships with the archive and cuts the
41 orthophotos into 512x512 tiles listed by `train.txt` / `val.txt` / `test.txt`.
"""),
("code", '''
import shutil

free_gb = shutil.disk_usage(DATA).free / 2**30
print(f"free disk: {free_gb:.1f} GB")
assert free_gb > 12, "not enough room to unpack LandCover.ai — free space first"

zip_path = DATA / "landcover.ai.v1.zip"
if not zip_path.exists():
    print("downloading — this is the slow part")
    urllib.request.urlretrieve(data_url, zip_path)
print(f"{zip_path.name}: {zip_path.stat().st_size / 2**30:.2f} GB")

if not (DATA / "images").is_dir():
    import zipfile
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(DATA)
print(sorted(p.name for p in DATA.iterdir()))
'''),
("code", '''
# split.py ships in the archive and writes output/<tile>.jpg + <tile>_m.png
if not (DATA / "output").is_dir():
    import subprocess
    print("tiling — a few minutes")
    subprocess.run([sys.executable, "split.py"], cwd=DATA, check=True)

def split_list(which):
    return [l.strip() for l in (DATA / f"{which}.txt").read_text().split("\\n") if l.strip()]

splits = {w: split_list(w) for w in ("train", "val", "test")}
for w, v in splits.items():
    print(f"{w:>6s}: {len(v):>6,} tiles")
'''),
("markdown", """
## Stage 0b — the compute budget, measured

One forward+backward on **this** CPU, timed. Everything downstream is arithmetic
from this number. If the extrapolated run is longer than a night, the honest move
is to shrink the probe or rent a GPU — not to start it and hope.
"""),
("code", '''
def build_model():
    m = smp.Unet(encoder_name=encoder, encoder_weights="imagenet",
                 in_channels=3, classes=len(OURS))
    return m


probe_model = build_model()
opt = torch.optim.AdamW(probe_model.parameters(), lr=lr)
x = torch.randn(batch, 3, crop, crop)
y = torch.randint(0, len(OURS), (batch, crop, crop))
lossfn = nn.CrossEntropyLoss()

probe_model.train()
t0 = time.time()
for _ in range(3):
    opt.zero_grad()
    lossfn(probe_model(x), y).backward()
    opt.step()
per_batch = (time.time() - t0) / 3

n_tiles = min(max_train_tiles, len(splits["train"]))
per_epoch = per_batch * n_tiles / batch
print(f"seconds/batch (b={batch}, {crop}px): {per_batch:.2f}")
print(f"tiles/epoch: {n_tiles:,}  ->  {per_epoch/60:.1f} min/epoch")
print(f"{epochs} epochs: {per_epoch*epochs/3600:.1f} hours")
print(f"\\nfull dataset at 512px would be ~{per_batch*4*len(splits['train'])/batch*epochs/3600:.0f} h")
print("\\n-> set train=True only if the number above is acceptable")
'''),
("markdown", """
## Stage 1 — fine-tune (opt-in)

From an ImageNet encoder, 256 px crops, a subset, few epochs. This is a **probe
model**, not a contender for LandCover.ai's 85.6% mIoU benchmark — the question
is whether `road` is learnable at all and whether it survives the trip to
Ukraine, not whether we can match the paper.

`road` is 1.6% of labelled area, so the loss is class-weighted. Without that the
model predicts "not road" everywhere and reports a fine accuracy — the same
shape of mistake as `bug-016`.
"""),
("code", '''
from torch.utils.data import Dataset, DataLoader


class LandCoverAI(Dataset):
    def __init__(self, names, crop=256, augment=False):
        self.names, self.crop, self.augment = names, crop, augment

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        stem = self.names[i]
        img = np.array(Image.open(DATA / "output" / f"{stem}.jpg").convert("RGB"))
        msk = np.array(Image.open(DATA / "output" / f"{stem}_m.png"))
        # split.py writes masks with cv2.imread() and no grayscale flag, so every
        # mask PNG is 3-channel with the three channels identical. Feeding that
        # to CrossEntropyLoss gives a (H, W, 3) target and a shape error.
        if msk.ndim == 3:
            msk = msk[..., 0]
        if self.crop < img.shape[0]:
            top = np.random.randint(0, img.shape[0] - self.crop + 1)
            left = np.random.randint(0, img.shape[1] - self.crop + 1)
            img = img[top:top + self.crop, left:left + self.crop]
            msk = msk[top:top + self.crop, left:left + self.crop]
        if self.augment and np.random.rand() < 0.5:
            img, msk = img[:, ::-1].copy(), msk[:, ::-1].copy()
        out = np.zeros_like(msk)
        for src, dst in LCAI_TO_OURS.items():
            out[msk == src] = dst
        x = (img.astype(np.float32) / 255.0 - 0.5) / 0.25
        return torch.from_numpy(x.transpose(2, 0, 1)), torch.from_numpy(out).long()


model = build_model()
ckpt = HERE / "model.pt"

if train:
    # num_workers=0 deliberately. macOS multiprocessing uses *spawn*, and a class
    # defined in a notebook cell cannot be re-imported by the child, so any
    # num_workers>0 dies with "Can't pickle <class 'LandCoverAI'>". It would fail
    # identically under papermill. In-process loading also avoids contending with
    # torch's 6 intra-op threads, which already saturate this CPU.
    tr = DataLoader(LandCoverAI(splits["train"][:max_train_tiles], crop, True),
                    batch_size=batch, shuffle=True, num_workers=0, drop_last=True)
    va = DataLoader(LandCoverAI(splits["val"][:400], crop), batch_size=batch, num_workers=0)

    # Median-frequency balancing, from pixel counts MEASURED over 150 random
    # train tiles (2026-09-24) rather than guessed:
    #   background 63.50%  building 0.87%  track 1.52%  woody 26.97%  water 7.15%
    # w_c = median(f) / f_c. The earlier hand-picked vector had track weighted
    # above building while building is the rarer class -- exactly the guess-before-
    # measuring mistake recorded for MIN_BUILDING_PX (bug-012).
    w = torch.tensor([0.113, 8.22, 4.70, 0.265, 1.0])
    lossfn = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * len(tr))

    for ep in range(epochs):
        model.train()
        t0, run = time.time(), 0.0
        for i, (x, y) in enumerate(tr):
            opt.zero_grad()
            loss = lossfn(model(x), y)
            loss.backward()
            opt.step()
            sched.step()
            run += loss.item()
            if i % 25 == 0:
                print(f"  ep{ep} {i}/{len(tr)} loss {run/(i+1):.4f} "
                      f"({(time.time()-t0)/60:.1f} min)", flush=True)

        model.eval()
        inter = np.zeros(len(OURS))
        union = np.zeros(len(OURS))
        with torch.no_grad():
            for x, y in va:
                p = model(x).argmax(1)
                for c in range(len(OURS)):
                    inter[c] += ((p == c) & (y == c)).sum().item()
                    union[c] += ((p == c) | (y == c)).sum().item()
        ious = inter / np.maximum(union, 1)
        print(f"ep{ep} val IoU " + "  ".join(f"{c}={v:.3f}" for c, v in zip(OURS, ious)))
        torch.save(model.state_dict(), ckpt)
elif ckpt.exists():
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    print("loaded", ckpt)
else:
    print("no checkpoint and train=False — stage 2 will be skipped")
model.eval()
'''),
("markdown", """
## Stage 2 — does it cross the border?

Same control as every other probe: OSM `building` and `water` over the Ukrainian
tiles, offset-corrected, with a null baseline. `track` has **no** control — OSM
does not map the tracks we care about, which is the whole reason for this class —
so it is scored two other ways:

- **cross-date consistency**, which needs no truth
- **eyes, at full resolution**, which is what `bug-014` cost us last time

A `track` number without one of those is not evidence.
"""),
("code", '''
@torch.no_grad()
def predict_ua(img, mpp, target_mpp=0.25):
    """LandCover.ai is 0.25 m/px; our tiles are 0.386. Resample to the training
    GSD, tile, predict, map back. Compare by ground scale, never by pixel count
    (the mistake recorded in pocs/drone-frame-boundaries)."""
    w0, h0 = img.size
    scale = mpp / target_mpp
    im = img.resize((int(w0 * scale), int(h0 * scale)), Image.BICUBIC)
    arr = (np.asarray(im, np.float32) / 255.0 - 0.5) / 0.25
    h, w = arr.shape[:2]
    logits = np.zeros((len(OURS), h, w), np.float32)
    count = np.zeros((h, w), np.float32)
    step = crop
    for top in range(0, h, step):
        for left in range(0, w, step):
            t, l = min(top, max(h - crop, 0)), min(left, max(w - crop, 0))
            tile = arr[t:t + crop, l:l + crop]
            ph, pw = tile.shape[:2]
            if ph < crop or pw < crop:
                tile = np.pad(tile, ((0, crop - ph), (0, crop - pw), (0, 0)))
            out = model(torch.from_numpy(tile.transpose(2, 0, 1))[None])[0].numpy()
            logits[:, t:t + ph, l:l + pw] += out[:, :ph, :pw]
            count[t:t + ph, l:l + pw] += 1
    logits /= np.maximum(count, 1)
    pred = logits.argmax(0).astype(np.uint8)
    return np.array(Image.fromarray(pred).resize((w0, h0), Image.NEAREST))


names = locations()[:n]
transfer, preds = [], {}
if train or ckpt.exists():
    for name in names:
        img, _, meta = load_location(name)
        p = predict_ua(img, meta["mpp"])
        preds[name] = p
        row = {"location": name}
        for i, cls in enumerate(OURS):
            row[cls] = {"pred_frac": float((p == i).mean())}
        for cls, osm_cls in (("building", "building"), ("water", "water"), ("woody", "wood")):
            m, t = p == OURS.index(cls), osm_mask(name, osm_cls)
            row[cls].update(iou=iou(m, t), null_iou=null_iou(m, t),
                            recall=float((m & t).sum() / t.sum()) if t.sum() else float("nan"))
        transfer.append(row)
        print(f"{name:30s} building IoU {row['building']['iou']:.3f} "
              f"(null {row['building']['null_iou']:.3f})  "
              f"track claims {row['track']['pred_frac']:.2%}")
    (HERE / "results-transfer.json").write_text(json.dumps(transfer, indent=1))
'''),
("markdown", "## Numbers"),
("code", '''
import statistics as st

if transfer:
    print(f"{'class':>10s} {'IoU':>7s} {'null':>7s} {'recall':>7s} {'claims':>8s}")
    for cls in OURS:
        rows = [r[cls] for r in transfer]
        if "iou" in rows[0]:
            m = lambda k: st.median([r[k] for r in rows if not np.isnan(r[k])])  # noqa: E731
            print(f"{cls:>10s} {m('iou'):>7.3f} {m('null_iou'):>7.3f} "
                  f"{m('recall'):>7.3f} {m('pred_frac'):>8.2%}")
        else:
            print(f"{cls:>10s} {'—':>7s} {'—':>7s} {'—':>7s} "
                  f"{st.median([r['pred_frac'] for r in rows]):>8.2%}   (no control)")
'''),
("markdown", """
## Look at it

`track` has no control, so this figure **is** the evidence for that class.
Full resolution. The question to answer by eye: are the predicted tracks on the
actual lanes and farm roads, or are they field edges and shadows?
"""),
("code", '''
palette = {"building": (1, 0, 0), "track": (1, 0.85, 0),
           "woody": (0, 0.8, 0), "water": (0, 0.4, 1)}
show = [n_ for n_ in names[:3] if n_ in preds]
if show:
    fig, axes = plt.subplots(len(show), 3, figsize=(16, 5.2 * len(show)))
    axes = np.atleast_2d(axes)
    for row, name in enumerate(show):
        img, _, _ = load_location(name)
        arr = np.array(img)
        p = preds[name]
        over = np.zeros((*arr.shape[:2], 3))
        for cls, colour in palette.items():
            m = p == OURS.index(cls)
            for c in range(3):
                over[..., c] = np.where(m, colour[c], over[..., c])
        axes[row, 0].imshow(arr)
        axes[row, 0].set_title(f"{name}\\nimagery", fontsize=8)
        axes[row, 1].imshow(arr)
        axes[row, 1].imshow(over, alpha=0.5)
        axes[row, 1].set_title("prediction (yellow = track)", fontsize=8)

        trk = p == OURS.index("track")
        ys, xs = np.nonzero(trk)
        if len(ys):
            cy, cx = int(np.median(ys)), int(np.median(xs))
            h = 320
            sl = (slice(max(0, cy - h), cy + h), slice(max(0, cx - h), cx + h))
            axes[row, 2].imshow(arr[sl])
            axes[row, 2].imshow(over[sl], alpha=0.5)
        axes[row, 2].set_title("native-resolution zoom — is that a track?", fontsize=8)
        for a in axes[row]:
            a.axis("off")
    plt.tight_layout()
    plt.savefig(FIGS / "landcoverai-transfer.png", dpi=130, bbox_inches="tight")
    plt.show()
'''),
("markdown", """
## Verdict

`worked` / `failed` / `parked`, two or three lines, then copy into `notes.md`.

Report **two** verdicts, they are separate findings:
1. Did `road` learn on LandCover.ai's own Polish test split?
2. Did it survive the move to Ukrainian Esri tiles?

A yes to (1) and no to (2) is the paved-road-bias result and it is worth writing
up properly — it would close the `track`-from-a-model route.

### What is *not* established
"""),
]


if __name__ == "__main__":
    write("ms-building-footprints", A)
    write("flair-landcover-transfer", B)
    write("landcoverai-finetune", C)
