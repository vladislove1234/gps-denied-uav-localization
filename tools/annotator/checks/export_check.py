"""Export bookkeeping check (M5), on a scratch workspace over real refs.

What the e2e run cannot see from the browser:
  * overlap priority: a building drawn over woody exports as building
  * a location with no override exports one shared label.png; adding an
    override switches it to label-<date>.png and removes the old file
  * --stale picks up an edit made after export; --new does not re-export
  * bumping classes_version flags every exported sample for review
  * seeding copies reference geometry bit-for-bit

    python tools/annotator/checks/export_check.py        (uavloc env, from src/)
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

SRC = Path(__file__).resolve().parents[3]
root = Path(tempfile.mkdtemp(prefix="annot-export-"))
(root / "refs").symlink_to(SRC / "data/derived/annotation/refs")
os.environ["ANNOT_ROOT"] = str(root)
os.environ["ANNOT_EXPORT"] = str(root / "export")
sys.path.insert(0, str(SRC))

import numpy as np                                    # noqa: E402
from PIL import Image                                 # noqa: E402

from tools.annotator import config                    # noqa: E402
from tools.annotator.geo import Grid                  # noqa: E402
from tools.annotator.refs import base as refs         # noqa: E402
from tools.annotator.server import export as ex       # noqa: E402
from tools.annotator.server.workspace import Workspace  # noqa: E402

SLUG = "birky-49p9105-23p9192"
results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))


try:
    ws = Workspace(SLUG)
    st = ws.open()
    grid = Grid.from_meta(ws.meta)
    dates = list(st["dates"])

    # seeding is a bit-exact copy of the reference geometry
    seeded = {f["properties"]["ref"]: f["geometry"] for f in ws.read_vectors(("base",))
              if f["properties"]["sublayer"] == "ms-footprints"}
    ref = {f"ms-footprints:{f['id']}": f["geometry"] for f in refs.read_vectors("ms-footprints", SLUG)}
    check("seed copies ms-footprints geometry exactly", seeded == ref, f"{len(seeded)} footprints")
    m_ref = refs.read_mask("flair", SLUG, "woody")
    check("seed copies flair woody mask exactly", np.array_equal(ws.read_mask(("base", "woody", "flair")), m_ref))

    # a building square drawn right over a woody area
    wood = ws.read_mask(("base", "woody", "flair"))
    ys, xs = np.nonzero(wood[400:2200, 400:3000])
    cy, cx = int(ys[len(ys) // 2]) + 400, int(xs[len(xs) // 2]) + 400
    sq = [(cx - 20, cy - 20), (cx + 20, cy - 20), (cx + 20, cy + 20), (cx - 20, cy + 20)]
    ring = grid.ring_to_lonlat(sq); ring.append(ring[0])
    feats = ws.read_vectors(("base",))
    feats.append({"type": "Feature", "id": "test-sq", "properties": {"class": "building", "sublayer": "manual", "kind": "polygon"},
                  "geometry": {"type": "Polygon", "coordinates": [ring]}})
    ws.write_vectors(("base",), feats)
    for d in dates:
        st["dates"][d]["status"] = "done"
    ws.write_state(st)

    s1 = ex.export("new", log=lambda *_: None)
    out = config.EXPORT_DATA / SLUG
    check("no override -> one shared label.png", sorted(p.name for p in out.glob("*.png")) == ["label.png"])
    check("one record per date", s1["written"] == len(dates), str(s1["written"]))
    lab = np.array(Image.open(out / "label.png"))
    check("building wins over woody inside the square", lab[cy, cx] == config.cls("building")["id"], str(lab[cy, cx]))
    ring_px = lab[cy - 30, cx]      # 10 px outside the square, still woody ground
    check("woody kept around it", ring_px == config.cls("woody")["id"] or not wood[cy - 30, cx], str(ring_px))
    check("export --new again writes nothing", ex.export("new", log=lambda *_: None)["written"] == 0)

    # an edit after export makes the samples stale
    m = ws.read_mask(("base", "woody", "manual"))
    m = np.zeros(ws.shape, bool) if m is None else m
    m[100:140, 100:140] = True
    ws.write_mask(("base", "woody", "manual"), m)
    p = ex.plan("stale")
    check("base edit -> every date stale", len(p["stale"]) == len(dates), str(len(p["stale"])))
    ex.export("stale", log=lambda *_: None)
    check("re-export clears stale", len(ex.plan("stale")["stale"]) == 0)

    # an override on the oldest date switches the layout to per-date files
    old = dates[0]
    add = np.zeros(ws.shape, bool); add[1000:1040, 1000:1040] = True
    ws.write_mask(("add", old, "building", "manual"), add)
    p = ex.plan("stale")
    check("override edit -> only that date stale", [k[1] for k in p["stale"]] == [old], str(p["stale"]))
    ex.export("stale", log=lambda *_: None)
    files = sorted(q.name for q in out.glob("*.png"))
    check("layout switched to per-date labels, label.png removed",
          files == sorted(f"label-{d}.png" for d in dates), str(files))
    a = np.array(Image.open(out / f"label-{old}.png")); b = np.array(Image.open(out / "label-current.png"))
    diff = a != b
    ys, xs = np.nonzero(diff)
    check("dates differ exactly inside the override square",
          diff.sum() > 0 and ys.min() >= 1000 and ys.max() < 1040 and xs.min() >= 1000 and xs.max() < 1040, f"{int(diff.sum())} px")
    recs = ex.read_records()
    check("records point at the per-date files", all(r["label_file"].endswith(f"label-{d}.png") for (s, d), r in recs.items()))

    # per-date image alignment: only that date's label moves, by the offset
    d1 = dates[1]
    hashes_before = {d: ex.annotation_hash(ws, d) for d in dates}
    cur_before = np.array(Image.open(out / "label-current.png"))
    st = ws.open(); st["dates"][d1]["offset_px"] = [5, -3]; ws.write_state(st)
    hashes_after = {d: ex.annotation_hash(ws, d) for d in dates}
    check("offset changes only that date's hash", [d for d in dates if hashes_before[d] != hashes_after[d]] == [d1])
    ex.export("stale", log=lambda *_: None)
    a1 = np.array(Image.open(out / f"label-{d1}.png"))
    cur = np.array(Image.open(out / "label-current.png"))
    check("dates without an offset export exactly as before", np.array_equal(cur, cur_before))
    # layer point (x, y) -> image pixel (x - 5, y + 3): a1[v, u] == cur[v - 3, u + 5]
    h, w = cur.shape
    check("offset date label = current label shifted by (-dx, -dy)",
          np.array_equal(a1[3:, :w - 5], cur[:h - 3, 5:]) and not a1[:3].any() and not a1[:, w - 5:].any())
    check("record carries image_offset_px", ex.read_records()[(SLUG, d1)]["image_offset_px"] == [5, -3])
    st = ws.open(); st["dates"][d1].pop("offset_px"); ws.write_state(st)
    check("removing the offset restores the original hash", ex.annotation_hash(ws, d1) == hashes_before[d1])
    ex.export("stale", log=lambda *_: None)

    # registry bump -> needs review
    reg = config.registry()
    reg["version"] = reg["version"] + 1
    check("classes_version bump flags every sample", len(ex.needs_review()) == len(dates), str(len(ex.needs_review())))
    check("... and makes them stale for re-export", len(ex.plan("stale")["stale"]) == len(dates))
    reg["version"] -= 1

    # samples.jsonl record content
    r = recs[(SLUG, "current")]
    need = {"location", "date", "label_file", "image_file", "done_at", "exported_at", "annotation_hash",
            "classes_version", "refs", "provenance", "mpp", "bounds", "shape", "excluded_dates"}
    check("record carries the spec fields", need <= set(r), str(sorted(need - set(r))))
    check("provenance sums sublayer shares per class", abs(sum(v["frac"] for v in r["provenance"].values())
                                                             - (np.array(Image.open(out / "label-current.png")) > 0).mean()) < 1e-9)
    print(json.dumps({k: r[k] for k in ("refs", "provenance")}, indent=1)[:1500])
finally:
    shutil.rmtree(root, ignore_errors=True)

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
