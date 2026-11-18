"""End-to-end check of the annotator in headless Chromium, on real data.

Starts the server on a SCRATCH workspace (refs are shared read-only via a
symlink, the real work/ and export are untouched), drives the UI with real
mouse events, and asserts on what lands on disk. Screenshots go to
``checks/figures/`` for a human to look at.

    # from src/, base env has playwright; the server runs in uavloc
    python tools/annotator/checks/e2e.py [--slug birky-49p9105-23p9192] [--keep]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

SRC = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
FIGS = HERE / "figures"
UAVLOC_PY = "/Users/vladyslavtaraban/miniconda3/envs/uavloc/bin/python"


def free_port() -> int:
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="birky-49p9105-23p9192")
    ap.add_argument("--keep", action="store_true", help="keep the scratch workspace")
    a = ap.parse_args()

    root = Path(tempfile.mkdtemp(prefix="annot-e2e-"))
    (root / "refs").symlink_to(SRC / "data/derived/annotation/refs")
    export_dir = root / "export"
    port = free_port()
    env = {**os.environ, "ANNOT_ROOT": str(root), "ANNOT_EXPORT": str(export_dir)}
    srv = subprocess.Popen([UAVLOC_PY, "-m", "tools.annotator", "serve", "--port", str(port)],
                           cwd=SRC, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    base = f"http://127.0.0.1:{port}"
    for _ in range(60):
        try:
            get(base + "/api/config"); break
        except Exception:
            time.sleep(0.5)
    else:
        print(srv.stdout.read()); raise SystemExit("server did not start")

    FIGS.mkdir(exist_ok=True)
    import hashlib
    jpgs = sorted((SRC / "data/derived/ua-village-multidate" / a.slug).glob("*.jpg"))
    jpg_hash = lambda: {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in jpgs}
    jpg_before = jpg_hash()
    results = []

    def check(name, cond, detail=""):
        results.append((name, bool(cond), detail))
        print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)

    from playwright.sync_api import sync_playwright
    errors = []
    try:
        with sync_playwright() as p:
            br = p.chromium.launch()
            pg = br.new_page(viewport={"width": 1600, "height": 1000}, device_scale_factor=1)
            pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
            pg.on("pageerror", lambda e: errors.append(str(e)))
            pg.goto(f"{base}/#{a.slug}")
            pg.wait_for_function("window.app && app.doc && app.renderer.images.size > 0", timeout=60000)
            pg.wait_for_timeout(1500)
            ev = pg.evaluate
            n_base = ev("app.doc.base.size")
            subs = ev("app.doc.state.sublayers")
            check("workspace seeded", n_base > 0, f"{n_base} base features")
            check("building seeded from osm + ms-footprints",
                  [s["name"] for s in subs["building"]][:2] == ["osm", "ms-footprints"], str([s["name"] for s in subs["building"]]))
            check("woody has 4 source sublayers + manual",
                  [s["name"] for s in subs["woody"]] == ["osm", "landcoverai", "worldcover", "flair", "manual"],
                  str([s["name"] for s in subs["woody"]]))
            check("field starts with only manual", [s["name"] for s in subs["field"]] == ["manual"])
            pg.screenshot(path=FIGS / "e2e-01-open.jpg", type="jpeg", quality=72)

            # -- canvas helpers: image px -> page coords
            box = pg.locator("#canvas").bounding_box()
            def at(x, y):
                sx, sy = ev(f"app.renderer.toScreen({x}, {y})")
                return box["x"] + sx, box["y"] + sy
            def zoom_to(x, y, s):
                ev(f"(() => {{ const r = app.renderer; r.view = {{ s: {s}, tx: r.cw/2 - {x}*{s}, ty: r.ch/2 - {y}*{s} }}; r.request(); }})()")
                pg.wait_for_timeout(200)

            # -- select + move a building (osm sublayer), then undo
            f = ev("""(() => { const f = [...app.doc.base.values()].find(f => f.cls==='building' && f.sub==='osm'
                      && f.bbox.x1-f.bbox.x0 > 15 && f.bbox.x0 > 800 && f.bbox.y0 > 700 && f.bbox.x1 < 2400 && f.bbox.y1 < 1800);
                      return {id: f.id, pts: f.pts, cx: (f.bbox.x0+f.bbox.x1)/2, cy: (f.bbox.y0+f.bbox.y1)/2}; })()""")
            zoom_to(f["cx"], f["cy"], 3)
            ev("app.ui.setActive('building','osm')"); pg.keyboard.press("v")
            x, y = at(f["cx"], f["cy"])
            pg.mouse.move(x, y); pg.mouse.down(); pg.mouse.move(x + 30, y + 15, steps=5); pg.mouse.up()
            moved = ev(f"app.doc.base.get('{f['id']}').pts[0]")
            check("drag moves a building by 10,5 px (zoom 3)",
                  abs(moved[0] - f["pts"][0][0] - 10) < 0.6 and abs(moved[1] - f["pts"][0][1] - 5) < 0.6, str(moved))
            pg.screenshot(path=FIGS / "e2e-02-moved.jpg", type="jpeg", quality=72)
            pg.keyboard.press("Meta+z")
            back = ev(f"app.doc.base.get('{f['id']}').pts[0]")
            check("undo restores it", abs(back[0] - f["pts"][0][0]) < 1e-9, str(back))
            pg.keyboard.press("Meta+Shift+z")
            check("redo re-applies", abs(ev(f"app.doc.base.get('{f['id']}').pts[0]")[0] - moved[0]) < 1e-9)

            # -- rotate via handle
            b = ev("app.tools.selBox()")
            hx, hy = at((b["x0"] + b["x1"]) / 2, b["y0"] - 28 / 3)
            pg.mouse.move(hx, hy); pg.mouse.down(); pg.mouse.move(hx + 60, hy + 40, steps=6); pg.mouse.up()
            rot = ev(f"app.doc.base.get('{f['id']}').pts")
            check("rotation handle rotates", rot[0] != moved, f"{len(rot)} pts")

            # -- rect on building/manual
            ev("app.ui.setActive('building','manual')"); pg.keyboard.press("r")
            n0 = ev("app.doc.base.size")
            x0, y0 = at(f["cx"] + 40, f["cy"] + 40)
            pg.mouse.move(x0, y0); pg.mouse.down(); pg.mouse.move(x0 + 60, y0 + 45, steps=4); pg.mouse.up()
            check("rectangle draws a building", ev("app.doc.base.size") == n0 + 1)
            # rotated 3-click rect
            x1, y1 = at(f["cx"] - 60, f["cy"] + 60)
            pg.mouse.click(x1, y1); pg.mouse.click(x1 + 50, y1 - 20); pg.mouse.click(x1 + 60, y1 + 10)
            check("3-click rotated rectangle", ev("app.doc.base.size") == n0 + 2)

            # -- polygon on field/manual (vector class)
            ev("app.ui.setActive('field','manual')"); pg.keyboard.press("p")
            zoom_to(f["cx"] - 200, f["cy"] - 150, 1)
            px, py = at(f["cx"] - 200, f["cy"] - 150)
            for dx, dy in [(0, 0), (120, 10), (140, 110), (10, 100)]:
                pg.mouse.click(px + dx, py + dy)
            pg.keyboard.press("Enter")
            fields = ev("[...app.doc.base.values()].filter(f=>f.cls==='field').length")
            check("polygon draws a field", fields == 1, str(fields))

            # -- brush on woody/manual (raster)
            ev("app.ui.setActive('woody','manual')"); pg.keyboard.press("b")
            ev("app.tools.opt.brushPx = 20")
            zoom_to(f["cx"] + 150, f["cy"] - 100, 2)
            bx, by = at(f["cx"] + 150, f["cy"] - 100)
            pg.mouse.move(bx, by); pg.mouse.down(); pg.mouse.move(bx + 120, by + 30, steps=10); pg.mouse.up()
            painted = ev("(()=>{const m=app.doc.mask('base','woody','manual'); let s=0; for(const v of m) s+=v; return s})()")
            check("brush paints woody/manual", painted > 1000, f"{painted} px")
            pg.keyboard.press("e")
            pg.mouse.move(bx + 60, by + 15); pg.mouse.down(); pg.mouse.move(bx + 61, by + 16); pg.mouse.up()
            erased = ev("(()=>{const m=app.doc.mask('base','woody','manual'); let s=0; for(const v of m) s+=v; return s})()")
            check("eraser removes pixels", erased < painted, f"{painted} -> {erased}")

            # -- wand on track/manual: click on a LandCover.ai road pixel
            road = ev("""(async () => { const m = await app.doc.refMaskData('landcoverai','track'); const w = app.doc.w;
                          for (let y = 900; y < 1700; y += 7) for (let x = 900; x < 2400; x += 7) {
                            let ok = true; for (let d = -3; d <= 3 && ok; d++) ok = m[(y+d)*w + x] && m[y*w + x + d];
                            if (ok) return [x, y]; } return null; })()""")
            ev("app.ui.setActive('track','manual')"); pg.keyboard.press("w")
            zoom_to(road[0], road[1], 2)
            wx, wy = at(road[0] + 0.5, road[1] + 0.5)
            pg.mouse.move(wx - 3, wy); pg.mouse.move(wx, wy)
            try:
                pg.wait_for_function("!!app.tools.wandPreview && !app.tools.wandBusy", timeout=15000); prev = True
            except Exception:
                prev = False
            check("wand hover shows a preview", prev)
            pg.screenshot(path=FIGS / "e2e-03-wand-preview.jpg", type="jpeg", quality=72)
            pg.mouse.click(wx, wy)
            pg.wait_for_function("(()=>{const m=app.doc.mask('base','track','manual'); return m && m.some(v=>v)})()", timeout=15000)
            wand_px = ev("(()=>{const m=app.doc.mask('base','track','manual'); if(!m) return 0; let s=0; for(const v of m) s+=v; return s})()")
            check("wand click fills track/manual", wand_px > 30, f"{wand_px} px at {road}")

            # -- translate whole woody/osm sublayer by numeric offset
            ev("app.ui.setActive('woody','osm')"); pg.keyboard.press("t")
            ev("window._before = app.doc.mask('base','woody','osm').slice()")
            ev("app.tools.translateBy(3, 0)")
            bad = ev("""(()=>{const a=window._before, m=app.doc.mask('base','woody','osm'), w=app.doc.w, h=app.doc.h; let bad=0, n=0;
                        for (let y=0;y<h;y++) for (let x=0;x<w-3;x++) { n+=a[y*w+x]; if (m[y*w+x+3]!==a[y*w+x]) bad++; }
                        for (let y=0;y<h;y++) for (let x=0;x<3;x++) if (m[y*w+x]) bad++;
                        return [bad, n]})()""")
            check("translate shifts a raster sublayer exactly +3 px", bad[0] == 0 and bad[1] > 0, f"{bad[0]} wrong of {bad[1]}")
            ev("app.doc.undo()")
            check("undo translate", ev("(()=>{const a=window._before, m=app.doc.mask('base','woody','osm'); for(let i=0;i<a.length;i++) if(a[i]!==m[i]) return false; return true})()"))

            # -- sublayer ops: delete worldcover (woody), undo
            ev("app.doc.deleteSublayer('woody','worldcover')")
            check("delete sublayer", "worldcover" not in [s["name"] for s in ev("app.doc.sublayers('woody')")])
            ev("app.doc.undo()")
            check("undo delete sublayer", "worldcover" in [s["name"] for s in ev("app.doc.sublayers('woody')")])
            ev("app.doc.mergeSublayer('woody','flair','landcoverai')")
            check("merge flair into landcoverai", [s["name"] for s in ev("app.doc.sublayers('woody')")] == ["osm", "landcoverai", "worldcover", "manual"])

            # -- per-date override: on the oldest date delete one base building
            dates = ev("app.doc.dates")
            old = dates[0]
            ev(f"app.doc.viewDate = '{old}'; app.ui.afterDateChange(); app.ui.setTarget('date')")
            check("target follows the viewed date", ev("app.doc.target") == old)
            victim = ev("[...app.doc.base.values()].find(f=>f.cls==='building' && f.sub==='ms-footprints').id")
            ev(f"app.tools.selection = new Set(['{victim}']); app.ui.setActive('building','ms-footprints')")
            pg.keyboard.press("v"); pg.keyboard.press("Delete")
            check("override delete hides base feature on that date only",
                  ev(f"app.doc.over['{old}'].removed.has('{victim}')") and ev(f"app.doc.base.has('{victim}')"))
            # new building only on this date
            ev("app.ui.setActive('building','manual')"); pg.keyboard.press("r")
            zoom_to(f["cx"], f["cy"], 3)
            x0, y0 = at(f["cx"] - 30, f["cy"] - 60)
            pg.mouse.move(x0, y0); pg.mouse.down(); pg.mouse.move(x0 + 40, y0 + 30, steps=3); pg.mouse.up()
            check("override adds a date-only building", ev(f"app.doc.over['{old}'].add.size") == 1)
            pg.screenshot(path=FIGS / "e2e-04-override.jpg", type="jpeg", quality=72)
            # align this date's image with the Move tool: drag 12 screen px = 4 image px at zoom 3
            pg.keyboard.press("t")
            mx, my = at(f["cx"], f["cy"])
            pg.mouse.move(mx, my); pg.mouse.down(); pg.mouse.move(mx + 12, my - 6, steps=4); pg.mouse.up()
            off = ev(f"app.doc.state.dates['{old}'].offset_px")
            check("Move tool on a date target aligns that date's image", off == [4, -2], str(off))
            check("... other dates keep no offset", ev(f"Object.entries(app.doc.state.dates).filter(([d,v]) => d !== '{old}' && v.offset_px).length") == 0)
            wp = ev(f"(() => {{ const p = app.renderer.pixelsOf('{old}', {{x: 1004, y: 998, w: 1, h: 1}}).data; return [...p]; }})()")
            src = ev(f"(() => {{ const c = new OffscreenCanvas(3328, 2560), g = c.getContext('2d'); g.drawImage(app.renderer.images.get('{old}'), 0, 0); return [...g.getImageData(1000, 1000, 1, 1).data]; }})()")
            check("wand samples the aligned image (layer px = image px + offset)", wp == src, f"{wp} vs {src}")
            pg.keyboard.press("Meta+z")
            check("undo alignment", ev(f"app.doc.state.dates['{old}'].offset_px") is None)
            pg.keyboard.press("Meta+Shift+z")
            pg.screenshot(path=FIGS / "e2e-04b-align.jpg", type="jpeg", quality=72)
            ev("app.ui.setTarget('base'); app.doc.viewDate = 'current'; app.ui.afterDateChange()")

            # -- compare modes render without errors
            for mode in ("opacity", "swipe", "blink"):
                ev(f"app.ui.compare.mode = '{mode}'; app.ui.compare.dateB = '{old}'; app.ui.renderDates(); app.renderer.request()")
                pg.wait_for_timeout(250)
            ev("app.ui.compare.mode='swipe'; app.renderer.fit()"); pg.wait_for_timeout(400)
            pg.screenshot(path=FIGS / "e2e-05-swipe.jpg", type="jpeg", quality=72)
            ev("app.ui.compare.mode='off'; app.ui.renderDates()")

            # -- ref overlays
            ev("""for (const k of ['ms-footprints|building','landcoverai|track','flair|woody']) app.ui.refOn[k] = {on: true, opacity: .6};
                  app.doc.refVectors('ms-footprints'); app.doc.refMaskData('landcoverai','track'); app.doc.refMaskData('flair','woody');
                  app.ui.renderRefs(); app.renderer.request()""")
            pg.wait_for_timeout(1500); ev("app.renderer.request()"); pg.wait_for_timeout(300)
            pg.screenshot(path=FIGS / "e2e-06-refs.jpg", type="jpeg", quality=72)
            # adopt one ms footprint into building/manual
            ev("app.ui.setActive('building','manual')"); pg.keyboard.press("a")
            ms = ev("(()=>{const f = app.doc.refVec['ms-footprints'].find(f => f.bbox.x0 > 1500 && f.bbox.y0 > 1200); return [(f.bbox.x0+f.bbox.x1)/2,(f.bbox.y0+f.bbox.y1)/2]})()")
            zoom_to(ms[0], ms[1], 3)
            n0 = ev("app.doc.base.size")
            pg.mouse.click(*at(ms[0], ms[1]))
            check("adopt copies a reference footprint", ev("app.doc.base.size") == n0 + 1)
            ev("for (const o of Object.values(app.ui.refOn)) o.on = false; app.renderer.request()")

            # -- conflicts overlay
            nconf = ev("app.renderer.computeConflicts()")
            check("conflict overlay computes", nconf >= 0, f"{nconf} px in 2+ classes")

            # -- status + save
            ev("app.doc.state.dates && Object.keys(app.doc.state.dates).forEach(d => app.doc.state.dates[d].status = 'done'); app.doc.setDate('current', {})")
            ev("app.doc.save()")
            pg.wait_for_function("app.doc.dirty.size === 0 && !app.doc.saving", timeout=60000)
            n_final = ev("app.doc.base.size")

            # -- drone frame: pixel-only, no references, manual layers only
            drone = get(f"{base}/api/locations")
            drone = [l for l in drone if l["collection"] == "drone"]
            check("44 drone frames listed", len(drone) == 44, str(len(drone)))
            dslug = drone[0]["slug"]
            djpg = SRC / "data/derived/drone-2025-easy" / dslug / "frame.jpg"
            djpg_before = hashlib.sha256(djpg.read_bytes()).hexdigest()
            pg.click(f"#loc-list li[data-slug='{dslug}']")
            pg.wait_for_function(f"app.doc && app.doc.slug === '{dslug}' && app.renderer.images.size > 0", timeout=60000)
            pg.wait_for_timeout(500)
            check("drone frame opens with a pixel grid", ev("app.doc.grid.pixel === true && app.doc.w === 1920 && app.doc.h === 1080"))
            check("drone frame has no references", ev("Object.keys(app.doc.refsMeta).length") == 0)
            check("drone sublayers are manual only",
                  all([x["name"] for x in v] == ["manual"] for v in ev("app.doc.state.sublayers").values()))
            check("drone frame has one date", ev("app.doc.dates") == ["frame"])
            ev("app.renderer.fit()"); pg.wait_for_timeout(200)
            box = pg.locator("#canvas").bounding_box()
            ev("app.ui.setActive('building','manual')"); pg.keyboard.press("r")
            x0, y0 = at(600, 300)
            pg.mouse.move(x0, y0); pg.mouse.down(); pg.mouse.move(*at(900, 500), steps=4); pg.mouse.up()
            check("rectangle on a drone frame", ev("app.doc.base.size") == 1)
            geo_pt = ev("app.doc.toGeo([...app.doc.base.values()][0]).geometry.coordinates[0][0]")
            check("drone shapes stored in frame pixels", abs(geo_pt[0] - 600) < 3 and abs(geo_pt[1] - 300) < 3, str(geo_pt))
            ev("app.ui.setActive('track','manual')"); pg.keyboard.press("b"); ev("app.tools.opt.brushPx = 60")
            pg.mouse.move(*at(100, 900)); pg.mouse.down(); pg.mouse.move(*at(1800, 950), steps=12); pg.mouse.up()
            check("brush on a drone frame", ev("(()=>{const m=app.doc.mask('base','track','manual'); let s=0; for(const v of m) s+=v; return s})()") > 50000)
            check("cursor shows px + cm/px, no lat/lon", "cm/px" in pg.inner_text("#cursor"), pg.inner_text("#cursor"))
            pg.screenshot(path=FIGS / "e2e-08-drone.jpg", type="jpeg", quality=72)
            ev("app.doc.setDate('frame', {status: 'done'})")
            ev("app.doc.save()")
            pg.wait_for_function("app.doc.dirty.size === 0 && !app.doc.saving", timeout=60000)

            # -- Gonchary 2025 ortho: pixel-only satellite image
            oslug = "gonchary-2025"
            ojpg = SRC / "data/derived/gonchary-2025-ortho" / oslug / "ortho.jpg"
            ojpg_before = hashlib.sha256(ojpg.read_bytes()).hexdigest()
            pg.click(f"#loc-list li[data-slug='{oslug}']")
            pg.wait_for_function(f"app.doc && app.doc.slug === '{oslug}' && app.renderer.images.size > 0", timeout=60000)
            pg.wait_for_timeout(500)
            check("ortho opens pixel-only at 2731x2009", ev("app.doc.grid.pixel && app.doc.w === 2731 && app.doc.h === 2009"))
            check("ortho: no references, manual sublayers only", ev("Object.keys(app.doc.refsMeta).length") == 0 and
                  all([x["name"] for x in v] == ["manual"] for v in ev("app.doc.state.sublayers").values()))
            ev("app.ui.setActive('woody','manual')"); pg.keyboard.press("w")
            zoom_to(1400, 1000, 1)
            wx, wy = at(1400.5, 1000.5)
            pg.mouse.move(wx - 3, wy); pg.mouse.move(wx, wy)
            pg.wait_for_function("!!app.tools.wandPreview && !app.tools.wandBusy", timeout=15000)
            pg.mouse.click(wx, wy)
            pg.wait_for_function("(()=>{const m=app.doc.mask('base','woody','manual'); return m && m.some(v=>v)})()", timeout=15000)
            check("wand fills on the ortho", True)
            ev("app.renderer.fit()"); pg.wait_for_timeout(300)
            pg.screenshot(path=FIGS / "e2e-09-ortho.jpg", type="jpeg", quality=72)
            ev("app.doc.setDate('ortho', {status: 'done'})")
            ev("app.doc.save()")
            pg.wait_for_function("app.doc.dirty.size === 0 && !app.doc.saving", timeout=60000)
            check("no console errors", not errors, "; ".join(errors[:3]))
            br.close()

        # -- disk state through the API (a fresh read, not the page's memory)
        work = get(f"{base}/api/loc/{a.slug}/work")
        check("saved base features match the page", len(work["base"]) == n_final, f"{len(work['base'])} vs {n_final}")
        check("override saved", len(work["overrides"][old]["add"]) == 1 and victim in work["overrides"][old]["remove"])
        check("masks saved", ["base", "woody", "manual"] in work["masks"] and ["base", "track", "manual"] in work["masks"])
        st = work["state"]
        check("all dates done on disk", all(s["status"] == "done" for s in st["dates"].values()))

        # -- export
        req = urllib.request.Request(f"{base}/api/export?mode=new", method="POST")
        with urllib.request.urlopen(req, timeout=600) as r:
            summary = json.load(r)
        check("export writes every date (+1 drone frame, +1 ortho)", summary["written"] == len(st["dates"]) + 2,
              str({k: summary[k] for k in ("written", "locations")}))
        dexp = Path(str(export_dir) + "-drone")
        dl = np.array(Image.open(dexp / dslug / "label.png"))
        check("drone label in the separate export", dl.shape == (1080, 1920) and dl[400, 750] == 1 and dl[925, 1000] == 2,
              f"{dl.shape} building={dl[400, 750]} track={dl[925, 1000]}")
        drec = [json.loads(l) for l in (dexp / "samples.jsonl").read_text().splitlines()]
        check("drone record: pixel crs + flight position, no bounds",
              len(drec) == 1 and drec[0]["crs"] == "pixel" and drec[0]["position"]["alt_m"] and "bounds" not in drec[0])
        check("satellite samples.jsonl has no drone records", all(not r["location"].startswith("drone-")
              for r in map(json.loads, (export_dir / "samples.jsonl").read_text().splitlines())))
        check("drone JPEG byte-identical", hashlib.sha256(djpg.read_bytes()).hexdigest() == djpg_before)
        oexp = Path(str(export_dir) + "-ortho")
        ol = np.array(Image.open(oexp / oslug / "label.png"))
        orec = [json.loads(l) for l in (oexp / "samples.jsonl").read_text().splitlines()]
        check("ortho label in its own export", ol.shape == (2009, 2731) and ol[1000, 1400] == 3, f"{ol.shape} {ol[1000, 1400]}")
        check("ortho record maps back to the raw frame", len(orec) == 1 and orec[0]["crs"] == "pixel"
              and orec[0]["source_shape"] == [6026, 8192] and abs(orec[0]["scale_to_source"][0] - 3) < 0.01)
        check("ortho JPEG byte-identical", hashlib.sha256(ojpg.read_bytes()).hexdigest() == ojpg_before)
        loc_dir = export_dir / a.slug
        files = sorted(p.name for p in loc_dir.glob("*.png"))
        check("per-date labels (location has an override)", files == sorted(f"label-{d}.png" for d in st["dates"]), str(files))
        recs = [json.loads(l) for l in (export_dir / "samples.jsonl").read_text().splitlines()]
        check("samples.jsonl has one record per date", len(recs) == len(st["dates"]))
        lab_old = np.array(Image.open(loc_dir / f"label-{old}.png"))
        lab_cur = np.array(Image.open(loc_dir / "label-current.png"))
        diff = lab_old != lab_cur
        # old date = override + 4,-2 px alignment; compare in its own frame
        # base env has no cv2, so no import of server.export: same shift inline
        sh = np.zeros_like(lab_cur); h, w = lab_cur.shape
        sh[2:, :w - 4] = lab_cur[:h - 2, 4:]        # out[v, u] = cur[v - 2, u + 4]
        diff = lab_old != sh
        check("old-date label = current shifted, differing only by the override",
              0 < diff.sum() < 20000, f"{int(diff.sum())} px differ")
        check("label uses ids 0..5 only", set(np.unique(lab_cur)) <= set(range(6)), str(np.unique(lab_cur)))
        # plan: nothing new now, nothing stale
        plan = get(f"{base}/api/export/plan")
        check("after export: 0 new, 0 stale", plan["new"] == 0 and plan["stale"] == 0, str(plan))
        check("record of aligned date carries its offset", next(r for r in recs if r["date"] == old)["image_offset_px"] == [4, -2])
        check("source JPEGs byte-identical after the whole session", jpg_hash() == jpg_before, f"{len(jpg_before)} files")
        # overlay figure of exported label on imagery
        img = np.array(Image.open(SRC / "data/derived/ua-village-multidate" / a.slug / "current.jpg").convert("RGB")).astype(float)
        pal = np.array([[0, 0, 0], [255, 59, 92], [255, 176, 32], [46, 204, 64], [212, 225, 87], [30, 144, 255]], float)
        m = lab_cur > 0
        img[m] = img[m] * 0.45 + pal[lab_cur[m]] * 0.55
        Image.fromarray(img.astype(np.uint8)).resize((1664, 1280)).save(FIGS / "e2e-07-export-overlay.jpg", quality=85)
    finally:
        srv.terminate()
        if not a.keep:
            shutil.rmtree(root, ignore_errors=True)
        else:
            print("scratch workspace kept at", root)

    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
