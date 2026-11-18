"""The evidence: what a drone frame covers, against what a label is made of."""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon as MplPoly
from matplotlib.lines import Line2D
from PIL import Image

SRC = Path("/Users/vladyslavtaraban/Documents/Projects/UAV_Localization/src")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC / "pocs/map-derived-labels")); sys.path.insert(0, str(HERE))
from tiles import TileGrid                                  # noqa: E402
from geo import Bounds                                      # noqa: E402
import osm                                                  # noqa: E402
from run import poses, sat_crop, FOV_W_DEG, FOV_H_DEG       # noqa: E402

gm = json.load(open(HERE / "gonchary-grid.json"))
grid = TileGrid(z=gm["z"], x0=gm["x0"], y0=gm["y0"], nx=gm["nx"], ny=gm["ny"])
mpp = grid.mpp()[0]
mosaic = Image.open(HERE / "satelites/gonchary-esri-z18.jpg").convert("RGB")
polys = np.load(HERE / "satelites/control_polys.npy", allow_pickle=True)

ps = [p for p in poses() if p["alt"] > 35.0]
chosen = ps[::max(1, len(ps) // 12)][:12]
track = np.array([grid.to_pixel(p["lat"], p["lon"]) for p in ps[::5]])

M = 400 / 111320.0
els = osm.fetch(Bounds(top=49.76210+M, left=24.16848-M*1.55,
                       bottom=49.76026-M, right=24.17332+M*1.55),
                cache_dir=SRC / "data/derived/osm-cache")

fig = plt.figure(figsize=(15.5, 7.4))
gs = fig.add_gridspec(2, 3, width_ratios=[1.6, 1, 1], hspace=0.3, wspace=0.16)

ax = fig.add_subplot(gs[:, 0])
ax.imshow(mosaic)
for p in polys:
    ax.add_patch(MplPoly(p, closed=True, fill=False, ec="#ffd166", lw=0.9, alpha=0.9))
for e in els:
    t = e.get("tags") or {}
    g = e.get("geometry") or []
    if not g or osm.classify(t) is None:
        continue
    xy = np.array([grid.to_pixel(q["lat"], q["lon"]) for q in g])
    ax.plot(xy[:, 0], xy[:, 1], color="#4361ee", lw=1.3, alpha=0.95)
ax.plot(track[:, 0], track[:, 1], ".", color="#4cc9f0", ms=1.1, alpha=0.6)
for pose in chosen:
    cx, cy = grid.to_pixel(pose["lat"], pose["lon"])
    w, h = pose["w_m"] / mpp, pose["h_m"] / mpp
    ax.add_patch(Rectangle((cx - w/2, cy - h/2), w, h, fill=False, ec="#ef476f", lw=2.0))
ax.add_patch(Rectangle((18, mosaic.height-40), 100/mpp, 7, color="w"))
ax.text(20, mosaic.height-48, "100 m", color="w", fontsize=9, weight="bold")
ax.set_xlim(0, mosaic.width); ax.set_ylim(mosaic.height, 0); ax.axis("off")
ax.set_title("Gonchary, Esri World Imagery z18 — 790 m across", fontsize=11, loc="left")
ax.legend(handles=[
    Line2D([], [], color="#ffd166", lw=1.6, label="91 Delineate Anything parcels"),
    Line2D([], [], color="#4361ee", lw=1.6, label="254 OSM ways (road / building)"),
    Line2D([], [], color="#4cc9f0", marker=".", ls="", ms=7, label="2025 flight track"),
    Line2D([], [], color="#ef476f", lw=2.0, label="camera footprint, 41 m AGL"),
], loc="lower right", fontsize=8.5, framealpha=0.85)

pose = chosen[1]
drone = Image.open(pose["path"]).convert("RGB")
sat = sat_crop(mosaic, grid, pose)
for k, (im, title) in enumerate([
        (drone.resize((512, int(512*pose["h_m"]/pose["w_m"])), Image.LANCZOS),
         f"drone frame at {pose['alt']:.0f} m AGL\n{pose['w_m']:.0f} × {pose['h_m']:.0f} m of ground · 2.5 cm/px"),
        (sat, f"the same ground from Esri z18\n{sat.width} × {sat.height} px · 38.6 cm/px")]):
    a = fig.add_subplot(gs[0, 1+k]); a.imshow(im); a.axis("off")
    a.set_title(title, fontsize=9, loc="left")

a = fig.add_subplot(gs[1, 1])
areas = np.array([0.5*abs(np.dot(p[:,0], np.roll(p[:,1],1)) - np.dot(p[:,1], np.roll(p[:,0],1)))*mpp*mpp
                  for p in polys]) / 1e4
a.hist(areas, bins=26, color="#ffd166", ec="#7a5c00", lw=0.6)
fa = pose["w_m"]*pose["h_m"]/1e4
a.axvline(fa, color="#ef476f", lw=2.4)
a.annotate(f"drone frame  {fa:.2f} ha", (fa, a.get_ylim()[1]*0.86), xytext=(1.1, a.get_ylim()[1]*0.86),
           color="#ef476f", fontsize=9, va="center",
           arrowprops=dict(arrowstyle="->", color="#ef476f", lw=1.4))
a.set_xlabel("parcel area (ha)"); a.set_ylabel("parcels")
a.set_title(f"median parcel {np.median(areas):.2f} ha — a frame is {fa/np.median(areas)*100:.0f}% of one",
            fontsize=9.5, loc="left")

a = fig.add_subplot(gs[1, 2])
alt = np.linspace(20, 200, 300)
w = 2*alt*np.tan(np.radians(FOV_W_DEG/2)); h = 2*alt*np.tan(np.radians(FOV_H_DEG/2))
per = (w*h/1e4)/np.median(areas)
a.plot(alt, per, color="#06d6a0", lw=2.4)
a.axhline(1.0, color="#666", ls="--", lw=1)
need = alt[np.argmax(per >= 1.0)]
a.axvline(pose["alt"], color="#ef476f", lw=1.6)
a.plot([need], [1.0], "o", color="#06d6a0", ms=7)
a.annotate(f"{need:.0f} m", (need, 1.0), xytext=(need+6, 1.28), fontsize=9, color="#06d6a0")
a.text(pose["alt"]+4, 2.4, "2025 flight\n41 m", color="#ef476f", fontsize=8.5)
a.text(126, 0.82, "one whole parcel in view", color="#666", fontsize=8.5)
a.set_xlabel("altitude AGL (m)"); a.set_ylabel("median parcels per frame")
a.set_title("field-boundary labels need altitude", fontsize=9.5, loc="left")
a.set_ylim(0, 3); a.set_xlim(20, 200)

fig.savefig(HERE/"figures/footprint-vs-parcel.png", dpi=140, bbox_inches="tight")
print(f"wrote figures/footprint-vs-parcel.png; one parcel in view at {need:.0f} m AGL")
