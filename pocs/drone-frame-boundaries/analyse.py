"""How much ground does a drone frame actually contain, in parcels?

Runs DA once over the 790 m Esri mosaic -- all the context the model could want
-- then asks how many of those parcels a 48 x 27 m drone footprint can hold.
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np
from PIL import Image

SRC = Path("/Users/vladyslavtaraban/Documents/Projects/UAV_Localization/src")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC / "pocs/map-derived-labels"))
from tiles import TileGrid            # noqa: E402
import boundaries                     # noqa: E402
sys.path.insert(0, str(HERE))
from run import poses, sat_crop, FOV_W_DEG, FOV_H_DEG   # noqa: E402


def polygon_area_px(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def main():
    gm = json.load(open(HERE / "gonchary-grid.json"))
    grid = TileGrid(z=gm["z"], x0=gm["x0"], y0=gm["y0"], nx=gm["nx"], ny=gm["ny"])
    mpp = grid.mpp()[0]
    mosaic = Image.open(HERE / "satelites/gonchary-esri-z18.jpg").convert("RGB")

    model = boundaries.load("v2")
    wide = boundaries.predict(model, mosaic, tile=1024, overlap=128, conf=0.15)
    areas = np.array([polygon_area_px(p) * mpp * mpp for p in wide])
    cents = np.array([p.mean(axis=0) for p in wide]) if wide else np.zeros((0, 2))
    print(f"control: {len(wide)} parcels over {mosaic.width*mpp:.0f} x {mosaic.height*mpp:.0f} m")
    if len(areas):
        print(f"  parcel area  median {np.median(areas):,.0f} m^2 "
              f"({np.median(areas)/1e4:.2f} ha)   p10 {np.percentile(areas,10):,.0f}"
              f"   p90 {np.percentile(areas,90):,.0f} m^2")
        print(f"  typical parcel side ~ {math.sqrt(np.median(areas)):.0f} m")
        print(f"  density: 1 parcel per {mosaic.width*mosaic.height*mpp*mpp/len(wide):,.0f} m^2")

    ps = [p for p in poses() if p["alt"] > 35.0]
    step = max(1, len(ps) // 12)
    inside, frames = [], ps[::step][:12]
    for pose in frames:
        cx, cy = grid.to_pixel(pose["lat"], pose["lon"])
        half_w, half_h = pose["w_m"] / mpp / 2, pose["h_m"] / mpp / 2
        r = math.hypot(half_w, half_h)          # heading-agnostic: circumscribing disc
        n = int(((cents[:, 0] - cx) ** 2 + (cents[:, 1] - cy) ** 2 < r * r).sum()) if len(cents) else 0
        inside.append(n)
    fa = frames[0]["w_m"] * frames[0]["h_m"]
    print(f"\ndrone footprint {frames[0]['w_m']:.0f} x {frames[0]['h_m']:.0f} m = {fa:,.0f} m^2"
          f"  ({fa/np.median(areas):.2f} median parcels)")
    print(f"control parcels whose centroid falls in a drone footprint: "
          f"{inside}  -> total {sum(inside)} across {len(frames)} frames")

    json.dump(dict(control_parcels=len(wide),
                   parcel_area_m2=dict(median=float(np.median(areas)),
                                       p10=float(np.percentile(areas, 10)),
                                       p90=float(np.percentile(areas, 90))),
                   footprint_m2=fa,
                   parcels_in_footprint=inside),
              open(HERE / "scale.json", "w"), indent=1)
    np.save(HERE / "satelites/control_polys.npy",
            np.array([p for p in wide], dtype=object), allow_pickle=True)


if __name__ == "__main__":
    main()
