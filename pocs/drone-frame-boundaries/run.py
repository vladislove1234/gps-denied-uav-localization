"""Does Delineate Anything see field boundaries in a drone frame?

The dataset pipeline labels satellite tiles. A drone frame is georeferenced by
its MAVLink pose, so the same coordinates address both -- but only if the model
that draws the labels actually fires on drone imagery.

The control matters more than the test. A drone frame at 40 m covers 48 x 27 m
of ground; a training crop covers 99 x 99 m. If DA fails on the drone frame it
could be the *domain* (rolling shutter, oblique sun, 2.5 cm/px detail) or simply
the *footprint* (not enough ground in view to contain a parcel). So every drone
frame is paired with a satellite crop of the identical ground at the identical
pixel size, and both are run through the same model.
"""
from __future__ import annotations

import bisect, csv, json, math, os, sys
from pathlib import Path

import numpy as np
from PIL import Image

SRC = Path("/Users/vladyslavtaraban/Documents/Projects/UAV_Localization/src")
ARCHIVE = SRC.parent / "archive/uav_localization_2025"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC / "pocs/map-derived-labels"))

from tiles import TileGrid                      # noqa: E402
import boundaries                               # noqa: E402

FOV_W_DEG, FOV_H_DEG, ALT_OFFSET = 63.0, 35.0, -13.0   # 2025 calibration
SCALES = [256, 512, 1024]
N_FRAMES = 12


def poses():
    rows = list(csv.DictReader(open(ARCHIVE / "dataset_utils/gps_data.csv")))
    gt = [int(r["TimeUS"]) for r in rows]
    out = []
    for p in sorted((ARCHIVE / "dataset_utils/image_pairs_v2").glob("*_drone.jpg")):
        ts = int(p.name.split("_")[1])
        i = bisect.bisect_left(gt, ts)
        i = min(range(max(0, i - 1), min(len(gt), i + 2)), key=lambda j: abs(gt[j] - ts))
        r = rows[i]
        alt = float(r["Altitude"]) + ALT_OFFSET
        out.append(dict(path=p, ts=ts, lat=float(r["Latitude"]), lon=float(r["Longitude"]),
                        alt=alt, yaw=float(r["Yaw"]),
                        w_m=2 * alt * math.tan(math.radians(FOV_W_DEG / 2)),
                        h_m=2 * alt * math.tan(math.radians(FOV_H_DEG / 2))))
    return out


def sat_crop(mosaic, grid, pose):
    """The same ground the drone sees, cut from the satellite mosaic and turned
    to the UAV's heading so the two images are comparable pixel for pixel."""
    mpp = grid.mpp()[0]
    cx, cy = grid.to_pixel(pose["lat"], pose["lon"])
    w_px, h_px = pose["w_m"] / mpp, pose["h_m"] / mpp
    pad = int(math.hypot(w_px, h_px) * 0.6) + 8
    box = (int(cx - pad), int(cy - pad), int(cx + pad), int(cy + pad))
    if box[0] < 0 or box[1] < 0 or box[2] > mosaic.width or box[3] > mosaic.height:
        return None
    patch = mosaic.crop(box).rotate(pose["yaw"], resample=Image.BICUBIC)
    mid = patch.width / 2, patch.height / 2
    return patch.crop((int(mid[0] - w_px / 2), int(mid[1] - h_px / 2),
                       int(mid[0] + w_px / 2), int(mid[1] + h_px / 2)))


def main():
    grid_meta = json.load(open(HERE / "gonchary-grid.json"))
    grid = TileGrid(z=grid_meta["z"], x0=grid_meta["x0"], y0=grid_meta["y0"],
                    nx=grid_meta["nx"], ny=grid_meta["ny"])
    mosaic = Image.open(HERE / "satelites/gonchary-esri-z18.jpg").convert("RGB")

    all_poses = [p for p in poses() if p["alt"] > 35.0]
    step = max(1, len(all_poses) // N_FRAMES)
    chosen = all_poses[::step][:N_FRAMES]
    print(f"{len(all_poses)} frames above 35 m; using {len(chosen)}")

    model = boundaries.load("v2")
    records = []
    for k, pose in enumerate(chosen):
        sat = sat_crop(mosaic, grid, pose)
        if sat is None:
            print(f"  [{k}] outside mosaic, skipped"); continue
        drone = Image.open(pose["path"]).convert("RGB")
        rec = dict(ts=pose["ts"], lat=pose["lat"], lon=pose["lon"], alt=round(pose["alt"], 1),
                   yaw=round(pose["yaw"], 1), w_m=round(pose["w_m"], 1), scales={})
        for s in SCALES:
            h = int(s * pose["h_m"] / pose["w_m"])
            d_im = drone.resize((s, h), Image.LANCZOS)
            s_im = sat.resize((s, h), Image.LANCZOS)
            d_poly = boundaries.predict(model, d_im, tile=max(s, 256), overlap=0, conf=0.15)
            s_poly = boundaries.predict(model, s_im, tile=max(s, 256), overlap=0, conf=0.15)
            rec["scales"][s] = dict(drone=len(d_poly), sat=len(s_poly),
                                    mpp=round(pose["w_m"] / s, 3))
            if k < 4:
                for tag, im, poly in (("drone", d_im, d_poly), ("sat", s_im, s_poly)):
                    im.save(HERE / f"figures/_{k:02d}_{tag}_{s}.png")
                    np.save(HERE / f"figures/_{k:02d}_{tag}_{s}.npy",
                            boundaries.rasterise_edges(poly, (h, s)))
        records.append(rec)
        print(f"  [{k}] alt {pose['alt']:.0f} m  " +
              "  ".join(f"{s}px drone={rec['scales'][s]['drone']} sat={rec['scales'][s]['sat']}"
                        for s in SCALES))

    # Control: the whole 790 m mosaic, all the context DA could want.
    wide = boundaries.predict(model, mosaic, tile=1024, overlap=128, conf=0.15)
    print(f"\ncontrol -- full 790 m Esri mosaic at 0.386 m/px: {len(wide)} polygons")
    json.dump(dict(frames=records, control_polygons=len(wide),
                   fov=[FOV_W_DEG, FOV_H_DEG], alt_offset=ALT_OFFSET),
              open(HERE / "results.json", "w"), indent=1)
    np.save(HERE / "figures/_control_edges.npy",
            boundaries.rasterise_edges(wide, (mosaic.height, mosaic.width)))


if __name__ == "__main__":
    main()
