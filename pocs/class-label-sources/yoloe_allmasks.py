"""Record EVERY mask, not just the largest -- does a small well-placed one exist?"""
import sys, json, warnings
import numpy as np
from PIL import Image
warnings.filterwarnings("ignore")
sys.path.insert(0, "pocs/class-label-sources")
import yoloe_probe as probe
import yoloe_smallcrop as small
from ultralytics import YOLOE

IMGSZ, EXTENTS = 640, [25, 50, 100, 200]
names = list(json.load(open("pocs/class-label-sources/results-field.json")))[:9]

model = YOLOE("yoloe-11l-seg.pt")
pr = probe.PROMPTS["satellite"]
model.set_classes(pr, model.get_text_pe(pr))
rng = np.random.default_rng(0)
rows = []

for name in names:
    image, label, meta = probe.load_location(name)
    control = probe.control_masks(label, meta)["building"]
    mpp = meta["mpp"]; h, w = meta["shape"]
    for cx, cy, blob in small.building_centres(control, 8, 60, rng):
        for extent in EXTENTS:
            half = int(round(extent / mpp / 2))
            x0, y0 = max(0, cx - half), max(0, cy - half)
            x1, y1 = min(w, cx + half), min(h, cy + half)
            if x1 - x0 < 16 or y1 - y0 < 16:
                continue
            crop = image.crop((x0, y0, x1, y1)).resize((IMGSZ, IMGSZ), Image.LANCZOS)
            tgt = np.array(Image.fromarray(blob[y0:y1, x0:x1].astype("uint8") * 255)
                           .resize((IMGSZ, IMGSZ), Image.NEAREST)) > 127
            r = model.predict(np.array(crop), imgsz=IMGSZ, conf=0.05, device="cpu",
                              verbose=False, retina_masks=True)[0]
            if r.masks is None:
                continue
            for poly, cf in zip(r.masks.xy, r.boxes.conf):
                m = probe.burn([poly], (IMGSZ, IMGSZ))
                inter = (m & tgt).sum(); union = (m | tgt).sum()
                rows.append({"extent": extent, "frac": float(m.mean()),
                             "iou": float(inter / union) if union else 0.0,
                             "bldg": float(tgt.mean()), "conf": float(cf)})
    print("done", name, flush=True)

json.dump(rows, open("runs/allmasks.json", "w"))
a = np.array([[r["extent"], r["frac"], r["iou"], r["bldg"]] for r in rows])
print("\ntotal masks:", len(rows))
print(f"{'extent':>7s} {'masks':>6s} {'med frac':>9s} {'min frac':>9s} {'p10 frac':>9s} {'frac<0.3':>9s} {'best iou':>9s}")
for e in EXTENTS:
    s = a[a[:, 0] == e]
    if not len(s): continue
    print(f"{e:7.0f} {len(s):6d} {np.median(s[:,1]):9.3f} {s[:,1].min():9.3f} "
          f"{np.percentile(s[:,1],10):9.3f} {100*np.mean(s[:,1]<0.3):8.1f}% {s[:,2].max():9.3f}")
sm = a[a[:, 1] < 0.3]
print(f"\nmasks under 30% of frame: {len(sm)} of {len(a)} ({100*len(sm)/len(a):.1f}%)")
if len(sm):
    print(f"  their IoU: median {np.median(sm[:,2]):.3f}, max {sm[:,2].max():.3f}")
    good = sm[sm[:, 2] > 0.3]
    print(f"  small AND well-placed (frac<0.3 and IoU>0.3): {len(good)}")
