"""Probe 4 -- can an open-vocabulary model label our classes from a text prompt?

The idea under test is the user's: skip training entirely and just *name* the
classes. `ultralytics` ships YOLOE, which is YOLO with a CLIP text encoder --
`set_classes(names, get_text_pe(names))` and it returns masks for concepts it
was never explicitly trained to segment.

Why this was worth running even though the prior is bad. Probe 1 needed a
labelled yardstick and had one only for `field`; this needs no labels and no
training at all, so the cost of an answer is an afternoon. And we have a free
control: `building` and `water` come from OSM and are measured good, so if the
model cannot find buildings that OSM proves are there, it cannot be trusted on
`track` and `woody` either. That makes the probe self-falsifying, which is
exactly what probe 1 lacked.

Three things are measured, not one, because probe 1's lesson was that a single
aggregate number hides the failure:

* **instance recall** -- of N OSM building blobs, how many does a predicted mask
  actually touch. This is the honest question; pixel coverage is not.
* **pixel IoU** per class.
* **confidence distribution** -- a detector that fires at 0.2 on everything is
  not detecting, it is guessing.

Both prompting modes are covered: text, and visual (an exemplar region cut from
a known-good OSM building, which sidesteps the text encoder's vocabulary).

Run from `src/` with the uavloc env.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "map-derived-labels"))
from tiles import TileGrid  # noqa: E402

warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = None

ROOT = Path("../data/derived/ua-village-multidate")
OUT = Path(__file__).resolve().parent

# Our class set minus background, in the order YOLOE will index them.
CLASSES = ["building", "track", "woody", "water"]

# `label.png` ids from osm.CLASSES. `road` is our `track`, `wood` our `woody`.
OSM_ID = {"building": 1, "track": 2, "water": 3, "woody": 4}

# Phrasings. The sweep showed bare nouns return nothing at all and the
# "in satellite image" suffix is what makes the text encoder fire, so the
# comparison is kept in the record rather than silently dropping the loser.
PROMPTS = {
    "plain": ["building", "dirt road", "trees", "river"],
    "aerial": [
        "aerial view of a building",
        "dirt road seen from above",
        "group of trees from above",
        "river seen from above",
    ],
    "satellite": [
        "building rooftop in satellite image",
        "unpaved track in satellite image",
        "forest in satellite image",
        "water body in satellite image",
    ],
}


def locations() -> list[str]:
    return json.loads((ROOT / "index.json").read_text())["built"]


def load_location(name: str):
    meta = json.loads((ROOT / name / "meta.json").read_text())
    image = Image.open(ROOT / name / "current.jpg").convert("RGB")
    label = np.array(Image.open(ROOT / name / "label.png"))
    return image, label, meta


def control_masks(label: np.ndarray, meta: dict) -> dict[str, np.ndarray]:
    """OSM class masks, shifted by the per-location offset the build measured.

    The dataset card records labels sitting a median 2.84 m off the imagery
    because OSM was traced against a different basemap. Scoring a detector
    against an unshifted control would charge that error to the detector.
    """
    offset = meta.get("label_offset") or {}
    dx, dy = int(offset.get("dx_px", 0)), int(offset.get("dy_px", 0))
    shifted = np.roll(np.roll(label, -dy, axis=0), -dx, axis=1)
    return {name: (shifted == value) for name, value in OSM_ID.items()}


def predict_multiclass(model, image, tile, overlap, conf, device, predictor=None, prompts=None,
                       refer=None):
    """Tiled inference that keeps class ids. Mirrors `boundaries.predict`'s seam
    handling -- a polygon counts only if its centroid is in the tile core, so a
    feature crossing a seam is not counted twice."""
    width, height = image.size
    step = tile - overlap
    found: list[tuple[int, np.ndarray, float]] = []

    for top in range(0, max(1, height - overlap), step):
        for left in range(0, max(1, width - overlap), step):
            right, bottom = min(left + tile, width), min(top + tile, height)
            patch = np.array(image.crop((left, top, right, bottom)))
            if patch.shape[0] < 32 or patch.shape[1] < 32:
                continue

            kwargs = dict(imgsz=tile, conf=conf, device=device, verbose=False,
                          retina_masks=True)
            if predictor is not None:
                # The kwarg is `visual_prompts`; passing `prompts` here is
                # silently swallowed into overrides and VP never activates,
                # which returns zero detections on everything (bug-018).
                kwargs.update(predictor=predictor, visual_prompts=prompts, refer_image=refer)
            result = model.predict(patch, **kwargs)[0]
            if result.masks is None or result.boxes is None:
                continue

            core_x = (left + overlap / 2 if left > 0 else -np.inf,
                      right - overlap / 2 if right < width else np.inf)
            core_y = (top + overlap / 2 if top > 0 else -np.inf,
                      bottom - overlap / 2 if bottom < height else np.inf)

            for polygon, cls, cf in zip(result.masks.xy, result.boxes.cls, result.boxes.conf):
                if len(polygon) < 3:
                    continue
                shifted = polygon + np.array([left, top])
                cx, cy = shifted.mean(axis=0)
                if core_x[0] <= cx <= core_x[1] and core_y[0] <= cy <= core_y[1]:
                    found.append((int(cls), shifted, float(cf)))
    return found


def burn(polygons, shape) -> np.ndarray:
    from PIL import ImageDraw
    canvas = Image.new("L", (shape[1], shape[0]), 0)
    draw = ImageDraw.Draw(canvas)
    for polygon in polygons:
        points = [tuple(map(float, p)) for p in polygon]
        if len(points) >= 3:
            draw.polygon(points, fill=1)
    return np.array(canvas, dtype=bool)


def instance_recall(control: np.ndarray, predicted: np.ndarray, min_px: int = 20):
    """Of the connected blobs OSM marks, how many does `predicted` touch at all.

    Deliberately generous -- *any* overlap counts as a hit. A detector that
    cannot clear this bar has not found the object in any sense.
    """
    from scipy import ndimage
    labelled, count = ndimage.label(control)
    if count == 0:
        return None, 0
    sizes = ndimage.sum(control, labelled, range(1, count + 1))
    keep = [i + 1 for i, s in enumerate(sizes) if s >= min_px]
    if not keep:
        return None, 0
    hits = sum(1 for i in keep if predicted[labelled == i].any())
    return hits / len(keep), len(keep)


def null_recall(control: np.ndarray, predicted: np.ndarray, trials: int = 20,
                seed: int = 0) -> float | None:
    """Instance recall after randomly translating the prediction.

    This is the guard probe 1 did not have. A circular shift preserves the
    number of blobs, their shapes and the total coverage exactly, and destroys
    only *where* they are. If real recall does not beat this, the detector has
    contributed no positional information and the score is an artifact of how
    much of the tile it painted.
    """
    rng = np.random.default_rng(seed)
    height, width = predicted.shape
    scores = []
    for _ in range(trials):
        dy, dx = rng.integers(0, height), rng.integers(0, width)
        rolled = np.roll(np.roll(predicted, dy, axis=0), dx, axis=1)
        value, _ = instance_recall(control, rolled)
        if value is not None:
            scores.append(value)
    return float(np.mean(scores)) if scores else None


def score(found, control: dict, shape) -> dict:
    out = {}
    for index, name in enumerate(CLASSES):
        polygons = [p for c, p, _ in found if c == index]
        confs = [f for c, _, f in found if c == index]
        predicted = burn(polygons, shape)
        truth = control[name]
        recall, total = instance_recall(truth, predicted)
        union = (predicted | truth).sum()
        out[name] = {
            "detections": len(polygons),
            "pred_cover": float(predicted.mean()),
            "osm_cover": float(truth.mean()),
            "osm_instances": total,
            "instance_recall": recall,
            "null_recall": null_recall(truth, predicted) if recall is not None else None,
            "iou": float((predicted & truth).sum() / union) if union else None,
            "max_conf": round(max(confs), 3) if confs else 0.0,
            "median_conf": round(float(np.median(confs)), 3) if confs else 0.0,
        }
    return out


def exemplar_box(control: np.ndarray, shape) -> np.ndarray | None:
    """Largest OSM building blob, as a bbox, to serve as a visual prompt."""
    from scipy import ndimage
    labelled, count = ndimage.label(control)
    if count == 0:
        return None
    sizes = ndimage.sum(control, labelled, range(1, count + 1))
    biggest = int(np.argmax(sizes)) + 1
    ys, xs = np.where(labelled == biggest)
    return np.array([[xs.min(), ys.min(), xs.max(), ys.max()]], dtype=np.float32)


def run(names, variant, prompt_set, conf, tile, device, mode):
    from ultralytics import YOLOE

    model = YOLOE(variant)
    results = {}
    prompt_names = PROMPTS[prompt_set]
    if mode == "text":
        model.set_classes(prompt_names, model.get_text_pe(prompt_names))

    for index, name in enumerate(names, 1):
        started = time.time()
        image, label, meta = load_location(name)
        shape = tuple(meta["shape"])
        control = control_masks(label, meta)

        if mode == "text":
            found = predict_multiclass(model, image, tile, 128, conf, device)
        else:
            from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
            box = exemplar_box(control["building"], shape)
            if box is None:
                continue
            # Crop a reference patch around the exemplar so the prompt box is
            # expressed in that patch's own coordinates.
            x0, y0, x1, y1 = box[0]
            pad = 256
            ref_box = (max(0, int(x0) - pad), max(0, int(y0) - pad),
                       min(shape[1], int(x1) + pad), min(shape[0], int(y1) + pad))
            refer = image.crop(ref_box)
            local = np.array([[x0 - ref_box[0], y0 - ref_box[1],
                               x1 - ref_box[0], y1 - ref_box[1]]], dtype=np.float32)
            prompts = {"bboxes": local, "cls": np.array([0])}
            found = predict_multiclass(
                model, image, tile, 128, conf, device,
                predictor=YOLOEVPSegPredictor, prompts=prompts, refer=np.array(refer),
            )

        stats = score(found, control, shape)
        results[name] = {
            "seconds": round(time.time() - started, 1),
            "detections": len(found),
            "by_class": dict(Counter(CLASSES[c] for c, _, _ in found)),
            "classes": stats,
        }
        b = stats["building"]
        print(
            f"[{index}/{len(names)}] {name}: {len(found)} dets, "
            f"building recall {b['instance_recall']:.2f} vs null {b['null_recall']:.2f} "
            f"of {b['osm_instances']}, cover {b['pred_cover']*100:.0f}%, "
            f"IoU {b['iou']:.3f}, {results[name]['seconds']}s",
            flush=True,
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=9)
    parser.add_argument("--names", nargs="*", default=None)
    parser.add_argument("--variant", default="yoloe-11l-seg.pt")
    parser.add_argument("--prompts", default="satellite", choices=list(PROMPTS))
    parser.add_argument("--mode", default="text", choices=["text", "visual"])
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--tile", type=int, default=1024)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    probe1 = json.loads((OUT / "results-field.json").read_text())
    names = args.names or list(probe1)[: args.n]

    results = run(names, args.variant, args.prompts, args.conf, args.tile,
                  args.device, args.mode)
    out = args.out or str(OUT / f"results-yoloe-{args.mode}-{args.prompts}.json")
    Path(out).write_text(json.dumps(
        {"config": vars(args), "locations": results}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
