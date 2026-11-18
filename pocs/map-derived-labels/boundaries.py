"""Field boundaries from Delineate Anything v2.

The Ukrainian cadastre would have been the obvious source for parcel geometry,
but public access has been restricted under martial law since 2022. This is the
better substitute anyway: cadastral lines are *legal* boundaries and can be
invisible in imagery, which makes them actively harmful as labels. This model
predicts boundaries that are actually visible, which is what a cross-correlation
peak can key on.

Delineate Anything v2 (ECCV 2026, Lavreniuk & Kussul) is a YOLO11 instance
segmentation model trained on FBIS-73M across 61 countries, Ukraine included.
It is resolution-agnostic over roughly 0.25-10 m/px, so a 6.6 cm/px orthophoto
must be *downsampled* before inference -- it is ten times finer than the model
has ever seen.

Weights are AGPL-3.0. That matters only for distribution; generating labels
offline and training our own network on them is unaffected.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from geo import Bounds

REPO = "MykolaL/DelineateAnything"
WEIGHTS = {"v2": "DelineateAnythingv2.pt", "small": "DelineateAnything-S.pt"}

# The model's trained band. Feeding it 6.6 cm/px produces nothing useful.
TARGET_MPP = 0.5


def load(variant: str = "v2"):
    """Fetch weights from the Hub (cached) and return a loaded model."""
    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO

    return YOLO(hf_hub_download(REPO, WEIGHTS[variant]))


def scale_for_model(image: Image.Image, bounds: Bounds, target_mpp: float = TARGET_MPP):
    """Downsample ``image`` to roughly ``target_mpp`` metres per pixel."""
    width = max(1, round(bounds.width_m() / target_mpp))
    height = max(1, round(bounds.height_m() / target_mpp))
    return image.resize((width, height), Image.LANCZOS)


def valid_mask(image: Image.Image, threshold: int = 30) -> np.ndarray:
    """True where the image holds real pixels rather than black padding.

    A Google Earth view exported with any rotation lands in the file as a
    rotated wedge on a black field. Half of the 2025 Gonchary orthophoto is
    padding like this, and feeding it to the model produces tile-sized
    rectangles that look like parcels and are not.
    """
    return np.array(image.convert("RGB")).sum(axis=2) > threshold


def predict(
    model,
    image: Image.Image,
    tile: int = 1024,
    overlap: int = 128,
    conf: float = 0.15,
    device: str = "cpu",
    min_valid: float = 0.9,
) -> list[np.ndarray]:
    """Run tiled inference and return field polygons in ``image`` pixel coordinates.

    Tiles overlap so that a field straddling a seam is still seen whole, but a
    polygon is only kept if its centroid lands in the tile's core -- otherwise
    every seam would produce two copies of the same field. Tiles that are not at
    least ``min_valid`` real imagery are skipped outright, and so is any polygon
    whose own area is largely padding.
    """
    width, height = image.size
    step = tile - overlap
    polygons: list[np.ndarray] = []
    valid = valid_mask(image)

    for top in range(0, max(1, height - overlap), step):
        for left in range(0, max(1, width - overlap), step):
            right, bottom = min(left + tile, width), min(top + tile, height)
            patch = np.array(image.crop((left, top, right, bottom)).convert("RGB"))
            if patch.shape[0] < 32 or patch.shape[1] < 32:
                continue
            if valid[top:bottom, left:right].mean() < min_valid:
                continue

            result = model.predict(
                patch, imgsz=tile, conf=conf, device=device, verbose=False, retina_masks=True
            )[0]
            if result.masks is None:
                continue

            core_x = (left + overlap / 2 if left > 0 else -np.inf,
                      right - overlap / 2 if right < width else np.inf)
            core_y = (top + overlap / 2 if top > 0 else -np.inf,
                      bottom - overlap / 2 if bottom < height else np.inf)

            for polygon in result.masks.xy:
                if len(polygon) < 3:
                    continue
                shifted = polygon + np.array([left, top])
                cx, cy = shifted.mean(axis=0)
                if not (core_x[0] <= cx <= core_x[1] and core_y[0] <= cy <= core_y[1]):
                    continue
                ys = np.clip(shifted[:, 1].astype(int), 0, height - 1)
                xs = np.clip(shifted[:, 0].astype(int), 0, width - 1)
                if valid[ys, xs].mean() < min_valid:
                    continue
                polygons.append(shifted)

    return polygons


def rasterise_edges(
    polygons: list[np.ndarray], shape: tuple[int, int], width_px: int = 2
) -> np.ndarray:
    """Burn polygon *outlines* into a binary mask of ``(height, width)``.

    Outlines rather than filled regions on purpose: a filled parcel over rural
    terrain is a near-uniform blob, and a blob has no correlation peak. The edge
    network is the part that carries position information.
    """
    height, width = shape[:2]
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    for polygon in polygons:
        points = [tuple(p) for p in polygon]
        draw.line(points + [points[0]], fill=1, width=width_px, joint="curve")
    return np.array(canvas)
