"""Scores for class-id maps over a set: per-class IoU and Dice with their means, plus the boundary metrics in one dict."""

from __future__ import annotations

import numpy as np

from uavloc.data.annotated import IGNORE_LABEL
from uavloc.features.segmentation import CLASSES
from uavloc.segmentation.boundary import BF_TOLERANCES_PX, BOUNDARY_IOU_DILATION_PX, boundary_scores


def iou_scores(preds: list[np.ndarray], labels: list[np.ndarray]) -> dict:
    """Per-class IoU and Dice summed over the set; mIoU and mean Dice over classes with labelled pixels in it."""
    n = len(CLASSES)
    inter, union, count = np.zeros(n), np.zeros(n), np.zeros(n)
    for p, t in zip(preds, labels):
        ok = t != IGNORE_LABEL
        for c in range(n):
            inter[c] += ((p == c) & (t == c) & ok).sum()
            union[c] += (((p == c) | (t == c)) & ok).sum()
            count[c] += ((t == c) & ok).sum()
    present = [c for c in range(n) if count[c] > 0]
    iou = {CLASSES[c]: float(inter[c] / union[c]) for c in present}
    # |P| + |T| = |P & T| + |P | T|
    dice = {CLASSES[c]: float(2 * inter[c] / (inter[c] + union[c])) for c in present}
    return {"iou": iou, "miou": float(np.mean(list(iou.values()))),
            "dice": dice, "mdice": float(np.mean(list(dice.values())))}


def segmentation_scores(preds: list[np.ndarray], labels: list[np.ndarray],
                        boundary_px: int = BOUNDARY_IOU_DILATION_PX,
                        tolerances_px: tuple[int, ...] = BF_TOLERANCES_PX) -> dict:
    """mIoU, mean Dice, Boundary IoU and BF1 at each tolerance: scalar means plus `*_per_class` dicts."""
    region = iou_scores(preds, labels)
    return {"miou": region["miou"], "iou_per_class": region["iou"],
            "mdice": region["mdice"], "dice_per_class": region["dice"],
            **boundary_scores(preds, labels, boundary_px, tolerances_px)}
