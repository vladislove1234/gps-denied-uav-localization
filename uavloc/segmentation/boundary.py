"""Boundary metrics for class-id maps: Boundary IoU (Cheng et al. 2021) and boundary F1 at a pixel tolerance.

Both are accumulated over a set of `(pred, label)` pairs per class, like `iou_scores`, and averaged over the
classes that have labelled pixels in the set. Pixels labelled `IGNORE_LABEL` are never counted, and a
boundary against ignored pixels is not a boundary, in the prediction or the label (ignored pixels count as
"same class" when eroding).
The image edge is not a boundary either — unlike the reference implementation, which zero-pads — because
the val images are crops of larger scenes (e.g. the 640-row birky strip).
"""

from __future__ import annotations

import cv2
import numpy as np

from uavloc.data.annotated import IGNORE_LABEL
from uavloc.features.segmentation import CLASSES

# 4 px = 1.54 m at 0.385 m/px: about the labels' positional error (OSM/FLAIR seeds aligned per date to
# ~1.5 m), half an LR-ASPP stride-8 cell, and under half the width of a village track. See the build notes.
BOUNDARY_IOU_DILATION_PX = 4
BF_TOLERANCES_PX = (1, 2)
_ERODE_KERNEL = np.ones((3, 3), np.uint8)


def inner_band(mask: np.ndarray, width_px: int) -> np.ndarray:
    """Pixels of boolean `mask` within `width_px` (chessboard) of its contour; the image edge is no contour."""
    eroded = cv2.erode(mask.astype(np.uint8), _ERODE_KERNEL, iterations=width_px,
                       borderType=cv2.BORDER_REPLICATE)
    return mask & ~eroded.astype(bool)


def class_band(ids: np.ndarray, cls: int, width_px: int, ignored: np.ndarray) -> np.ndarray:
    """`inner_band` of class `cls` in a class-id map, where `ignored` pixels make no contour and are excluded."""
    return inner_band((ids == cls) | ignored, width_px) & (ids == cls) & ~ignored


def label_band(label: np.ndarray, cls: int, width_px: int) -> np.ndarray:
    return class_band(label, cls, width_px, label == IGNORE_LABEL)


def disk(radius_px: int) -> np.ndarray:
    """Structuring element of every pixel within Euclidean `radius_px` of the centre."""
    yy, xx = np.mgrid[-radius_px:radius_px + 1, -radius_px:radius_px + 1]
    return (yy ** 2 + xx ** 2 <= radius_px ** 2).astype(np.uint8)


def within(points: np.ndarray, target: np.ndarray, tolerance_px: int) -> np.ndarray:
    """Pixels of `points` that lie within Euclidean `tolerance_px` of any pixel of `target`."""
    return points & cv2.dilate(target.astype(np.uint8), disk(tolerance_px)).astype(bool)


def boundary_counts(pred: np.ndarray, label: np.ndarray, boundary_px: int,
                    tolerances_px: tuple[int, ...]) -> np.ndarray:
    """Per class `(C, 2 + 4T)`: Boundary-IoU intersection, union, then per tolerance
    matched-pred, pred, matched-label, label boundary pixel counts."""
    ignored = label == IGNORE_LABEL
    rows = []
    for c in range(len(CLASSES)):
        p_band, g_band = class_band(pred, c, boundary_px, ignored), class_band(label, c, boundary_px, ignored)
        row = [(p_band & g_band).sum(), (p_band | g_band).sum()]
        p_edge, g_edge = class_band(pred, c, 1, ignored), class_band(label, c, 1, ignored)
        for t in tolerances_px:
            row += [within(p_edge, g_edge, t).sum(), p_edge.sum(), within(g_edge, p_edge, t).sum(), g_edge.sum()]
        rows.append(row)
    return np.array(rows, np.int64)


def f1(matched_pred: float, n_pred: float, matched_label: float, n_label: float) -> float:
    """Boundary F1 from match counts; 0 when either side has boundary and nothing matches."""
    precision = matched_pred / n_pred if n_pred else 0.0
    recall = matched_label / n_label if n_label else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def boundary_scores(preds: list[np.ndarray], labels: list[np.ndarray], boundary_px: int = BOUNDARY_IOU_DILATION_PX,
                    tolerances_px: tuple[int, ...] = BF_TOLERANCES_PX) -> dict:
    """Boundary IoU and BF1 per class and as a mean over classes with labelled pixels and a label boundary."""
    counts = sum(boundary_counts(p, t, boundary_px, tolerances_px) for p, t in zip(preds, labels))
    present = [c for c in range(len(CLASSES)) if any(((t == c).any() for t in labels)) and counts[c, 1] > 0]
    out = {"biou_per_class": {CLASSES[c]: float(counts[c, 0] / counts[c, 1]) for c in present}}
    for k, t in enumerate(tolerances_px):
        cols = counts[:, 2 + 4 * k:6 + 4 * k]
        out[f"bf1_{t}px_per_class"] = {CLASSES[c]: f1(*cols[c]) for c in present if cols[c, 3] > 0}
    for key in list(out):
        out[key.removesuffix("_per_class")] = float(np.mean(list(out[key].values())))
    return out
