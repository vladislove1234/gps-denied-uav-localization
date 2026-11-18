"""Connectivity scores for thin classes (tracks): centreline recall and precision, clDice, and fragments per km.

Area metrics (IoU, Dice) barely notice a road broken into dashes. These count the label's centreline the prediction
covers, the prediction's centreline that lies on the label, and how many separate pieces the prediction breaks the
labelled tracks into, per km of labelled centreline (Shit et al. 2021 for clDice; Kirchhoff et al. 2024 for
skeleton recall). Accumulated over a set of `(pred, label)` pairs, like `iou_scores`. Pieces under
`MIN_FRAGMENT_PX`, in the label and in the prediction, are dropped first: they are slivers, not tracks.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

from uavloc.data.annotated import IGNORE_LABEL
from uavloc.features.segmentation import CLASSES
from uavloc.segmentation.instances import M_PER_PX, components

TRACK = CLASSES.index("track")
SKELETON_TOLERANCE_PX = 2      # a centreline pixel counts as covered within this chessboard distance (0.77 m)
MIN_FRAGMENT_PX = 16           # as MIN_BUILDING_PX: smaller predicted pieces are noise, not fragments
M_PER_KM = 1000.0


def near(mask: np.ndarray, tolerance_px: int) -> np.ndarray:
    """`mask` dilated by `tolerance_px` (chessboard)."""
    if not tolerance_px:
        return mask
    return ndi.binary_dilation(mask, np.ones((2 * tolerance_px + 1,) * 2, bool))


def centreline_counts(pred: np.ndarray, label: np.ndarray, cls: int, tolerance_px: int,
                      min_px: int) -> np.ndarray:
    """`[covered label centreline px, label centreline px, pred centreline px on label, pred centreline px,
    pred fragments touching the label, label pieces]` for one image."""
    valid = label != IGNORE_LABEL
    truth_ids, n_truth = components((label == cls) & valid, min_px)
    fragments, n = components((pred == cls) & valid, min_px)
    truth, predicted = truth_ids > 0, fragments > 0
    truth_line, pred_line = skeletonize(truth), skeletonize(predicted)
    touching = len(np.setdiff1d(np.unique(fragments[truth]), [0])) if n else 0
    return np.array([(truth_line & near(predicted, tolerance_px)).sum(), truth_line.sum(),
                     (pred_line & near(truth, tolerance_px)).sum(), pred_line.sum(),
                     touching, n_truth], dtype=np.float64)


def track_scores(preds: list[np.ndarray], labels: list[np.ndarray], cls: int = TRACK,
                 tolerance_px: int = SKELETON_TOLERANCE_PX, min_px: int = MIN_FRAGMENT_PX,
                 m_per_px: float = M_PER_PX) -> dict[str, float]:
    """`track_skeleton_recall`, `track_skeleton_precision`, `track_cldice`, `track_fragments_per_km` (predicted
    pieces touching a labelled track) and `track_label_pieces_per_km` (the label's own count, the floor)."""
    covered, truth_px, on_label, pred_px, fragments, pieces = sum(
        centreline_counts(p, t, cls, tolerance_px, min_px) for p, t in zip(preds, labels))
    recall = covered / truth_px if truth_px else float("nan")
    precision = on_label / pred_px if pred_px else float("nan")
    cldice = 2 * recall * precision / (recall + precision) if recall + precision > 0 else float("nan")
    km = truth_px * m_per_px / M_PER_KM
    return {"track_skeleton_recall": float(recall), "track_skeleton_precision": float(precision),
            "track_cldice": float(cldice),
            "track_fragments_per_km": float(fragments / km) if km else float("nan"),
            "track_label_pieces_per_km": float(pieces / km) if km else float("nan")}
