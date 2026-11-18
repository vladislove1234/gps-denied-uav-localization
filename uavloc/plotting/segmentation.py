"""Segmentation result grids as plain RGB arrays, for TensorBoard's image slider and quick `imshow`s.

One row per crop, one column per view (image, label, predictions, error maps), a title strip on top and a class
legend at the bottom. Pure numpy + OpenCV text, no matplotlib, so a training loop can build one every log point.
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from uavloc.data.annotated import IGNORE_LABEL
from uavloc.features.segmentation import CLASSES
from uavloc.plotting.correlation import SEG_PALETTE

IGNORE_COLOUR = (255, 255, 255)
CORRECT_COLOUR = (235, 235, 235)
WRONG_COLOUR = (220, 40, 40)
BOTH_WRONG_COLOUR = (190, 190, 190)
FIXED_COLOUR = (40, 170, 60)
BROKEN_COLOUR = (220, 40, 40)
BACKGROUND_COLOUR = (255, 255, 255)
TEXT_COLOUR = (0, 0, 0)
GAP_PX = 4
STRIP_PX = 22
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.45
TEXT_BASELINE_PX = 15
LEGEND_SWATCH_PX = 12


def colour_labels(label: np.ndarray) -> np.ndarray:
    """`(H, W)` class ids → `(H, W, 3)` uint8 in `SEG_PALETTE`; ignored pixels white."""
    out = np.full((*label.shape, 3), IGNORE_COLOUR, np.uint8)
    ok = label != IGNORE_LABEL
    out[ok] = SEG_PALETTE[label[ok]]
    return out


def error_map(label: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Red where `pred` is wrong, light grey where right, white where unlabelled."""
    ok = label != IGNORE_LABEL
    out = np.full((*label.shape, 3), IGNORE_COLOUR, np.uint8)
    out[ok] = CORRECT_COLOUR
    out[ok & (pred != label)] = WRONG_COLOUR
    return out


def fixed_broken(label: np.ndarray, before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Green where `after` fixed a pixel `before` got wrong, red where it broke one, grey where both are wrong."""
    ok = label != IGNORE_LABEL
    out = np.full((*label.shape, 3), IGNORE_COLOUR, np.uint8)
    out[ok & (before != label) & (after != label)] = BOTH_WRONG_COLOUR
    out[ok & (before != label) & (after == label)] = FIXED_COLOUR
    out[ok & (before == label) & (after != label)] = BROKEN_COLOUR
    return out


def _strip(width: int) -> np.ndarray:
    return np.full((STRIP_PX, width, 3), BACKGROUND_COLOUR, np.uint8)


def _legend(width: int) -> np.ndarray:
    strip, x = _strip(width), GAP_PX
    for name, colour in zip(CLASSES, SEG_PALETTE):
        top = (STRIP_PX - LEGEND_SWATCH_PX) // 2
        strip[top:top + LEGEND_SWATCH_PX, x:x + LEGEND_SWATCH_PX] = colour
        x += LEGEND_SWATCH_PX + GAP_PX
        cv2.putText(strip, name, (x, TEXT_BASELINE_PX), FONT, FONT_SCALE, TEXT_COLOUR, 1, cv2.LINE_AA)
        x += cv2.getTextSize(name, FONT, FONT_SCALE, 1)[0][0] + 3 * GAP_PX
    return strip


def segmentation_grid(rows: Sequence[Sequence[np.ndarray]], titles: Sequence[str]) -> np.ndarray:
    """`rows[i][j]` `(H, W, 3)` uint8 panels, all the same size → one `(H', W', 3)` uint8 grid with titles and legend."""
    h, w = rows[0][0].shape[:2]
    width = len(titles) * (w + GAP_PX) - GAP_PX
    header = _strip(width)
    for j, title in enumerate(titles):
        cv2.putText(header, title, (j * (w + GAP_PX) + GAP_PX, TEXT_BASELINE_PX), FONT, FONT_SCALE, TEXT_COLOUR, 1,
                    cv2.LINE_AA)
    parts = [header]
    for row in rows:
        line = np.full((h, width, 3), BACKGROUND_COLOUR, np.uint8)
        for j, panel in enumerate(row):
            line[:, j * (w + GAP_PX):j * (w + GAP_PX) + w] = panel
        parts += [line, np.full((GAP_PX, width, 3), BACKGROUND_COLOUR, np.uint8)]
    return np.concatenate([*parts, _legend(width)])


def centre_window(batch: np.ndarray, size_px: int) -> np.ndarray:
    """The centre `size_px` square of `(N, H, W, ...)` crops, so a model can see context the grid does not show."""
    h, w = batch.shape[1:3]
    top, left = max((h - size_px) // 2, 0), max((w - size_px) // 2, 0)
    return batch[:, top:top + size_px, left:left + size_px]


def prediction_grid(rgb: np.ndarray, labels: np.ndarray, preds: np.ndarray) -> np.ndarray:
    """image | label | prediction | errors, one row per crop. Inputs `(N, H, W, 3)` uint8 and `(N, H, W)` ids."""
    rows = [[r, colour_labels(y), colour_labels(p), error_map(y, p)] for r, y, p in zip(rgb, labels, preds)]
    return segmentation_grid(rows, ["image", "label", "prediction", "errors (red)"])


def refinement_grid(rgb: np.ndarray, labels: np.ndarray, before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """image | label | before | after | fixed (green) / broken (red), one row per crop."""
    rows = [[r, colour_labels(y), colour_labels(b), colour_labels(a), fixed_broken(y, b, a)]
            for r, y, b, a in zip(rgb, labels, before, after)]
    return segmentation_grid(rows, ["image", "label", "input", "refined", "fixed green / broken red"])
