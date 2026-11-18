"""Training-style crops for the segmentation model, as `pocs/village-seg-baseline/round2.ipynb` drew them.

Moved here on second use: BatchNorm recalibration only reproduces round 2's
scores when its statistics come from the same crop distribution.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from uavloc.features.imagenet import IMAGENET_MEAN, IMAGENET_STD, UINT8_MAX

BRIGHTNESS_JITTER = 0.15
TINT_JITTER = 0.05
CONTRAST_JITTER = 0.15
GSD_JITTER = (0.8, 1.25)
RARE_CLASSES = (2, 3)  # track, woody
QUARTER_TURNS = 4
FLIP_P = 0.5


def jittered_input(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """`(H, W, 3)` uint8 → `(3, H, W)` float32: random rot90/flip, brightness, tint, contrast, ImageNet norm."""
    x = np.rot90(np.asarray(rgb, np.float32) / UINT8_MAX, rng.integers(QUARTER_TURNS))
    if rng.random() < FLIP_P:
        x = x[:, ::-1]
    x = x * (1 + rng.uniform(-BRIGHTNESS_JITTER, BRIGHTNESS_JITTER) + rng.uniform(-TINT_JITTER, TINT_JITTER, 3))
    mean = x.mean()
    x = (x - mean) * (1 + rng.uniform(-CONTRAST_JITTER, CONTRAST_JITTER)) + mean
    return ((x - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1).astype(np.float32)


def satellite_crop(image: np.ndarray, crop_px: int, rng: np.random.Generator) -> np.ndarray:
    h, w = image.shape[:2]
    top, left = rng.integers(h - crop_px + 1), rng.integers(w - crop_px + 1)
    return image[top:top + crop_px, left:left + crop_px]


def drone_crop(image: np.ndarray, label: np.ndarray, mpp: float, rng: np.random.Generator,
               gsd_m: float, crop_px: int, rare_class_p: float) -> np.ndarray:
    """A `crop_px` square at ~`gsd_m` (jittered); a share centred on track/woody; zero-padded where the frame is short."""
    n = int(round(crop_px * gsd_m * rng.uniform(*GSD_JITTER) / mpp))
    h, w = label.shape
    rare = [c for c in RARE_CLASSES if (label == c).any()]
    if rng.random() < rare_class_p and rare:
        ys, xs = np.nonzero(label == rng.choice(rare))
        k = rng.integers(len(ys))
        cy, cx = ys[k], xs[k]
    else:
        cy, cx = rng.integers(h), rng.integers(w)
    top = int(np.clip(cy - n // 2, 0, h - n)) if n <= h else (h - n) // 2
    left = int(np.clip(cx - n // 2, 0, w - n)) if n <= w else (w - n) // 2
    out = np.zeros((n, n, 3), np.uint8)
    rows, cols = slice(max(top, 0), min(top + n, h)), slice(max(left, 0), min(left + n, w))
    out[rows.start - top:rows.stop - top, cols.start - left:cols.stop - left] = image[rows, cols]
    return np.asarray(Image.fromarray(out).resize((crop_px, crop_px), Image.BILINEAR))
