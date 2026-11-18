"""ImageNet input normalisation shared by every pretrained backbone here."""

from __future__ import annotations

import numpy as np
import torch

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
UINT8_MAX = 255.0


def normalise(rgb: np.ndarray) -> np.ndarray:
    """`(H, W, 3)` uint8 → `(H, W, 3)` float32, ImageNet-normalised."""
    return (np.asarray(rgb, np.float32) / UINT8_MAX - IMAGENET_MEAN) / IMAGENET_STD


def quantised_rgb(x: np.ndarray) -> np.ndarray:
    """A normalised `(3, H, W)` crop back to uint8 `(H, W, 3)`, i.e. an image as inference would see it."""
    rgb = x.transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(np.round(rgb * UINT8_MAX), 0, UINT8_MAX).astype(np.uint8)


def to_batch(rgb: np.ndarray, device: torch.device | str) -> torch.Tensor:
    """`(H, W, 3)` uint8 → `(1, 3, H, W)` normalised tensor on `device`."""
    return torch.from_numpy(normalise(rgb).transpose(2, 0, 1).copy())[None].to(device)
