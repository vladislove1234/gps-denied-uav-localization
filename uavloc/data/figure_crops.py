"""Choosing crop windows of labelled rasters for thesis figures, by what the label contains."""

from __future__ import annotations

import cv2
import numpy as np
import torch

from uavloc.segmentation.losses import class_skeleton

IGNORE_LABEL = 255


def window_shares(label: np.ndarray, size_px: tuple[int, int], stride_px: int, n_classes: int) -> tuple[np.ndarray, np.ndarray]:
    """`(corners, shares)`: `(N, 2)` top-left `(row, col)` of every window with no unlabelled pixel, and `(N, C)` class shares."""
    height, width = size_px
    corners, shares = [], []
    for top in range(0, label.shape[0] - height + 1, stride_px):
        for left in range(0, label.shape[1] - width + 1, stride_px):
            window = label[top:top + height, left:left + width]
            if (window == IGNORE_LABEL).any():
                continue
            corners.append((top, left))
            shares.append(np.bincount(window.ravel(), minlength=n_classes)[:n_classes] / window.size)
    return np.array(corners), np.array(shares)


def class_diversity(shares: np.ndarray, min_share: float) -> np.ndarray:
    """Entropy of the class shares, counted only for classes covering at least `min_share` (others add nothing)."""
    kept = np.where(shares >= min_share, shares, 0.0)
    kept = kept / np.maximum(kept.sum(axis=1, keepdims=True), 1e-9)
    return -(np.where(kept > 0, kept * np.log(np.maximum(kept, 1e-12)), 0.0)).sum(axis=1)


def most_diverse_window(label: np.ndarray, size_px: tuple[int, int], stride_px: int, n_classes: int,
                        min_share: float = 0.03) -> tuple[int, int]:
    """Top-left `(row, col)` of the window with the most even mix of classes."""
    corners, shares = window_shares(label, size_px, stride_px, n_classes)
    return tuple(int(v) for v in corners[class_diversity(shares, min_share).argmax()])


def densest_window(label: np.ndarray, class_id: int, size_px: tuple[int, int], stride_px: int,
                   n_classes: int, max_share: float = 0.35) -> tuple[int, int]:
    """Top-left `(row, col)` of the window where `class_id` covers most of it without exceeding `max_share`."""
    corners, shares = window_shares(label, size_px, stride_px, n_classes)
    score = np.where(shares[:, class_id] <= max_share, shares[:, class_id], -1.0)
    return tuple(int(v) for v in corners[score.argmax()])


def display_skeleton(label: np.ndarray, class_id: int, min_px: int, thickness_px: int) -> np.ndarray:
    """`(H, W)` bool: the centreline of `class_id` (pieces under `min_px` left out), thickened to `thickness_px` to be visible."""
    skeleton = class_skeleton(torch.from_numpy(label[None].astype(np.int64)), class_id, 0, min_px)[0].numpy()
    kernel = np.ones((thickness_px, thickness_px), np.uint8)
    return cv2.dilate(skeleton.astype(np.uint8), kernel).astype(bool)
