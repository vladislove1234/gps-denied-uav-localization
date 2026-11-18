"""Tiled segmentation inference: overlapping windows, logits averaged where they overlap."""

from __future__ import annotations

from typing import Callable

import numpy as np

TILE_PX = 512
TILE_STEP_PX = 384

# `(B, 3, P, P)` float32 normalised tiles → `(B, C, P / out_stride, P / out_stride)` logits
BatchPredictor = Callable[[np.ndarray], np.ndarray]


def tiled_predict(predict: BatchPredictor, x: np.ndarray, patch: int, step: int, n_out: int,
                  batch: int = 8, out_stride: int = 1) -> np.ndarray:
    """Average logits of overlapping ``patch`` windows over an ``(H, W, 3)``
    normalised float array. Edge windows are clamped inside the image (the PoC
    scheme), and padded only when the image is smaller than one patch.

    ``out_stride`` > 1 is for a ``predict`` whose output is ``out_stride`` times
    coarser than its input (e.g. LR-ASPP's stride-8 logits): the result is then
    ``(n_out, H / out_stride, W / out_stride)``, and H, W, ``patch`` and ``step``
    must be multiples of ``out_stride`` so every window lands on whole cells."""
    h, w = x.shape[:2]
    if any(v % out_stride for v in (h, w, patch, step)):
        raise ValueError(f"H={h}, W={w}, patch={patch}, step={step} must be multiples of out_stride={out_stride}")
    tops = sorted({min(t, max(h - patch, 0)) for t in range(0, max(h - (patch - step), 1), step)})
    lefts = sorted({min(l, max(w - patch, 0)) for l in range(0, max(w - (patch - step), 1), step)})
    s = out_stride
    logits = np.zeros((n_out, h // s, w // s), np.float32)
    count = np.zeros((h // s, w // s), np.float32)
    windows = [(t, l) for t in tops for l in lefts]

    for i in range(0, len(windows), batch):
        chunk = windows[i:i + batch]
        tiles = []
        for t, l in chunk:
            tile = x[t:t + patch, l:l + patch]
            ph, pw = tile.shape[:2]
            if ph < patch or pw < patch:
                tile = np.pad(tile, ((0, patch - ph), (0, patch - pw), (0, 0)))
            tiles.append(tile.transpose(2, 0, 1))
        out = predict(np.stack(tiles).astype(np.float32))
        for (t, l), o in zip(chunk, out):
            ph, pw = min(patch, h - t) // s, min(patch, w - l) // s
            logits[:, t // s:t // s + ph, l // s:l // s + pw] += o[:, :ph, :pw]
            count[t // s:t // s + ph, l // s:l // s + pw] += 1
    return logits / np.maximum(count, 1)


def tiled_logits(model, x: np.ndarray, patch: int, step: int, n_out: int,
                 dev, batch: int = 8, out_stride: int = 1) -> np.ndarray:
    """`tiled_predict` for a torch ``model`` on device ``dev``."""
    import torch

    def predict(tiles: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return model(torch.from_numpy(tiles).float().to(dev)).float().cpu().numpy()

    return tiled_predict(predict, x, patch, step, n_out, batch, out_stride)
