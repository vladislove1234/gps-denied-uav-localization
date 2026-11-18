"""Tiled segmentation inference shared by the model-based adapters."""

from __future__ import annotations

from uavloc.features.tiling import tiled_logits  # noqa: F401  lives in uavloc; re-exported for the adapters


def device():
    import torch
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
