"""TensorBoard segmentation previews for any training: the same fixed crops, drawn at every log point.

A segmenter is any `nn.Module` mapping normalised `(N, 3, H, W)` to per-pixel class scores `(N, C, H, W)`.
The model sees the whole crop; the grid shows its centre `PREVIEW_PX` window at full resolution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from uavloc.features.imagenet import quantised_rgb
from uavloc.plotting.segmentation import centre_window, prediction_grid
from uavloc.track import log_images

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

PREVIEW_CROPS = 4                        # rows of the grid, per split
PREVIEW_PX = 256                         # shown window, centred in the crop
PREVIEW_TAG = "segmentation"             # tags are "<split>/segmentation": identical across runs, so they compare

Crops = tuple[torch.Tensor, torch.Tensor]    # normalised (N, 3, H, W) float, labels (N, H, W) int64


@torch.no_grad()
def segmenter_grid(model: nn.Module, crops: Crops, device: torch.device | str, window_px: int = PREVIEW_PX) -> np.ndarray:
    """image | label | prediction | errors of `crops`; the model's train/eval mode is restored afterwards."""
    x, labels = crops
    training = model.training
    pred = model.eval()(x.to(device)).argmax(1).cpu().numpy()
    model.train(training)
    rgb = np.stack([quantised_rgb(c) for c in x.numpy()])
    return prediction_grid(*(centre_window(a, window_px) for a in (rgb, labels.numpy(), pred)))


def log_segmenter_previews(writer: SummaryWriter | None, model: nn.Module, previews: dict[str, Crops], step: int,
                           device: torch.device | str, tag: str = PREVIEW_TAG) -> None:
    """One grid per split under `<split>/<tag>`; TensorBoard's step slider scrubs through them."""
    if writer is None:
        return
    log_images(writer, {f"{split}/{tag}": segmenter_grid(model, crops, device) for split, crops in previews.items()},
               step)
