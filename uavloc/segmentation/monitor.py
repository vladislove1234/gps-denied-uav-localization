"""Train and val metrics on fixed crop sets during training: every loss term plus mIoU, in eval mode.

The same crops are scored at every log point, so the `train_*` and `val_*` curves compare pixel for pixel across
steps and runs. Keys follow `track.log_history`: `train_ce` becomes the TensorBoard tag `train/ce`.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from torch import nn

from uavloc.data.annotated import IGNORE_LABEL, AnnotatedImage
from uavloc.features.imagenet import normalise
from uavloc.segmentation.instances import status_shares
from uavloc.segmentation.metrics import segmentation_scores
from uavloc.segmentation.topology import track_scores
from uavloc.segmentation.preview import Crops

LossTerms = Callable[[torch.Tensor, torch.Tensor], dict[str, torch.Tensor]]    # (logits, labels) → {name: scalar}


def tiled_crops(images: list[AnnotatedImage], crop_px: int) -> Crops:
    """Every non-overlapping `crop_px` window with a labelled pixel, over all `images`: `(N, 3, P, P)`, `(N, P, P)`."""
    xs, ys = [], []
    for image in images:
        h, w = image.label.shape
        for top in range(0, h - crop_px + 1, crop_px):
            for left in range(0, w - crop_px + 1, crop_px):
                label = image.label[top:top + crop_px, left:left + crop_px]
                if (label != IGNORE_LABEL).any():
                    xs.append(normalise(image.rgb[top:top + crop_px, left:left + crop_px]).transpose(2, 0, 1))
                    ys.append(label)
    return torch.from_numpy(np.stack(xs)).float(), torch.from_numpy(np.stack(ys).astype(np.int64))


@torch.no_grad()
def crop_metrics(model: nn.Module, crops: Crops, loss_terms: LossTerms, batch: int,
                 device: torch.device | str, prefix: str) -> dict[str, float]:
    """`<prefix>_<term>` for each loss term (crop-weighted mean), every `segmentation_scores` scalar (`miou`, `mdice`,
    `biou`, `bf1_*`), per-class `iou_<class>`, `building_iou`, the building separate / merged / missed shares and the
    track centreline scores (`topology.track_scores`).

    Scored in eval mode; the model's train/eval mode is restored afterwards.
    """
    x, y = crops
    training = model.training
    model.eval()
    sums: dict[str, float] = {}
    preds = []
    for start in range(0, len(x), batch):
        xb, yb = x[start:start + batch].to(device), y[start:start + batch].to(device)
        logits = model(xb)
        for name, value in loss_terms(logits, yb).items():
            sums[name] = sums.get(name, 0.0) + value.item() * len(xb)
        preds.extend(logits.argmax(1).cpu().numpy())
    model.train(training)
    labels = list(y.numpy())
    scores = segmentation_scores(preds, labels)
    per_class = {f"{prefix}_iou_{c}": v for c, v in scores["iou_per_class"].items()}
    scalars = {f"{prefix}_{k}": v for k, v in scores.items() if not k.endswith("_per_class")}
    shares = {f"{prefix}_{k}": v for k, v in status_shares(labels, preds).items()}
    tracks = {f"{prefix}_{k}": v for k, v in track_scores(preds, labels).items()}
    return {**{f"{prefix}_{name}": total / len(x) for name, total in sums.items()}, **scalars,
            f"{prefix}_building_iou": scores["iou_per_class"].get("building", float("nan")), **per_class, **shares,
            **tracks}
