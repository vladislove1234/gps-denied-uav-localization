"""LR-ASPP with a second, binary boundary output read from stride-4 features (experiment 015-boundary-head).

The semantic path is LR-ASPP unchanged (`coarse_logits`, stride 8). The boundary head taps the MobileNetV3
backbone at stride 4 (layer 3, 24 ch) and at LR-ASPP's stride-8 `low` branch (layer 4, 40 ch) and predicts a
class-edge band at stride 4. At inference the band can guide a watershed that re-assigns the pixels inside it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage.segmentation import watershed
from torch import nn

from uavloc.data.annotated import IGNORE_LABEL
from uavloc.features.imagenet import normalise
from uavloc.features.segmentation import CLASSES, ENCODER_STRIDE_PX, TILE_PX, TILE_STEP_PX, LRASPPLogits
from uavloc.features.tiling import tiled_logits
from uavloc.segmentation.eval import STRIDE, ModelOutput, pad_to_multiple, upsample_logits

STRIDE4_TAP = "3"       # MobileNetV3-L features[3]: last stride-4 block, 24 ch
LOW_TAP = "4"           # LR-ASPP's `low` input: first stride-8 block, 40 ch
HIGH_TAP = "16"
STRIDE4_CHANNELS, LOW_CHANNELS = 24, 40
BOUNDARY_STRIDE_PX = 4
HEAD_CHANNELS = 16
BOUNDARY_THRESHOLD = 0.5
RIM_KERNEL = np.ones((3, 3), np.uint8)


class BoundaryHead(nn.Module):
    """Stride-4 and stride-8 features → one boundary logit per stride-4 cell."""

    def __init__(self, channels: int = HEAD_CHANNELS) -> None:
        super().__init__()
        self.reduce_stride4 = nn.Conv2d(STRIDE4_CHANNELS, channels, 1)
        self.reduce_low = nn.Conv2d(LOW_CHANNELS, channels, 1)
        self.fuse = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(2 * channels, channels, 3, padding=1),
                                  nn.ReLU(inplace=True), nn.Conv2d(channels, channels, 3, padding=1),
                                  nn.ReLU(inplace=True), nn.Conv2d(channels, 1, 1))

    def forward(self, stride4: torch.Tensor, low: torch.Tensor) -> torch.Tensor:
        low = F.interpolate(self.reduce_low(low), size=stride4.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([self.reduce_stride4(stride4), low], 1))


class LRASPPBoundary(nn.Module):
    """LR-ASPP plus an optional boundary head sharing its backbone; `head=None` is the control arm."""

    def __init__(self, n_classes: int = len(CLASSES), with_head: bool = True) -> None:
        super().__init__()
        self.segmenter = LRASPPLogits(n_classes)
        self.head = BoundaryHead() if with_head else None

    def features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        taps = {STRIDE4_TAP: "stride4", LOW_TAP: "low", HIGH_TAP: "high"}
        out = {}
        for name, module in self.segmenter.net.backbone.items():
            x = module(x)
            if name in taps:
                out[taps[name]] = x
        return out

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """`(B, C, H/8, W/8)` semantic logits and `(B, 1, H/4, W/4)` boundary logits (None without a head)."""
        f = self.features(x)
        semantic = self.segmenter.net.classifier({"low": f["low"], "high": f["high"]})
        return semantic, (self.head(f["stride4"], f["low"]) if self.head is not None else None)

    def coarse_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.segmenter.coarse_logits(x)

    def boundary_logits(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)
        return self.head(f["stride4"], f["low"])


def load_boundary_model(checkpoint: Path, device: torch.device | str, with_head: bool) -> LRASPPBoundary:
    """From a `LRASPPBoundary` state dict, or from a plain `LRASPPLogits` one (the head then starts fresh)."""
    model = LRASPPBoundary(with_head=with_head)
    state = torch.load(checkpoint, map_location="cpu")
    if any(k.startswith("segmenter.") for k in state):
        model.load_state_dict(state)
    else:
        model.segmenter.load_state_dict(state)
    return model.to(device)


def freeze_batchnorm(model: nn.Module) -> None:
    """Keep every BatchNorm on its running (satellite) statistics while the rest trains."""
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def dice_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Soft Dice over the classes present in the batch, ignored pixels excluded (round-3 recipe)."""
    ok = labels != IGNORE_LABEL
    p = logits.softmax(1) * ok[:, None]
    one_hot = F.one_hot(labels.clamp(max=len(CLASSES) - 1), len(CLASSES)).permute(0, 3, 1, 2) * ok[:, None]
    present = one_hot.sum((0, 2, 3)) > 0
    inter, den = (p * one_hot).sum((0, 2, 3)), p.sum((0, 2, 3)) + one_hot.sum((0, 2, 3))
    return (1 - (2 * inter + 1) / (den + 1))[present].mean()


def seg_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """The round-3 / LR-ASPP training loss: CE (ignoring 255) + soft Dice, on full-res logits."""
    return F.cross_entropy(logits, labels, ignore_index=IGNORE_LABEL) + dice_loss(logits, labels)


def boundary_target(labels: torch.Tensor, radius_px: int) -> tuple[torch.Tensor, torch.Tensor]:
    """`(B, H, W)` class ids → float band of pixels within `radius_px + 1` (chessboard) of a class edge, and a
    validity mask. An edge needs two different labelled classes; ignored pixels and the crop border make none,
    and pixels whose band window touches an ignored pixel are not scored."""
    ignored = (labels == IGNORE_LABEL)[:, None].float()
    ids = labels[:, None].float().masked_fill(ignored.bool(), -1)
    padded = F.pad(ids, (1, 1, 1, 1), mode="replicate")
    high = F.max_pool2d(padded, 3, stride=1)
    low = -F.max_pool2d(-padded.masked_fill(padded < 0, float(len(CLASSES))), 3, stride=1)
    edge = ((high != low) & (high >= 0) & (low < len(CLASSES)) & (ids >= 0)).float()
    window = 2 * radius_px + 1
    band = F.max_pool2d(edge, window, stride=1, padding=radius_px)
    valid = F.max_pool2d(ignored, window + 2, stride=1, padding=radius_px + 1) == 0
    return band[:, 0], valid[:, 0]


def boundary_loss(boundary_logits: torch.Tensor, labels: torch.Tensor, radius_px: int) -> torch.Tensor:
    """Class-balanced BCE (HED-style: positives weighted by n_neg / n_pos) on the full-res upsampled logits."""
    target, valid = boundary_target(labels, radius_px)
    up = F.interpolate(boundary_logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    n_pos = (target * valid).sum().clamp_min(1)
    pos_weight = (valid.sum() - n_pos).clamp_min(1) / n_pos
    losses = F.binary_cross_entropy_with_logits(up, target, pos_weight=pos_weight, reduction="none")
    return (losses * valid).sum() / valid.sum().clamp_min(1)


@dataclass(frozen=True)
class LossWeights:
    boundary: float = 1.0
    radius_px: int = 1


def multitask_loss(model: LRASPPBoundary, x: torch.Tensor, labels: torch.Tensor,
                   weights: LossWeights) -> tuple[torch.Tensor, dict[str, float]]:
    """Semantic loss on the ×8 bilinear upsample (as torchvision does), plus the boundary loss when there is a head."""
    semantic, boundary = model(x)
    semantic = F.interpolate(semantic, size=x.shape[-2:], mode="bilinear", align_corners=False)
    loss = seg_loss(semantic, labels)
    parts = {"seg": loss.item()}
    if boundary is not None:
        b_loss = boundary_loss(boundary, labels, weights.radius_px)
        loss = loss + weights.boundary * b_loss
        parts["boundary"] = b_loss.item()
    return loss, parts


@torch.no_grad()
def boundary_probabilities(model: LRASPPBoundary, rgb: np.ndarray, device: torch.device | str) -> np.ndarray:
    """`(H, W)` boundary probability, tiled like `lraspp_outputs` (512 px windows, logits averaged at stride 4)."""
    x = pad_to_multiple(normalise(rgb), STRIDE)
    h, w = rgb.shape[:2]
    logits = tiled_logits(model.boundary_logits, x, TILE_PX, TILE_STEP_PX, 1, device, out_stride=BOUNDARY_STRIDE_PX)
    return upsample_boundary(torch.from_numpy(logits)[None], (h, w))


@torch.no_grad()
def multitask_outputs(model: LRASPPBoundary, rgb: np.ndarray,
                      device: torch.device | str) -> tuple[ModelOutput, np.ndarray]:
    """One backbone pass on a tile (≤ 1024 px): the harness `ModelOutput` and the `(H, W)` boundary probability."""
    h, w = rgb.shape[:2]
    batch = torch.from_numpy(pad_to_multiple(normalise(rgb), ENCODER_STRIDE_PX).transpose(2, 0, 1).copy())[None]
    semantic, boundary = model(batch.to(device))
    logits = semantic[0].float().cpu().numpy()
    probs = torch.from_numpy(upsample_logits(logits, (h, w))).softmax(0).numpy()
    return ModelOutput(logits, probs), upsample_boundary(boundary.float().cpu(), (h, w))


def optimizer(model: LRASPPBoundary, lr: float, head_lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """AdamW with the pretrained segmenter at `lr` and the new head (if any) at `head_lr`."""
    groups = [{"params": model.segmenter.parameters(), "lr": lr}]
    if model.head is not None:
        groups.append({"params": model.head.parameters(), "lr": head_lr})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def upsample_boundary(logits: torch.Tensor, size: tuple[int, int]) -> np.ndarray:
    """`(1, 1, h, w)` stride-4 logits → `(H, W)` sigmoid probability, cropped to `size`."""
    full = (logits.shape[-2] * BOUNDARY_STRIDE_PX, logits.shape[-1] * BOUNDARY_STRIDE_PX)
    up = F.interpolate(logits, size=full, mode="bilinear", align_corners=False)[0, 0, :size[0], :size[1]]
    return up.sigmoid().numpy()


def watershed_fusion(probs: np.ndarray, boundary: np.ndarray, threshold: float = BOUNDARY_THRESHOLD) -> np.ndarray:
    """Keep the argmax where the boundary map is below `threshold`; flood the band above it from those seeds,
    with the boundary probability as the elevation, so class edges settle on the predicted boundary ridge.
    Flooding only enters band pixels, so it runs on the band plus its 1-px rim (same result, far less work)."""
    argmax = probs.argmax(0).astype(np.uint8)
    band = boundary >= threshold
    if band.all() or not band.any():
        return argmax
    markers = np.where(band, 0, argmax.astype(np.int32) + 1)
    flooded = watershed(boundary, markers, mask=cv2.dilate(band.astype(np.uint8), RIM_KERNEL).astype(bool))
    return np.where(band & (flooded > 0), flooded - 1, argmax).astype(np.uint8)


def band_counts(boundary: np.ndarray, label: np.ndarray, radius_px: int,
                threshold: float = BOUNDARY_THRESHOLD) -> np.ndarray:
    """`[true positive, predicted, target]` pixel counts of `boundary >= threshold` against the training band."""
    target, valid = boundary_target(torch.from_numpy(label.astype(np.int64))[None], radius_px)
    target, valid = target[0].numpy().astype(bool) & valid[0].numpy(), valid[0].numpy()
    pred = (boundary >= threshold) & valid
    return np.array([(pred & target).sum(), pred.sum(), target.sum()], np.int64)


def band_scores(counts: np.ndarray) -> dict[str, float]:
    """Precision, recall and F1 of the predicted band from summed `band_counts`."""
    tp, n_pred, n_target = counts.astype(float)
    precision, recall = tp / max(n_pred, 1.0), tp / max(n_target, 1.0)
    return {"precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}
