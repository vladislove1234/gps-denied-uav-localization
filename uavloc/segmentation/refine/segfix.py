"""SegFix-style boundary displacement (Yuan et al., ECCV 2020): a small full-res net predicts, per pixel,
whether it lies near a class edge and which of 8 directions points into its own object's interior; the
refiner copies the base prediction from `offset_px` along that direction onto each boundary pixel.

Targets come from label maps: a pixel is boundary when its own class's distance transform is <= `width_px`,
and its direction is the 8-bin quantised gradient of that distance transform. Ignored pixels never form
an edge (they count as "own class" in the distance transform) and carry no target.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from uavloc.data.annotated import IGNORE_LABEL
from uavloc.features.imagenet import normalise
from uavloc.segmentation.eval import ModelOutput, pad_to_multiple

N_DIRECTIONS = 8
DIRECTION_IGNORE = -1
_BIN_RAD = 2 * math.pi / N_DIRECTIONS
# (dy, dx) of each direction bin; bin b is angle b * 45 deg in image coordinates (y down).
DIRECTION_OFFSETS = np.array([(round(math.sin(b * _BIN_RAD)), round(math.cos(b * _BIN_RAD)))
                              for b in range(N_DIRECTIONS)], np.int64)
# Distance-transform slope below which the direction is undefined (ridge of a thin object).
MIN_GRADIENT = 0.25
_SOBEL_NORM = 1 / 8  # a 3x3 Sobel of a unit-slope ramp reads 8
NET_STRIDE_PX = 8
HEAD_STRIDE_PX = 2
BOUNDARY_THRESHOLD = 0.5


@dataclass(frozen=True)
class SegFixConfig:
    widths: tuple[int, int, int] = (16, 32, 64)
    context_dilations: tuple[int, ...] = (2, 4)


@dataclass(frozen=True)
class BoundaryField:
    """Per-pixel boundary probability `(H, W)` float32 and direction bin `(H, W)` uint8."""

    boundary_prob: np.ndarray
    direction: np.ndarray


def boundary_targets(label: np.ndarray, width_px: int) -> tuple[np.ndarray, np.ndarray]:
    """GT boundary mask `(H, W)` bool and direction bin `(H, W)` int64 (`DIRECTION_IGNORE` where undefined)."""
    ignored = label == IGNORE_LABEL
    boundary = np.zeros(label.shape, bool)
    direction = np.full(label.shape, DIRECTION_IGNORE, np.int64)
    present = np.bincount(label.ravel(), minlength=IGNORE_LABEL + 1)[:IGNORE_LABEL] > 0
    for cls in np.flatnonzero(present):
        own = label == cls
        dist = cv2.distanceTransform((own | ignored).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_5)
        band = own & (dist <= width_px)
        ys, xs = np.nonzero(band)
        gy = cv2.Sobel(dist, cv2.CV_32F, 0, 1, ksize=3, scale=_SOBEL_NORM)[ys, xs]
        gx = cv2.Sobel(dist, cv2.CV_32F, 1, 0, ksize=3, scale=_SOBEL_NORM)[ys, xs]
        defined = np.hypot(gx, gy) >= MIN_GRADIENT
        bins = np.round(np.arctan2(gy, gx) / _BIN_RAD).astype(np.int64) % N_DIRECTIONS
        boundary[ys, xs] = True
        direction[ys[defined], xs[defined]] = bins[defined]
    return boundary, direction


def batch_targets(labels: torch.Tensor, width_px: int) -> tuple[torch.Tensor, torch.Tensor]:
    """`(B, H, W)` labels → boundary `(B, H, W)` float with -1 on ignored pixels, direction `(B, H, W)` long."""
    boundaries, directions = [], []
    for lab in labels.numpy().astype(np.uint8):
        boundary, direction = boundary_targets(lab, width_px)
        boundaries.append(np.where(lab == IGNORE_LABEL, DIRECTION_IGNORE, boundary).astype(np.float32))
        directions.append(direction)
    return torch.from_numpy(np.stack(boundaries)), torch.from_numpy(np.stack(directions))


def conv_bn(c_in: int, c_out: int, stride: int = 1, dilation: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, stride, dilation, dilation, bias=False),
                         nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))


class SegFixNet(nn.Module):
    """Tiny encoder-decoder, stride 8 at the bottom, heads at stride 2 (full res is only needed for the loss).

    `(B, 3, H, W)` normalised RGB, H and W multiples of `NET_STRIDE_PX` → boundary logits `(B, 1, H/2, W/2)`,
    direction logits `(B, N_DIRECTIONS, H/2, W/2)`. The stride-2 merge is a 1x1 conv: 3x3 there dominated CPU time.
    """

    def __init__(self, config: SegFixConfig = SegFixConfig()) -> None:
        super().__init__()
        c1, c2, c3 = config.widths
        self.down1 = conv_bn(3, c1, stride=2)
        self.down2 = nn.Sequential(conv_bn(c1, c2, stride=2), conv_bn(c2, c2))
        self.down3 = nn.Sequential(conv_bn(c2, c3, stride=2),
                                   *[conv_bn(c3, c3, dilation=d) for d in config.context_dilations])
        self.up2 = conv_bn(c3 + c2, c2)
        self.up1 = nn.Sequential(nn.Conv2d(c2 + c1, c1, 1, bias=False), nn.BatchNorm2d(c1), nn.ReLU(inplace=True))
        self.head = nn.Conv2d(c1, 1 + N_DIRECTIONS, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f1 = self.down1(x)
        f2 = self.down2(f1)
        f3 = self.down3(f2)
        u2 = self.up2(torch.cat([F.interpolate(f3, size=f2.shape[2:], mode="bilinear"), f2], 1))
        out = self.head(self.up1(torch.cat([F.interpolate(u2, size=f1.shape[2:], mode="bilinear"), f1], 1)))
        return out[:, :1], out[:, 1:]


def full_res(logits: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(logits, size=size, mode="bilinear")


def segfix_loss(boundary_logits: torch.Tensor, direction_logits: torch.Tensor, boundary: torch.Tensor,
                direction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """BCE on labelled pixels (target -1 = ignored) and CE on GT boundary pixels with a defined direction.

    Stride-2 logits are upsampled bilinearly to the `(B, H, W)` targets.
    """
    size = boundary.shape[1:]
    boundary_logits, direction_logits = full_res(boundary_logits, size), full_res(direction_logits, size)
    labelled = boundary >= 0
    bce = F.binary_cross_entropy_with_logits(boundary_logits[:, 0][labelled], boundary[labelled])
    ce = F.cross_entropy(direction_logits, direction, ignore_index=DIRECTION_IGNORE)
    return bce, ce


@torch.no_grad()
def predict_field(net: SegFixNet, rgb: np.ndarray, device: torch.device | str) -> BoundaryField:
    """Run the net on a whole `(H, W, 3)` uint8 image (reflect-padded to the net stride).

    The boundary logit is upsampled bilinearly; the direction is argmaxed at stride 2 and repeated.
    """
    h, w = rgb.shape[:2]
    x = torch.from_numpy(pad_to_multiple(normalise(rgb), NET_STRIDE_PX).transpose(2, 0, 1).copy())[None]
    boundary_logits, direction_logits = net.eval()(x.to(device))
    prob = full_res(boundary_logits, x.shape[2:])[0, 0, :h, :w].sigmoid().cpu().numpy()
    coarse = direction_logits[0].argmax(0).to(torch.uint8).cpu().numpy()
    direction = coarse.repeat(HEAD_STRIDE_PX, 0).repeat(HEAD_STRIDE_PX, 1)[:h, :w]
    return BoundaryField(prob, direction)


def displace(pred: np.ndarray, boundary: np.ndarray, direction: np.ndarray, offset_px: int) -> np.ndarray:
    """`pred` with each `boundary` pixel relabelled from `offset_px` along its direction bin (clamped to the image)."""
    ys, xs = np.nonzero(boundary)
    step = DIRECTION_OFFSETS[direction[ys, xs]] * offset_px
    src_y = np.clip(ys + step[:, 0], 0, pred.shape[0] - 1)
    src_x = np.clip(xs + step[:, 1], 0, pred.shape[1] - 1)
    out = pred.copy()
    out[ys, xs] = pred[src_y, src_x]
    return out


def refine_with_field(pred: np.ndarray, field: BoundaryField, offset_px: int,
                      threshold: float = BOUNDARY_THRESHOLD) -> np.ndarray:
    return displace(pred, field.boundary_prob > threshold, field.direction, offset_px)


class SegFixRefiner:
    """`RefineFn`: LR-ASPP bilinear argmax, displaced along the net's predicted directions."""

    def __init__(self, net: SegFixNet, device: torch.device | str, offset_px: int,
                 threshold: float = BOUNDARY_THRESHOLD) -> None:
        self.net, self.device, self.offset_px, self.threshold = net.eval(), device, offset_px, threshold

    def __call__(self, output: ModelOutput, rgb: np.ndarray) -> np.ndarray:
        pred = output.probs.argmax(0).astype(np.uint8)
        return refine_with_field(pred, predict_field(self.net, rgb, self.device), self.offset_px, self.threshold)


def load_segfix(checkpoint: Path, device: torch.device | str, config: SegFixConfig = SegFixConfig()) -> SegFixNet:
    net = SegFixNet(config)
    net.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return net.to(device).eval()


class SegFixProbabilities:
    """`ProbabilityRefiner`: each boundary pixel takes the whole probability vector from `offset_px` along its
    direction. Its argmax is `SegFixRefiner`'s label; the soft map keeps the model's confidence for matching."""

    def __init__(self, net: SegFixNet, device: torch.device | str, offset_px: int,
                 threshold: float = BOUNDARY_THRESHOLD) -> None:
        self.net, self.device, self.offset_px, self.threshold = net.eval(), device, offset_px, threshold

    def __call__(self, probabilities: np.ndarray, rgb: np.ndarray) -> np.ndarray:
        field = predict_field(self.net, rgb, self.device)
        moved = displace(probabilities.transpose(1, 2, 0), field.boundary_prob > self.threshold, field.direction,
                         self.offset_px)
        return np.ascontiguousarray(moved.transpose(2, 0, 1))


def field_scores(field: BoundaryField, label: np.ndarray, width_px: int,
                 threshold: float = BOUNDARY_THRESHOLD) -> dict[str, float]:
    """Boundary precision/recall of the net against GT targets, and direction accuracy on GT boundary pixels."""
    boundary, direction = boundary_targets(label, width_px)
    labelled = label != IGNORE_LABEL
    predicted = (field.boundary_prob > threshold) & labelled
    hits = (predicted & boundary).sum()
    defined = direction != DIRECTION_IGNORE
    return {"boundary_precision": float(hits / max(predicted.sum(), 1)),
            "boundary_recall": float(hits / max(boundary.sum(), 1)),
            "direction_accuracy": float((field.direction[defined] == direction[defined]).mean())}
