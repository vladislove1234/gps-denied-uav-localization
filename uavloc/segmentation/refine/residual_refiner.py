"""A full-resolution residual refiner over frozen LR-ASPP scores (experiment 017-gray-logit-refiner).

`forward(x, z) -> z + delta`: `x` is the ImageNet-normalised image `(B, 3, H, W)`, reduced to grayscale (or kept
as RGB), `z` the bilinear-upsampled LR-ASPP logits `(B, C, H, W)`. Layers, all at full resolution:
Conv 3x3 (1 or 3 + C -> width), one depthwise 3x3 per dilation, Conv 1x1 (width -> C). The 1x1 head starts at
zero, so an untrained refiner returns `z` unchanged. Training and inference reuse `refinement_cnn`'s crop bank,
loop and tiling with `as_logits=True`.

Two options address what `experiments/017-gray-logit-refiner/diagnose.ipynb` found. `input_norm` puts a BatchNorm on
the stacked input, because raw logits (std up to 7.5) swamp gray (std 0.13) about 20x at the stem. `pointwise_skip`
turns each depthwise layer into 013's `DilatedBlock` (depthwise -> 1x1 -> ReLU, plus a skip), because depthwise
layers with only a ReLU never mix channels, and 14 of 32 channels were dead by the last layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from uavloc.features.segmentation import CLASSES
from uavloc.segmentation.refine.refinement_cnn import DilatedBlock, luma

IMAGE_CHANNELS = {"gray": 1, "rgb": 3}


@dataclass(frozen=True)
class ResidualRefinerConfig:
    width: int = 32
    dilations: tuple[int, ...] = (1, 2, 4)
    image: str = "gray"                 # "gray" | "rgb"
    input_norm: bool = False
    pointwise_skip: bool = False
    n_classes: int = len(CLASSES)
    unshuffle: int = 1                  # full resolution; read by `refinement_cnn.refine` as the padding cell

    @property
    def receptive_radius_px(self) -> int:
        """Stem 3x3 plus one dilated 3x3 per block, plus one pixel of slack for tiled inference."""
        return 1 + sum(self.dilations) + 1


class ResidualRefiner(nn.Module):
    """`forward(x, z) -> z + delta_z`, all `(B, ., H, W)`."""

    def __init__(self, config: ResidualRefinerConfig) -> None:
        super().__init__()
        self.config = config
        in_channels = IMAGE_CHANNELS[config.image] + config.n_classes
        self.input_norm = nn.BatchNorm2d(in_channels) if config.input_norm else nn.Identity()
        self.stem = nn.Conv2d(in_channels, config.width, 3, padding=1)
        if config.pointwise_skip:
            self.depthwise = nn.ModuleList([DilatedBlock(config.width, d) for d in config.dilations])
        else:
            self.depthwise = nn.ModuleList([nn.Conv2d(config.width, config.width, 3, padding=d, dilation=d,
                                                      groups=config.width) for d in config.dilations])
        self.head = nn.Conv2d(config.width, config.n_classes, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def delta(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """The logit correction `delta_z`, `(B, C, H, W)`."""
        image = luma(x) if self.config.image == "gray" else x
        h = F.relu(self.stem(self.input_norm(torch.cat([image, z], 1))))
        for block in self.depthwise:
            h = block(h) if self.config.pointwise_skip else F.relu(block(h))
        return self.head(h)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return z + self.delta(x, z)
