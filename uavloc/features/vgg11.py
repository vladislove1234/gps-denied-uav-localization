"""2025 VGG11 feature matcher: conv1..conv5_1 of VGG11, fine-tuned on drone/satellite pairs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.models import vgg11

from uavloc.data.satellite import GeoImage
from uavloc.features.imagenet import to_batch
from uavloc.geo import LatLon

VGG11_CONV5_1_END = 20  # features[:20] stops after conv5_1 + ReLU, before the last pool


class Vgg11FeatureNet(nn.Module):
    """Same layout and state-dict keys as the 2025 `VGG11FeatureNet`."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = vgg11(weights=None).features[:VGG11_CONV5_1_END]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def load_vgg11(checkpoint: Path, device: torch.device | str) -> Vgg11FeatureNet:
    model = Vgg11FeatureNet()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=True)
    return model.to(device).eval()


class Vgg11Features:
    """Both inputs squashed to squares, as in 2025: drone to `drone_input_px`, patch to `drone_input_px * search_scale`."""

    def __init__(self, model: nn.Module, satellite: GeoImage, drone_input_px: int, search_scale: float,
                 device: torch.device | str) -> None:
        self.model = model
        self.satellite = satellite
        self.drone_input_px = drone_input_px
        self.patch_input_px = int(drone_input_px * search_scale)
        self.device = device

    @torch.no_grad()
    def drone(self, image: Image.Image, width_m: float, height_m: float, scale: float = 1.0) -> torch.Tensor:
        """`scale` x the footprint at the patch's pixels per metre: the square input shrinks to `scale` x its side."""
        side_px = self.drone_input_px if scale == 1.0 else round(self.drone_input_px * scale)
        resized = image.resize((side_px, side_px), Image.BILINEAR)
        return self.model(to_batch(np.asarray(resized), self.device))

    @torch.no_grad()
    def satellite_patch(self, center: LatLon, yaw_deg: float, width_m: float, height_m: float) -> torch.Tensor:
        patch = self.satellite.crop_rotated(center, yaw_deg, width_m, height_m)
        resized = Image.fromarray(patch).resize((self.patch_input_px, self.patch_input_px), Image.BILINEAR)
        return self.model(to_batch(np.asarray(resized), self.device))
