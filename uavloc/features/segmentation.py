"""Class-probability maps from the village/drone segmentation models, used as matching features."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.segmentation import lraspp_mobilenet_v3_large

from uavloc.data.satellite import GeoImage
from uavloc.features.imagenet import normalise
from uavloc.features.classes import BACKGROUND, CLASSES
from uavloc.features.tiling import TILE_PX, TILE_STEP_PX, tiled_logits
from uavloc.geo import LatLon

UNET_ENCODER = "resnet34"
ENCODER_STRIDE_PX = 32
LRASPP_OUTPUT_STRIDE_PX = 8
SINGLE_PASS_MAX_PX = 1024  # drone frames at 0.10 m/px fit in one pass, as round 2 evaluated them
ARCHITECTURES = ("unet", "lraspp")
# Segmenters trained from ImageNet init in 021; `build_segmenter` name -> U-Net encoder or LR-ASPP.
SEGMENTERS = ("unet-resnet18", "unet-resnet34", "lraspp-mobilenetv3")
IMAGENET_WEIGHTS = "imagenet"
CHANNEL_SETS = {
    "all": tuple(range(len(CLASSES))),
    "no-background": tuple(i for i in range(len(CLASSES)) if i != BACKGROUND),
}

# Post-process of a model's class map: `(C, H, W)` probabilities and the `(H, W, 3)` uint8 image they came from
# → refined `(C, H, W)` probabilities.
ProbabilityRefiner = Callable[[np.ndarray, np.ndarray], np.ndarray]


class LRASPPLogits(nn.Module):
    """MobileNetV3-Large LR-ASPP (`pocs/mobilenet-lraspp-speed`); returns logits `(B, C, H, W)`, not torchvision's dict."""

    def __init__(self, n_classes: int, pretrained_backbone: bool = False) -> None:
        super().__init__()
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        self.net = lraspp_mobilenet_v3_large(weights=None, weights_backbone=weights, num_classes=n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)["out"]

    def coarse_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Head output before torchvision's bilinear upsample: `(B, C, H / 8, W / 8)` for H, W multiples of 8."""
        return self.net.classifier(self.net.backbone(x))


def load_segmenter(checkpoint: Path, device: torch.device | str, architecture: str = "unet") -> nn.Module:
    """A trained segmenter in eval mode; `architecture` is an `ARCHITECTURES` or a `SEGMENTERS` name."""
    if architecture not in ARCHITECTURES + SEGMENTERS:
        raise ValueError(f"architecture must be one of {ARCHITECTURES + SEGMENTERS}, got {architecture!r}")
    if architecture == "unet":
        return load_unet(checkpoint, device)
    model = build_segmenter(architecture, pretrained=False) if architecture in SEGMENTERS else LRASPPLogits(len(CLASSES))
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return model.to(device).eval()


def build_segmenter(name: str, pretrained: bool = True) -> nn.Module:
    """A fresh `SEGMENTERS` model with `len(CLASSES)` outputs, logits `(B, C, H, W)`; ImageNet encoder if `pretrained`."""
    if name not in SEGMENTERS:
        raise ValueError(f"segmenter must be one of {SEGMENTERS}, got {name!r}")
    if name == "lraspp-mobilenetv3":
        return LRASPPLogits(len(CLASSES), pretrained_backbone=pretrained)
    encoder = name.removeprefix("unet-")
    return smp.Unet(encoder, encoder_weights=IMAGENET_WEIGHTS if pretrained else None, classes=len(CLASSES))


def load_unet(checkpoint: Path, device: torch.device | str) -> nn.Module:
    model = smp.Unet(UNET_ENCODER, encoder_weights=None, classes=len(CLASSES))
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return model.to(device).eval()


@torch.no_grad()
def recalibrate_batchnorm(model: nn.Module, batches: Iterable[torch.Tensor], device: torch.device | str) -> nn.Module:
    """Copy of `model` whose BatchNorm statistics are the plain average over `batches`."""
    recalibrated = copy.deepcopy(model)
    for module in recalibrated.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.reset_running_stats()
            module.momentum = None
    recalibrated.train()
    for batch in batches:
        recalibrated(batch.to(device))
    return recalibrated.eval()


@torch.no_grad()
def class_probabilities(model: nn.Module, rgb: np.ndarray, device: torch.device | str) -> np.ndarray:
    """`(H, W, 3)` uint8 → `(C, H, W)` float32 softmax probabilities."""
    x = normalise(rgb)
    h, w = x.shape[:2]
    if max(h, w) > SINGLE_PASS_MAX_PX:
        logits = torch.from_numpy(tiled_logits(model, x, TILE_PX, TILE_STEP_PX, len(CLASSES), device))
    else:
        padded_h, padded_w = -(-h // ENCODER_STRIDE_PX) * ENCODER_STRIDE_PX, -(-w // ENCODER_STRIDE_PX) * ENCODER_STRIDE_PX
        batch = torch.from_numpy(x.transpose(2, 0, 1).copy())[None]
        batch = F.pad(batch, (0, padded_w - w, 0, padded_h - h), mode="reflect")
        logits = model(batch.to(device))[0, :, :h, :w].float().cpu()
    return logits.softmax(0).numpy()


def label_probabilities(labels: GeoImage, mpp: float, ignore_label: int = 255) -> GeoImage:
    """Class-id raster → one-hot `(H, W, C)` area-averaged to `mpp`; ignored pixels belong to no class."""
    ids = torch.from_numpy(labels.pixels.astype(np.int64))
    one_hot = F.one_hot(ids.clamp(max=len(CLASSES)), len(CLASSES) + 1)[..., :len(CLASSES)]
    one_hot[ids == ignore_label] = 0
    width_m, height_m = labels.size_m
    size = (round(height_m / mpp), round(width_m / mpp))
    averaged = F.interpolate(one_hot.permute(2, 0, 1)[None].float(), size=size, mode="area")[0]
    return labels.with_pixels(np.ascontiguousarray(averaged.permute(1, 2, 0).numpy()))


class SegmentationFeatures:
    """Drone and satellite each segmented at their own training scale, then compared on one metric grid.

    The satellite is segmented once, whole; each query cuts its patch from that probability map.
    `refine`, when given, post-processes both sides' maps at the model's input scale.
    """

    def __init__(self, drone_model: nn.Module, satellite_model: nn.Module, satellite: GeoImage, channels: str,
                 drone_gsd_m: float, match_gsd_m: float, device: torch.device | str,
                 refine: ProbabilityRefiner | None = None) -> None:
        at_match_gsd = satellite.resampled(match_gsd_m)
        probabilities = class_probabilities(satellite_model, at_match_gsd.pixels, device)
        if refine is not None:
            probabilities = refine(probabilities, at_match_gsd.pixels)
        self._setup(drone_model, at_match_gsd.with_pixels(np.ascontiguousarray(probabilities.transpose(1, 2, 0))),
                    channels, drone_gsd_m, match_gsd_m, device, refine)

    @classmethod
    def from_satellite_probabilities(cls, drone_model: nn.Module, probabilities: GeoImage, channels: str,
                                     drone_gsd_m: float, match_gsd_m: float, device: torch.device | str,
                                     refine: ProbabilityRefiner | None = None) -> SegmentationFeatures:
        """Satellite side given directly, e.g. a hand-labelled map from `label_probabilities`; `refine` is drone-only."""
        features = cls.__new__(cls)
        features._setup(drone_model, probabilities, channels, drone_gsd_m, match_gsd_m, device, refine)
        return features

    def _setup(self, drone_model: nn.Module, probabilities: GeoImage, channels: str, drone_gsd_m: float,
               match_gsd_m: float, device: torch.device | str, refine: ProbabilityRefiner | None) -> None:
        self.drone_model = drone_model
        self.probabilities = probabilities
        self.channels = list(CHANNEL_SETS[channels])
        self.drone_gsd_m = drone_gsd_m
        self.match_gsd_m = match_gsd_m
        self.device = device
        self.refine = refine

    def drone_probabilities(self, image: Image.Image, width_m: float, height_m: float) -> np.ndarray:
        """`(C, H, W)` at `match_gsd_m`, all classes, for a frame assumed to cover `width_m` x `height_m`."""
        at_model_gsd = image.resize((round(width_m / self.drone_gsd_m), round(height_m / self.drone_gsd_m)),
                                    Image.BILINEAR)
        rgb = np.asarray(at_model_gsd)
        probabilities = class_probabilities(self.drone_model, rgb, self.device)
        if self.refine is not None:
            probabilities = self.refine(probabilities, rgb)
        probabilities = torch.from_numpy(probabilities)
        size = (round(height_m / self.match_gsd_m), round(width_m / self.match_gsd_m))
        return F.interpolate(probabilities[None], size=size, mode="area")[0].numpy()

    def drone(self, image: Image.Image, width_m: float, height_m: float, scale: float = 1.0) -> torch.Tensor:
        """`scale` multiplies the footprint: the frame is taken to cover `scale * width_m` x `scale * height_m`."""
        probabilities = self.drone_probabilities(image, width_m * scale, height_m * scale)[self.channels]
        return torch.from_numpy(probabilities)[None].to(self.device)

    def satellite_patch(self, center: LatLon, yaw_deg: float, width_m: float, height_m: float) -> torch.Tensor:
        patch = self.probabilities.crop_rotated(center, yaw_deg, width_m, height_m, out_mpp=self.match_gsd_m)
        return torch.from_numpy(patch[..., self.channels].transpose(2, 0, 1).copy())[None].to(self.device)
