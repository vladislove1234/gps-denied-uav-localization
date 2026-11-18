"""U-Net/ResNet-34 fine-tuned on LandCover.ai (rural Poland, 0.25 m). Ported from
``pocs/landcoverai-finetune/probe.ipynb``; weights stay in the PoC folder.

The only source with a connected, lane-following track network including
unpaved lanes. Known failure: over-claims water on ploughed fields (PoC:
19.5% claimed vs 0.2% actual) and track ~3x. Every class is imported anyway --
the person decides per location.

Licence of the training data: CC-BY-NC-SA 4.0.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .. import config
from ._infer import device, tiled_logits
from .base import Adapter, Location, RefResult, register

OURS = ["background", "building", "track", "woody", "water"]     # model output order
TARGET_MPP = 0.25
CROP = 256


@register
class LandCoverAI(Adapter):
    name = "landcoverai"
    version = "1"
    provides = ("building", "track", "woody", "water")
    params = {"weights": "pocs/landcoverai-finetune/model.pt", "arch": "Unet/resnet34",
              "target_mpp": TARGET_MPP, "crop": CROP}

    def __init__(self):
        self.model = None

    def load(self):
        if self.model is None:
            import segmentation_models_pytorch as smp
            import torch
            if not config.LANDCOVERAI_WEIGHTS.exists():
                raise FileNotFoundError(
                    f"LandCover.ai weights not found at {config.LANDCOVERAI_WEIGHTS}; "
                    "set LANDCOVERAI_WEIGHTS or re-run pocs/landcoverai-finetune")
            m = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3,
                         classes=len(OURS))
            m.load_state_dict(torch.load(config.LANDCOVERAI_WEIGHTS, map_location="cpu"))
            self.dev = device()
            self.model = m.eval().to(self.dev)
        return self.model

    def run(self, loc: Location) -> RefResult:
        model = self.load()
        img = loc.image()
        w0, h0 = img.size
        scale = loc.mpp / TARGET_MPP
        im = img.resize((int(w0 * scale), int(h0 * scale)), Image.BICUBIC)
        x = (np.asarray(im, np.float32) / 255.0 - 0.5) / 0.25
        logits = tiled_logits(model, x, CROP, CROP, len(OURS), self.dev)
        pred = logits.argmax(0).astype(np.uint8)
        pred = np.array(Image.fromarray(pred).resize((w0, h0), Image.NEAREST))
        masks = {c: pred == i for i, c in enumerate(OURS) if c != "background"}
        return RefResult(masks=masks, info={"frac": {c: float(m.mean()) for c, m in masks.items()}})

    def close(self):
        self.model = None
