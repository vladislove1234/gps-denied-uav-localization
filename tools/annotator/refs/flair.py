"""IGN FLAIR land cover, FPN/ResNet-34, RGB. Ported from
``pocs/flair-landcover-transfer/probe.ipynb``.

Fed at the tile's native 0.386 m/px: upsampling to FLAIR's 0.2 m lost 8 of 9
locations in the PoC (it fabricates detail the model then labels). FLAIR has
no road class, and agreed with itself on only 44-57% of pixels across two
dates -- so it is a per-location candidate, never a default truth.

Licence: Etalab 2.0.
"""

from __future__ import annotations

import numpy as np

from ._infer import device, tiled_logits
from .base import Adapter, Location, RefResult, register

REPO = "IGNF/FLAIR-INC_rgb_15cl_resnet34-fpn"
FLAIR_CLASSES = [
    "building", "pervious surface", "impervious surface", "bare soil", "water",
    "coniferous", "deciduous", "brushwood", "vineyard", "herbaceous vegetation",
    "agricultural land", "plowed land", "swimming pool", "snow",
    "clear cut", "mixed", "ligneous", "greenhouse", "other",
]
TO_OURS = {
    "building": ["building"],
    "water": ["water", "swimming pool"],
    "woody": ["coniferous", "deciduous", "brushwood"],
    "track": ["impervious surface"],          # no road class: tarmac only
    "field": ["agricultural land", "plowed land"],
}
MEAN = np.array([105.08, 110.87, 101.82], np.float32)
STD = np.array([52.17, 45.38, 44.00], np.float32)
PATCH, OVERLAP = 512, 64


@register
class FLAIR(Adapter):
    name = "flair"
    version = "1"
    provides = tuple(TO_OURS)
    params = {"repo": REPO, "mode": "native", "patch": PATCH, "overlap": OVERLAP,
              "to_ours": TO_OURS}

    def __init__(self):
        self.model = None

    def load(self):
        if self.model is None:
            import segmentation_models_pytorch as smp
            import torch
            from huggingface_hub import hf_hub_download, list_repo_files
            files = list_repo_files(REPO)
            weights = next(f for f in files if f.endswith((".ckpt", ".pth", ".pt", ".safetensors")))
            raw = torch.load(hf_hub_download(REPO, filename=weights), map_location="cpu")
            prefix = "model.seg_model."
            state = {k[len(prefix):]: v for k, v in raw.items() if k.startswith(prefix)}
            assert state, f"prefix {prefix!r} matched nothing"
            m = smp.FPN(encoder_name="resnet34", encoder_weights=None, in_channels=3,
                        classes=len(FLAIR_CLASSES))
            m.load_state_dict(state, strict=True)
            self.dev = device()
            self.model = m.eval().to(self.dev)
        return self.model

    def run(self, loc: Location) -> RefResult:
        model = self.load()
        x = (np.asarray(loc.image(), np.float32) - MEAN) / STD
        logits = tiled_logits(model, x, PATCH, PATCH - OVERLAP, len(FLAIR_CLASSES), self.dev, batch=4)
        pred = logits.argmax(0)
        masks = {ours: np.isin(pred, [FLAIR_CLASSES.index(c) for c in names])
                 for ours, names in TO_OURS.items()}
        return RefResult(masks=masks, info={"frac": {c: float(m.mean()) for c, m in masks.items()}})

    def close(self):
        self.model = None
