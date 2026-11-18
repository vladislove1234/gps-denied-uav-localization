"""The three hand-labelled satellite val sets, the LR-ASPP outputs a refiner plugs into, and one-call scoring.

A refiner is any `RefineFn`: `(ModelOutput, rgb) -> (H, W)` class ids. `evaluate(refine_fn, cases)` scores it
on every val set with `segmentation_scores` and adds the unweighted mean over the sets.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm

from uavloc.data.annotated import VILLAGE_EXPORT, AnnotatedImage, load_annotated_images
from uavloc.features.imagenet import normalise
from uavloc.features.segmentation import (CLASSES, ENCODER_STRIDE_PX, LRASPP_OUTPUT_STRIDE_PX, SINGLE_PASS_MAX_PX,
                                          TILE_PX, TILE_STEP_PX, LRASPPLogits, class_probabilities, load_segmenter)
from uavloc.features.tiling import tiled_logits
from uavloc.segmentation.boundary import BF_TOLERANCES_PX, BOUNDARY_IOU_DILATION_PX
from uavloc.segmentation.metrics import segmentation_scores
from uavloc.track import REPO_ROOT

# The val split of pocs/mobilenet-lraspp-speed/probe.ipynb (= round 3): birky's top rows, bushcha whole.
STRIP_LOCATION = "birky-49p9105-23p9192"
STRIP_ROWS = 640
HOLDOUT_LOCATION = "bushcha-50p2422-25p8305"
GOOGLE_DATE = "google"
VAL_SETS = ("birky-strip", "birky-strip-google", "bushcha")
MEAN_KEY = "mean"
BASELINE_CHECKPOINT = REPO_ROOT / "pocs" / "mobilenet-lraspp-speed" / "model-lraspp-bn-satellite.pt"
STRIDE = LRASPP_OUTPUT_STRIDE_PX


@dataclass(frozen=True)
class ModelOutput:
    """LR-ASPP output for one `(H, W)` image.

    `logits` is the head output at stride 8, `(C, Hp / 8, Wp / 8)`, on the image reflect-padded to
    `(Hp, Wp) >= (H, W)`; cell `(i, j)` covers pixels `[8i, 8i + 8) x [8j, 8j + 8)`.
    `probs` is `(C, H, W)` softmax of its bilinear x8 upsample — exactly what the model's forward returns.
    """

    logits: np.ndarray
    probs: np.ndarray


@dataclass(frozen=True)
class Case:
    image: AnnotatedImage
    output: ModelOutput


RefineFn = Callable[[ModelOutput, np.ndarray], np.ndarray]
# `(output, rgb, mask) -> (N,)` class ids of the N `mask` pixels in row-major order (cost scales with N).
MaskedRefineFn = Callable[[ModelOutput, np.ndarray, np.ndarray], np.ndarray]


def load_val_sets(dataset: str = VILLAGE_EXPORT) -> dict[str, list[AnnotatedImage]]:
    """birky-strip (Esri dates), birky-strip-google, bushcha (unseen village, all dates), as probe.ipynb."""
    images = load_annotated_images(dataset)
    strip = [replace(a, rgb=a.rgb[:STRIP_ROWS], label=a.label[:STRIP_ROWS])
             for a in images if a.location == STRIP_LOCATION]
    return {"birky-strip": [a for a in strip if a.date != GOOGLE_DATE],
            "birky-strip-google": [a for a in strip if a.date == GOOGLE_DATE],
            "bushcha": [a for a in images if a.location == HOLDOUT_LOCATION]}


def load_baseline_model(device: torch.device | str) -> LRASPPLogits:
    """MobileNetV3-L LR-ASPP with satellite-domain BatchNorm statistics."""
    return load_segmenter(BASELINE_CHECKPOINT, device, architecture="lraspp")


def pad_to_multiple(x: np.ndarray, multiple: int) -> np.ndarray:
    """`(H, W, ...)` reflect-padded at the bottom and right up to multiples of `multiple`."""
    h, w = x.shape[:2]
    pad = ((0, -h % multiple), (0, -w % multiple)) + ((0, 0),) * (x.ndim - 2)
    return np.pad(x, pad, mode="reflect")


def upsample_logits(logits: np.ndarray, size: tuple[int, int], mode: str = "bilinear",
                    stride: int = STRIDE) -> np.ndarray:
    """Stride-8 `(C, h, w)` → `(C, H, W)`: the model's own x8 upsample (`bilinear` or `nearest`), cropped to `size`.
    Another `stride` upsamples by that factor instead."""
    full = (logits.shape[1] * stride, logits.shape[2] * stride)
    align = {"align_corners": False} if mode == "bilinear" else {}
    up = F.interpolate(torch.from_numpy(logits)[None], size=full, mode=mode, **align)[0]
    return up[:, :size[0], :size[1]].numpy()


@torch.no_grad()
def lraspp_outputs(model: LRASPPLogits, rgb: np.ndarray, device: torch.device | str) -> ModelOutput:
    """Stride-8 logits and full-res probabilities; tiled like `class_probabilities` for large images."""
    x = normalise(rgb)
    h, w = x.shape[:2]
    if max(h, w) > SINGLE_PASS_MAX_PX:
        logits = tiled_logits(model.coarse_logits, pad_to_multiple(x, STRIDE), TILE_PX, TILE_STEP_PX,
                              len(CLASSES), device, out_stride=STRIDE)
    else:
        batch = torch.from_numpy(pad_to_multiple(x, ENCODER_STRIDE_PX).transpose(2, 0, 1).copy())[None]
        logits = model.coarse_logits(batch.to(device))[0].float().cpu().numpy()
    probs = torch.from_numpy(upsample_logits(logits, (h, w))).softmax(0).numpy()
    return ModelOutput(logits, probs)


def run_model(model: LRASPPLogits, val_sets: dict[str, list[AnnotatedImage]],
              device: torch.device | str) -> dict[str, list[Case]]:
    """Model outputs for every val image, computed once so several refiners can be scored on them."""
    return {name: [Case(a, lraspp_outputs(model, a.rgb, device)) for a in tqdm(images, desc=f"LR-ASPP {name}")]
            for name, images in val_sets.items()}


def bilinear_argmax(output: ModelOutput, rgb: np.ndarray) -> np.ndarray:
    """The plain LR-ASPP prediction: argmax of the bilinear upsample."""
    return output.probs.argmax(0).astype(np.uint8)


def nearest_argmax(output: ModelOutput, rgb: np.ndarray) -> np.ndarray:
    """Argmax at stride 8, blown up as 8 x 8 blocks: the prediction with no upsampling at all."""
    return upsample_logits(output.logits, rgb.shape[:2], mode="nearest").argmax(0).astype(np.uint8)


def label_at_stride(label: np.ndarray, mode: str, stride: int = STRIDE) -> np.ndarray:
    """Oracle: the label's class shares per `stride` x `stride` cell, upsampled (`nearest` | `bilinear`) and argmaxed.

    The best any prediction at that stride can do with that upsample. Ignored pixels count for no class.
    """
    ids = torch.from_numpy(pad_to_multiple(label, stride).astype(np.int64))
    one_hot = F.one_hot(ids.clamp(max=len(CLASSES)), len(CLASSES) + 1)[..., :len(CLASSES)]
    shares = F.avg_pool2d(one_hot.permute(2, 0, 1)[None].float(), stride)[0].numpy()
    return upsample_logits(shares, label.shape, mode=mode, stride=stride).argmax(0).astype(np.uint8)


def score_sets(preds: dict[str, list[np.ndarray]], labels: dict[str, list[np.ndarray]],
               boundary_px: int = BOUNDARY_IOU_DILATION_PX,
               tolerances_px: tuple[int, ...] = BF_TOLERANCES_PX) -> dict[str, dict]:
    """`segmentation_scores` per set, plus `mean`: the unweighted mean of each scalar metric over the sets."""
    results = {name: segmentation_scores(preds[name], labels[name], boundary_px, tolerances_px) for name in labels}
    scalars = [k for k in results[next(iter(results))] if not k.endswith("_per_class")]
    results[MEAN_KEY] = {k: float(np.mean([results[n][k] for n in labels])) for k in scalars}
    return results


def evaluate_predictions(preds: dict[str, list[np.ndarray]], cases: dict[str, list[Case]],
                         boundary_px: int = BOUNDARY_IOU_DILATION_PX,
                         tolerances_px: tuple[int, ...] = BF_TOLERANCES_PX) -> dict[str, dict]:
    """Scores per val set, plus `mean`: the unweighted mean of each scalar metric over the sets."""
    labels = {name: [c.image.label for c in items] for name, items in cases.items()}
    return score_sets(preds, labels, boundary_px, tolerances_px)


@torch.no_grad()
def model_predictions(model: nn.Module, val_sets: dict[str, list[AnnotatedImage]],
                      device: torch.device | str) -> dict[str, list[np.ndarray]]:
    """Class ids of any segmenter on every val image, tiled by `class_probabilities`; the model is put in eval mode."""
    model.eval()
    return {name: [class_probabilities(model, a.rgb, device).argmax(0).astype(np.uint8)
                   for a in tqdm(images, desc=f"predict {name}")] for name, images in val_sets.items()}


def evaluate(refine_fn: RefineFn, cases: dict[str, list[Case]], boundary_px: int = BOUNDARY_IOU_DILATION_PX,
             tolerances_px: tuple[int, ...] = BF_TOLERANCES_PX) -> dict[str, dict]:
    """Run `refine_fn` on every case and score it; see `evaluate_predictions`."""
    preds = {name: [refine_fn(c.output, c.image.rgb) for c in tqdm(items, desc=f"predict {name}")]
             for name, items in cases.items()}
    return evaluate_predictions(preds, cases, boundary_px, tolerances_px)


def results_table(results: dict[str, dict]) -> pd.DataFrame:
    """One row per val set (and `mean`), one column per scalar metric."""
    return pd.DataFrame({name: {k: v for k, v in r.items() if not isinstance(v, dict)}
                         for name, r in results.items()}).T