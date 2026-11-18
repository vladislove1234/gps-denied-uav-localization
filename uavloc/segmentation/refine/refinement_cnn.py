"""A tiny residual CNN that sharpens frozen LR-ASPP masks from full-res RGB (experiment 013-refinement-cnn).

Input: the ImageNet-normalised image `(B, 3, H, W)`, LR-ASPP's bilinear-upsampled log-probabilities `(B, C, H, W)`
and, optionally, the Sobel edge magnitude of the image. Output: corrected log-probability-like scores, the input
log-probabilities plus a residual. The residual head starts at zero, so an untrained refiner is the baseline.

For CPU cost the convolutions run on a space-to-depth view: the image is pixel-unshuffled by `unshuffle`
(lossless, every full-res pixel is still an input), and the head's residual is pixel-shuffled back to full res,
so each output pixel gets its own correction while no convolution runs at full resolution.

The crop bank, training loop and tiled inference also drive other residual refiners (017's `ResidualRefiner`):
any `nn.Module` with `forward(x, scores)` and a `config` giving `unshuffle` and `receptive_radius_px`. With
`as_logits=True` the scores are the bilinear-upsampled raw logits instead of log-probabilities.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm

from uavloc.data.annotated import IGNORE_LABEL
from uavloc.features.imagenet import IMAGENET_MEAN, IMAGENET_STD, UINT8_MAX, normalise, quantised_rgb
from uavloc.features.segmentation import CLASSES, LRASPPLogits
from uavloc.plotting.segmentation import centre_window, refinement_grid
from uavloc.segmentation.eval import STRIDE, Case, ModelOutput, RefineFn, pad_to_multiple, upsample_logits
from uavloc.segmentation.preview import PREVIEW_CROPS, PREVIEW_PX, PREVIEW_TAG
from uavloc.track import LOG_EVERY_STEPS, log_history, log_images

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

PROB_FLOOR = 1e-6                        # log floor, -13.8: bounds the log-probability input
GREY_WEIGHTS = (0.299, 0.587, 0.114)     # ITU-R BT.601 luma
SOBEL_X = ((-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 1.0))
REFINE_TILE_PX = 1024                    # inference tile on large images, bounds memory
FLIP_P = 0.5


@dataclass(frozen=True)
class RefinerConfig:
    width: int = 32
    dilations: tuple[int, ...] = (1, 2, 4)
    unshuffle: int = 4
    use_edges: bool = False
    n_classes: int = len(CLASSES)

    @property
    def receptive_radius_px(self) -> int:
        """Full-res radius, rounded up to whole unshuffled cells: stem 3x3, dilated 3x3 per block, one cell slack."""
        return self.unshuffle * (2 + sum(self.dilations))


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 3000
    batch: int = 16
    crop_px: int = 256
    lr: float = 3e-3
    weight_decay: float = 1e-4
    warmup_frac: float = 0.05
    log_every: int = LOG_EVERY_STEPS


def log_probs(probs: torch.Tensor) -> torch.Tensor:
    return probs.clamp_min(PROB_FLOOR).log()


def denormalise(x: torch.Tensor) -> torch.Tensor:
    """ImageNet-normalised `(B, 3, H, W)` back to RGB in [0, 1]."""
    mean = torch.as_tensor(IMAGENET_MEAN, device=x.device)[:, None, None]
    std = torch.as_tensor(IMAGENET_STD, device=x.device)[:, None, None]
    return x * std + mean


def luma(x: torch.Tensor) -> torch.Tensor:
    """Grayscale in [0, 1] of normalised `(B, 3, H, W)` input: `(B, 1, H, W)`."""
    return (denormalise(x) * torch.as_tensor(GREY_WEIGHTS, device=x.device)[:, None, None]).sum(1, keepdim=True)


def sobel_magnitude(x: torch.Tensor) -> torch.Tensor:
    """Sobel gradient magnitude of the luma of normalised `(B, 3, H, W)` input: `(B, 1, H, W)`."""
    grey = luma(x)
    kx = torch.as_tensor(SOBEL_X, device=x.device)
    kernels = torch.stack([kx, kx.T])[:, None]
    g = F.conv2d(F.pad(grey, (1, 1, 1, 1), mode="replicate"), kernels)
    return g.pow(2).sum(1, keepdim=True).sqrt()


class DilatedBlock(nn.Module):
    """Depthwise 3x3 (dilated) -> pointwise 1x1 -> ReLU, with a skip."""

    def __init__(self, width: int, dilation: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(width, width, 3, padding=dilation, dilation=dilation, groups=width)
        self.pointwise = nn.Conv2d(width, width, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + F.relu(self.pointwise(self.depthwise(h)))


class RefinementCNN(nn.Module):
    """`forward(x, log_p) -> log_p + residual`, all `(B, ., H, W)` with H, W multiples of `config.unshuffle`."""

    def __init__(self, config: RefinerConfig) -> None:
        super().__init__()
        self.config = config
        cell = config.unshuffle ** 2
        self.image_stem = nn.Conv2d((3 + int(config.use_edges)) * cell, config.width, 3, padding=1)
        self.logit_stem = nn.Conv2d(config.n_classes, config.width, 1, bias=False)
        self.blocks = nn.Sequential(*[DilatedBlock(config.width, d) for d in config.dilations])
        self.head = nn.Conv2d(config.width, config.n_classes * cell, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor, log_p: torch.Tensor) -> torch.Tensor:
        s = self.config.unshuffle
        image = torch.cat([x, sobel_magnitude(x)], 1) if self.config.use_edges else x
        h = F.relu(self.image_stem(F.pixel_unshuffle(image, s)) + self.logit_stem(F.avg_pool2d(log_p, s)))
        return log_p + F.pixel_shuffle(self.head(self.blocks(h)), s)


def n_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------- #
# inference
# --------------------------------------------------------------------------- #

def tile_starts(length: int, tile: int) -> list[int]:
    starts = list(range(0, length - tile + 1, tile))
    return starts + [length - tile] if starts[-1] + tile < length else starts


def input_scores(output: ModelOutput, size: tuple[int, int], as_logits: bool) -> np.ndarray:
    """`(C, H, W)` refiner input: bilinear-upsampled logits, or the log of the model's probabilities."""
    if as_logits:
        return upsample_logits(output.logits, size)
    return log_probs(torch.from_numpy(output.probs)).numpy()


@torch.no_grad()
def refine(model: nn.Module, output: ModelOutput, rgb: np.ndarray, device: torch.device | str,
           tile_px: int = REFINE_TILE_PX, as_logits: bool = False) -> np.ndarray:
    """`(H, W)` uint8 class ids: argmax of the refined scores, run in overlapping tiles (padded to the cell size)."""
    h, w = rgb.shape[:2]
    cell = model.config.unshuffle
    x = torch.from_numpy(pad_to_multiple(normalise(rgb), cell).transpose(2, 0, 1).copy())[None]
    scores = pad_to_multiple(input_scores(output, (h, w), as_logits).transpose(1, 2, 0), cell).transpose(2, 0, 1)
    log_p = torch.from_numpy(scores.copy())[None]
    (hp, wp), margin = x.shape[2:], model.config.receptive_radius_px
    tile_h, tile_w = min(tile_px, hp), min(tile_px, wp)
    pred = np.empty((hp, wp), np.uint8)
    for top in tile_starts(hp, tile_h):
        for left in tile_starts(wp, tile_w):
            t0, l0 = max(top - margin, 0), max(left - margin, 0)
            window = np.s_[:, :, t0:min(top + tile_h + margin, hp), l0:min(left + tile_w + margin, wp)]
            ids = model(x[window].to(device), log_p[window].to(device))[0].cpu().numpy().argmax(0)
            pred[top:top + tile_h, left:left + tile_w] = ids[top - t0:top - t0 + tile_h, left - l0:left - l0 + tile_w]
    return pred[:h, :w]


def refine_fn(model: nn.Module, device: torch.device | str, as_logits: bool = False) -> RefineFn:
    """The harness's `RefineFn` for a trained refiner."""
    model = model.to(device).eval()
    return lambda output, rgb: refine(model, output, rgb, device, as_logits=as_logits)


# --------------------------------------------------------------------------- #
# training data: frozen LR-ASPP outputs on the LR-ASPP training crops
# --------------------------------------------------------------------------- #

@dataclass
class CropBank:
    """`n` training crops: uint8 RGB `(n, H, W, 3)`, stride-8 LR-ASPP logits `(n, C, H/8, W/8)`, labels `(n, H, W)`."""

    rgb: np.ndarray
    logits: np.ndarray
    labels: np.ndarray


@torch.no_grad()
def build_crop_bank(segmenter: LRASPPLogits, sampler: Callable, n_crops: int, batch: int, seed: int,
                    device: torch.device | str) -> CropBank:
    """Draw `n_crops` augmented crops, quantise them to uint8, and store the frozen model's stride-8 logits."""
    rng = np.random.default_rng(seed)
    segmenter = segmenter.to(device).eval()
    rgbs, logits, labels = [], [], []
    for start in tqdm(range(0, n_crops, batch), desc="crop bank", unit="batch"):
        crops = [sampler(rng) for _ in range(min(batch, n_crops - start))]
        rgb = np.stack([quantised_rgb(x) for x, _ in crops])
        xb = torch.from_numpy(np.stack([normalise(r).transpose(2, 0, 1) for r in rgb])).to(device)
        logits.append(segmenter.coarse_logits(xb).float().cpu().numpy())
        rgbs.append(rgb)
        labels.append(np.stack([lab for _, lab in crops]).astype(np.uint8))
    return CropBank(np.concatenate(rgbs), np.concatenate(logits), np.concatenate(labels))


def val_crop_bank(cases: dict[str, list[Case]], crop_px: int) -> CropBank:
    """Every val image cut into `crop_px` crops on the stride-8 grid (the last row/column overlaps), with the
    stride-8 logits of the whole-image forward pass. Crops without labelled pixels are dropped."""
    rgbs, logits, labels = [], [], []
    cells = crop_px // STRIDE
    for case in (c for items in cases.values() for c in items):
        h, w = case.image.label.shape
        for top in (t - t % STRIDE for t in tile_starts(h, crop_px)):
            for left in (l - l % STRIDE for l in tile_starts(w, crop_px)):
                label = case.image.label[top:top + crop_px, left:left + crop_px]
                if (label == IGNORE_LABEL).all():
                    continue
                rgbs.append(case.image.rgb[top:top + crop_px, left:left + crop_px])
                logits.append(case.output.logits[:, top // STRIDE:top // STRIDE + cells,
                                                 left // STRIDE:left // STRIDE + cells])
                labels.append(label)
    return CropBank(np.stack(rgbs), np.stack(logits), np.stack(labels).astype(np.uint8))


def window_scores(logits: np.ndarray, top: int, left: int, crop_px: int, as_logits: bool = False) -> torch.Tensor:
    """Log-probs (or, `as_logits`, raw logits) of the full-res bilinear upsample of stride-8 `logits`, only over a
    `crop_px` window.

    `top`, `left` are multiples of the stride. Upsampling the logit block with a one-cell margin gives the same
    values inside the window as upsampling the whole crop, at a fraction of the cost.
    """
    row, col, cells = top // STRIDE, left // STRIDE, crop_px // STRIDE
    row0, col0 = max(row - 1, 0), max(col - 1, 0)
    block = logits[:, row0:row + cells + 1, col0:col + cells + 1]
    up = upsample_logits(block, (block.shape[1] * STRIDE, block.shape[2] * STRIDE))
    dy, dx = (row - row0) * STRIDE, (col - col0) * STRIDE
    window = torch.from_numpy(up[:, dy:dy + crop_px, dx:dx + crop_px].copy())
    return window if as_logits else log_probs(window.softmax(0))


def bank_batch(bank: CropBank, index: np.ndarray, crop_px: int, rng: np.random.Generator,
               device: torch.device | str, as_logits: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalised image, upsampled log-probs (or logits) and labels for `index`: random sub-crops on the stride-8 grid (the
    logit grid keeps the phase it has at inference), then a shared random rot90 and flip."""
    xs, lps, ys = [], [], []
    for k in index:
        h, w = bank.labels[k].shape
        top = STRIDE * int(rng.integers((h - crop_px) // STRIDE + 1))
        left = STRIDE * int(rng.integers((w - crop_px) // STRIDE + 1))
        win = np.s_[top:top + crop_px, left:left + crop_px]
        views = [torch.from_numpy(normalise(bank.rgb[k][win]).transpose(2, 0, 1).copy()),
                 window_scores(bank.logits[k], top, left, crop_px, as_logits),
                 torch.from_numpy(bank.labels[k][win].astype(np.int64))[None]]
        turns, flip = int(rng.integers(4)), rng.random() < FLIP_P
        views = [torch.rot90(v, turns, (1, 2)) for v in views]
        views = [v.flip(2) for v in views] if flip else views
        for acc, v in zip((xs, lps, ys), views):
            acc.append(v)
    return torch.stack(xs).to(device), torch.stack(lps).to(device), torch.cat(ys).to(device)


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def cross_entropy(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(scores, labels, ignore_index=IGNORE_LABEL)


@torch.no_grad()
def bank_losses(model: nn.Module, bank: CropBank, batch: int, device: torch.device | str,
                loss_fn: LossFn = cross_entropy, as_logits: bool = False) -> dict[str, float]:
    """Batch-mean CE and `loss_fn` of the refiner and of its unrefined input (`*_identity`) over a whole bank at
    full crop size."""
    model.eval()
    rng = np.random.default_rng(0)
    sums = {"ce": [], "ce_identity": [], "loss": [], "loss_identity": []}
    n, crop_px = bank.labels.shape[:2]
    for start in range(0, n, batch):
        x, scores, y = bank_batch(bank, np.arange(start, min(start + batch, n)), crop_px, rng, device, as_logits)
        refined = model(x, scores)
        for key, fn, s in (("ce", cross_entropy, refined), ("ce_identity", cross_entropy, scores),
                           ("loss", loss_fn, refined), ("loss_identity", loss_fn, scores)):
            sums[key].append(fn(s, y).item())
    return {k: float(np.mean(v)) for k, v in sums.items()}


@torch.no_grad()
def preview_predictions(model: nn.Module, bank: CropBank, n_crops: int, device: torch.device | str,
                        as_logits: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """RGB `(N, P, P, 3)`, labels, input and refined class ids `(N, P, P)` of `n_crops` evenly spaced bank crops,
    un-augmented, cut to the centre `PREVIEW_PX` window."""
    model.eval()
    index = np.linspace(0, len(bank.labels) - 1, n_crops).round().astype(int)
    crop_px = bank.labels.shape[1]
    x = torch.from_numpy(np.stack([normalise(bank.rgb[k]).transpose(2, 0, 1) for k in index])).to(device)
    scores = torch.stack([window_scores(bank.logits[k], 0, 0, crop_px, as_logits) for k in index]).to(device)
    before, after = scores.argmax(1).cpu().numpy(), model(x, scores).argmax(1).cpu().numpy()
    return tuple(centre_window(a, PREVIEW_PX) for a in (bank.rgb[index], bank.labels[index], before, after))


def log_previews(writer: SummaryWriter | None, model: nn.Module, banks: dict[str, CropBank], step: int,
                 device: torch.device | str, as_logits: bool = False, n_crops: int = PREVIEW_CROPS) -> None:
    """One `refinement_grid` per bank under `<name>/segmentation`, so TensorBoard's slider shows it evolve."""
    if writer is None:
        return
    log_images(writer, {f"{name}/{PREVIEW_TAG}": refinement_grid(*preview_predictions(model, bank, n_crops, device,
                                                                                       as_logits))
                        for name, bank in banks.items()}, step)


def bank_subset(bank: CropBank, n_crops: int) -> CropBank:
    """`n_crops` evenly spaced crops of `bank`, a fixed set to score at every log point."""
    index = np.linspace(0, len(bank.labels) - 1, min(n_crops, len(bank.labels))).round().astype(int)
    return CropBank(bank.rgb[index], bank.logits[index], bank.labels[index])


def eval_losses(model: nn.Module, train_eval: CropBank, holdout: CropBank, val: CropBank | None, batch: int,
                device: torch.device | str, loss_fn: LossFn = cross_entropy,
                as_logits: bool = False) -> dict[str, float]:
    """Every `bank_losses` term on the fixed train crops and, with `val`, on val (identical keys after the prefix),
    and the hold-out CE of refiner vs identity."""
    held = bank_losses(model, holdout, batch, device, loss_fn, as_logits)
    losses = {f"train_{k}": v for k, v in bank_losses(model, train_eval, batch, device, loss_fn, as_logits).items()}
    losses.update({"holdout_ce": held["ce"], "holdout_ce_identity": held["ce_identity"]})
    if val is not None:
        losses.update({f"val_{k}": v for k, v in bank_losses(model, val, batch, device, loss_fn, as_logits).items()})
    return losses


def train_refiner(model: nn.Module, train: CropBank, holdout: CropBank, config: TrainConfig, seed: int,
                  device: torch.device | str, loss_fn: LossFn = cross_entropy, as_logits: bool = False,
                  val: CropBank | None = None, writer: SummaryWriter | None = None) -> list[dict]:
    """AdamW + one-cycle on `loss_fn`. Every `log_every` steps the history gets the running train loss and CE of that
    window (`train_running_*`), the CE and loss of refiner vs identity on a fixed train subset the size of val (or
    hold-out) scored exactly like val (`train_*`), the hold-out CE and, with `val`, the same terms on val.
    Val is only watched, never used to pick a step. `writer` also gets the learning rate at each log point, the
    hold-out and val losses of the untrained model at step 0 (not in the returned history), and at step 0 and every
    log point a train and val segmentation grid (`log_previews`)."""
    rng = np.random.default_rng(seed)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=config.lr, total_steps=config.steps,
                                                pct_start=config.warmup_frac)
    history, window, window_ce, start = [], [], [], time.time()
    train_eval = bank_subset(train, len((val if val is not None else holdout).labels))
    previews = {"train": train, **({"val": val} if val is not None else {})}
    if writer is not None:
        log_history(writer, {"step": 0, **eval_losses(model, train_eval, holdout, val, config.batch, device, loss_fn,
                                                             as_logits)})
    log_previews(writer, model, previews, 0, device, as_logits)
    bar = tqdm(range(1, config.steps + 1), desc="train", unit="step")
    for step in bar:
        model.train()
        x, log_p, y = bank_batch(train, rng.integers(len(train.labels), size=config.batch), config.crop_px, rng,
                                 device, as_logits)
        scores = model(x, log_p)
        loss = loss_fn(scores, y)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        window.append(loss.item())
        window_ce.append(cross_entropy(scores.detach(), y).item())
        bar.set_postfix(loss=f"{window[-1]:.4f}", ce=f"{window_ce[-1]:.4f}", lr=f"{sched.get_last_lr()[0]:.1e}")
        if step % config.log_every == 0 or step == config.steps:
            entry = {"step": step, "minutes": (time.time() - start) / 60,
                     "train_running_loss": float(np.mean(window)), "train_running_ce": float(np.mean(window_ce)),
                     **eval_losses(model, train_eval, holdout, val, config.batch, device, loss_fn, as_logits)}
            history.append(entry)
            log_history(writer, {**entry, "train_lr": sched.get_last_lr()[0]})
            log_previews(writer, model, previews, step, device, as_logits)
            bar.write(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in entry.items()))
            window, window_ce = [], []
    return history


def is_finite(history: list[dict]) -> bool:
    return all(math.isfinite(v) for h in history for v in h.values() if isinstance(v, float))
