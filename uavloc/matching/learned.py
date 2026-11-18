"""Learned matching (experiment 023): heads that score a drone class map against a satellite patch.

Every head returns logits on the same cell grid as `weighted_ncc_heatmap`, so `decide` and the benchmark are
unchanged. Correlations run through the FFT on CPU: torch 2.2 has no FFT on MPS, and a batched direct conv with
a 68 x 120 kernel takes ~3 s per batch there.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm

from uavloc.matching.correlator import MIN_PRESENT_SHARE, MIN_WINDOW_STD, NCC_EPS, z_scored
from uavloc.plotting.correlation import heatmap_grid
from uavloc.track import log_history, log_images

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

    from uavloc.data.match_pairs import PairBank

ARMS = ("weights", "head")
LEARNED_VAR_EPS = 1e-6     # keeps NCC smooth on flat windows and flat queries of learned features
GROUP_NORM_GROUPS = 4      # per-sample normalisation: BatchNorm shared by query and patch mixes their statistics
WITHIN_M = 1.0             # `within_1m_share`: argmax this close to the truth


def fast_fft_size(n: int) -> int:
    """Smallest size >= n with no prime factor above 5: a prime size such as 109 makes the FFT 3x slower."""
    while True:
        m = n
        for p in (2, 3, 5):
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1


def xcorr_valid(satellite: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Per-channel sliding dot product, `(B, C, H, W)` x `(B, C, h, w)` -> `(B, C, H - h + 1, W - w + 1)`, on CPU.

    Zero-padding to a fast size leaves the valid positions untouched: the kernel never wraps there.
    """
    height, width = satellite.shape[-2:]
    size = (fast_fft_size(height), fast_fft_size(width))
    spectrum = torch.fft.rfft2(satellite.cpu(), s=size) * torch.fft.rfft2(kernel.cpu(), s=size).conj()
    full = torch.fft.irfft2(spectrum, s=size)
    return full[..., :height - kernel.shape[-2] + 1, :width - kernel.shape[-1] + 1]


def box_sum(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Sum over every `height` x `width` window, valid positions only, via an integral image in float64."""
    integral = F.pad(x.double(), (1, 0, 1, 0)).cumsum(-1).cumsum(-2)
    return (integral[..., height:, width:] - integral[..., :-height, width:]
            - integral[..., height:, :-width] + integral[..., :-height, :-width])


def batched_ncc(query: torch.Tensor, satellite: torch.Tensor, min_window_std: float,
                var_eps: float = 0.0) -> torch.Tensor:
    """`class_ncc` for every (sample, channel) at once, `(B, C, H', W')` on CPU.

    Windows whose satellite std is below `min_window_std` score 0, as in `class_ncc`. `var_eps` > 0 adds a floor to
    the window and kernel variance so a learned feature map that goes flat keeps a finite gradient.
    """
    query, satellite = query.cpu().float(), satellite.cpu().float()
    height, width = query.shape[-2:]
    n = height * width
    kernel = query - query.mean((2, 3), keepdim=True)
    numerator = xcorr_valid(satellite, kernel)
    mean = box_sum(satellite, height, width) / n
    std = ((box_sum(satellite ** 2, height, width) / n - mean ** 2).clamp(min=0) + var_eps).sqrt().float()
    kernel_norm = (kernel.flatten(2).pow(2).sum(2) + n * var_eps).sqrt()
    ncc = numerator / (kernel_norm[..., None, None] * std * math.sqrt(n) + NCC_EPS)
    return torch.where(std >= min_window_std, ncc, torch.zeros_like(ncc))


def class_presence(query: torch.Tensor) -> torch.Tensor:
    """`(B, C)` bool: the class is the argmax on at least `MIN_PRESENT_SHARE` of the query, as in `class_ncc_maps`."""
    labels = query.float().argmax(1).flatten(1).cpu()
    shares = F.one_hot(labels, query.shape[1]).float().mean(1)
    return shares >= MIN_PRESENT_SHARE


class ClassWeightHead(nn.Module):
    """007's weighted NCC with the class weights and a temperature learned; every present class takes part.

    Logits are `T * sum_c w_c * NCC_c` over present classes. 007 also divides by the present weights' sum and
    z-scores; neither moves the argmax, but both make the loss jump where a weight crosses 0 (a pair whose only
    present class has weight 0 goes from flat to full strength), so training uses the plain sum.
    """

    def __init__(self, class_weights: tuple[float, ...]) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.tensor(class_weights, dtype=torch.float32))
        self.log_temperature = nn.Parameter(torch.zeros(()))

    def forward(self, query: torch.Tensor, satellite: torch.Tensor, ncc: torch.Tensor,
                presence: torch.Tensor) -> torch.Tensor:
        device = self.weights.device
        weights = self.weights * presence.to(device).float()
        return (weights[..., None, None] * ncc.to(device)).sum(1) * self.log_temperature.exp()


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1),
                         nn.GroupNorm(GROUP_NORM_GROUPS, out_channels), nn.ReLU())


class CorrelationHead(nn.Module):
    """Shared encoder on both maps -> NCC of the encoded channels, with the raw per-class NCC maps -> small CNN."""

    def __init__(self, n_classes: int, encoder_channels: tuple[int, ...], refine_channels: int) -> None:
        super().__init__()
        widths = (n_classes, *encoder_channels)
        blocks = [conv_block(a, b) for a, b in zip(widths[:-2], widths[1:-1])]
        self.encoder = nn.Sequential(*blocks, nn.Conv2d(widths[-2], widths[-1], 3, padding=1))
        self.refine = nn.Sequential(nn.Conv2d(1 + n_classes, refine_channels, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(refine_channels, 1, 3, padding=1))

    def forward(self, query: torch.Tensor, satellite: torch.Tensor, ncc: torch.Tensor,
                presence: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        encoded = batched_ncc(self.encoder(query.to(device)), self.encoder(satellite.to(device)), 0.0, LEARNED_VAR_EPS)
        raw = ncc.cpu() * presence.cpu().float()[..., None, None]
        return self.refine(torch.cat([encoded.mean(1, keepdim=True), raw], 1).to(device))[:, 0]


def build_head(arm: str, class_weights: tuple[float, ...], encoder_channels: tuple[int, ...],
               refine_channels: int) -> nn.Module:
    if arm == "weights":
        return ClassWeightHead(class_weights)
    if arm == "head":
        return CorrelationHead(len(class_weights), encoder_channels, refine_channels)
    raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")


def soft_target(shape: tuple[int, int], cells: torch.Tensor, sigma_cells: float) -> torch.Tensor:
    """`(B, H', W')` Gaussians at fractional `(row, col)` cells, each summing to 1."""
    rows = torch.arange(shape[0], dtype=torch.float32)[None, :, None]
    cols = torch.arange(shape[1], dtype=torch.float32)[None, None, :]
    distance2 = (rows - cells[:, 0, None, None]) ** 2 + (cols - cells[:, 1, None, None]) ** 2
    target = torch.exp(-distance2 / (2 * sigma_cells ** 2))
    return target / target.sum((1, 2), keepdim=True)


def heatmap_loss(logits: torch.Tensor, cells: torch.Tensor, sigma_cells: float) -> torch.Tensor:
    """Cross-entropy over the surface cells against a Gaussian at the true cell."""
    target = soft_target(tuple(logits.shape[1:]), cells.cpu(), sigma_cells).to(logits.device)
    return -(target.flatten(1) * logits.flatten(1).log_softmax(1)).sum(1).mean()


def argmax_error_m(logits: torch.Tensor, cells: torch.Tensor, cell_m: tuple[float, float]) -> torch.Tensor:
    """Distance in metres from each surface's argmax to its true cell; `cell_m` is (row, col) size."""
    best = logits.detach().flatten(1).argmax(1).cpu()
    rows, cols = best // logits.shape[2], best % logits.shape[2]
    return torch.hypot((rows - cells[:, 0].cpu()) * cell_m[0], (cols - cells[:, 1].cpu()) * cell_m[1])


@dataclass(frozen=True)
class TrainConfig:
    steps: int
    batch_size: int
    lr: float
    weight_decay: float
    target_sigma_cells: float
    log_every_steps: int
    monitor_pairs: int
    preview_pairs: int
    seed: int
    warmup_frac: float = 0.1


def forward_batch(head: nn.Module, bank: PairBank, indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    batch = bank.batch(indices)
    return head(batch["query"], batch["satellite"], batch["ncc"], batch["presence"]), batch["cell"]


@torch.no_grad()
def evaluate(head: nn.Module, bank: PairBank, indices: np.ndarray, config: TrainConfig) -> dict[str, float]:
    """`loss`, `error_m` and `within_1m_share` over fixed pairs, in eval mode."""
    head.eval()
    losses, errors = [], []
    for start in range(0, len(indices), config.batch_size):
        chunk = indices[start:start + config.batch_size]
        logits, cells = forward_batch(head, bank, chunk)
        losses.append(heatmap_loss(logits, cells, config.target_sigma_cells).item() * len(chunk))
        errors.append(argmax_error_m(logits, cells, bank.cell_m))
    errors = torch.cat(errors)
    return {"loss": sum(losses) / len(indices), "error_m": float(errors.mean()),
            "within_1m_share": float((errors <= WITHIN_M).float().mean())}


@torch.no_grad()
def preview(head: nn.Module, bank: PairBank, indices: np.ndarray) -> np.ndarray:
    head.eval()
    logits, cells = forward_batch(head, bank, indices)
    return heatmap_grid(logits.float().cpu().numpy(), cells.numpy())


def monitor_indices(bank: PairBank, n: int) -> np.ndarray:
    """Evenly spaced pairs: the same pixels at every log point and for every seed."""
    return np.linspace(0, len(bank) - 1, min(n, len(bank))).astype(int)


def log_point(head: nn.Module, banks: dict[str, PairBank], monitors: dict[str, np.ndarray],
              config: TrainConfig) -> dict[str, float]:
    return {f"{split}_{key}": value for split, bank in banks.items()
            for key, value in evaluate(head, bank, monitors[split], config).items()}


def log_previews(writer: SummaryWriter | None, head: nn.Module, banks: dict[str, PairBank],
                 monitors: dict[str, np.ndarray], config: TrainConfig, step: int, suffix: str = "") -> None:
    if writer is not None:
        log_images(writer, {f"{split}/heatmaps{suffix}": preview(head, bank, monitors[split][:config.preview_pairs])
                            for split, bank in banks.items()}, step)


def train_head(head: nn.Module, train: PairBank, val: PairBank, config: TrainConfig, device: torch.device | str,
               writer: SummaryWriter | None = None) -> pd.DataFrame:
    """AdamW + one-cycle on `heatmap_loss`. Step 0 and every `log_every_steps`: train and val `loss`, `error_m`,
    `within_1m_share` on fixed pairs in eval mode, the running training loss, the learning rate and heatmap grids.
    The last step is the model: val is watched, never used to pick a step."""
    head = head.to(device)
    rng = np.random.default_rng(config.seed)
    opt = torch.optim.AdamW(head.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=config.lr, total_steps=config.steps,
                                                pct_start=config.warmup_frac)
    banks = {"train": train, "val": val}
    monitors = {split: monitor_indices(bank, config.monitor_pairs) for split, bank in banks.items()}
    start = time.time()
    history = [{"step": 0, "minutes": 0.0, "train_lr": sched.get_last_lr()[0], **log_point(head, banks, monitors, config)}]
    log_history(writer, history[0])
    log_previews(writer, head, banks, monitors, config, 0)
    window = []
    bar = tqdm(range(1, config.steps + 1), desc="train", unit="step")
    for step in bar:
        head.train()
        logits, cells = forward_batch(head, train, np.sort(rng.integers(len(train), size=config.batch_size)))
        loss = heatmap_loss(logits, cells, config.target_sigma_cells)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        window.append(loss.item())
        bar.set_postfix(loss=f"{window[-1]:.3f}", lr=f"{sched.get_last_lr()[0]:.1e}")
        if step % config.log_every_steps == 0 or step == config.steps:
            entry = {"step": step, "minutes": (time.time() - start) / 60, "train_running_loss": float(np.mean(window)),
                     "train_lr": sched.get_last_lr()[0], **log_point(head, banks, monitors, config)}
            history.append(entry)
            log_history(writer, entry)
            log_previews(writer, head, banks, monitors, config, step)
            bar.write(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in entry.items()))
            window = []
    log_previews(writer, head, banks, monitors, config, config.steps, "_final")
    head.eval()
    return pd.DataFrame(history)


class LearnedScorer:
    """A trained head as the correlator's `SurfaceScorer`: z-scored logits, NaN outside a round window."""

    def __init__(self, head: nn.Module, device: torch.device | str) -> None:
        self.head = head.to(device).eval()

    @torch.no_grad()
    def __call__(self, drone: torch.Tensor, satellite: torch.Tensor, mask: np.ndarray | None) -> np.ndarray:
        drone, satellite = drone.float(), satellite.float()
        ncc = batched_ncc(drone, satellite, MIN_WINDOW_STD)
        logits = self.head(drone, satellite, ncc, class_presence(drone))[0].float().cpu()
        return z_scored(logits, mask)
