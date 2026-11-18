"""Segmentation training losses: the round-3 CE + soft Dice, and a cross-entropy on the label-boundary band.

`ce_dice_loss` is the loss `pocs/mobilenet-lraspp-speed/probe.ipynb` trained LR-ASPP with (defined inline there),
moved here on second use. `boundary_ce` is the CE averaged over pixels within `band_px` of a label edge, the
extra term of experiment 016-boundary-loss (`ce_dice_loss + lambda * boundary_ce`). `gap_weighted_ce` is the loss of
019-building-gap-loss: CE up-weighted on the `gap_band` between two nearby buildings (Ronneberger et al. 2015).
`structure_loss_terms` (021) adds a per-building CE weight, Tversky on building and track, and skeleton recall on
track centrelines to that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from skimage.morphology import skeletonize

from uavloc.data.annotated import IGNORE_LABEL
from uavloc.features.segmentation import CLASSES
from uavloc.segmentation.instances import BUILDING, MIN_BUILDING_PX, components

DICE_SMOOTH = 1.0
NEIGHBOURHOOD_PX = 3  # 3 x 3: a pixel is on an edge when a different labelled class touches it
NO_COMPONENT = np.iinfo(np.int32).max
TRACK = CLASSES.index("track")
BUILDING_MEDIAN_PX = 1046  # median building area in the 021 training labels (birky + dubrova, 4,055 buildings)


def valid_one_hot(y: torch.Tensor, n_classes: int, ignore: int = IGNORE_LABEL) -> torch.Tensor:
    """`(B, H, W)` ids → `(B, C, H, W)` float one-hot, all zero at ignored pixels."""
    ok = y != ignore
    return F.one_hot(y.clamp(max=n_classes - 1), n_classes).permute(0, 3, 1, 2).float() * ok[:, None]


def dice_loss(logits: torch.Tensor, y: torch.Tensor, ignore: int = IGNORE_LABEL) -> torch.Tensor:
    """Soft Dice over the batch, averaged over classes present in `y`."""
    ok = (y != ignore)[:, None]
    p = logits.softmax(1) * ok
    onehot = valid_one_hot(y, logits.shape[1], ignore)
    present = onehot.sum((0, 2, 3)) > 0
    inter, den = (p * onehot).sum((0, 2, 3)), p.sum((0, 2, 3)) + onehot.sum((0, 2, 3))
    return (1 - (2 * inter + DICE_SMOOTH) / (den + DICE_SMOOTH))[present].mean()


def ce_dice_loss(logits: torch.Tensor, y: torch.Tensor, ignore: int = IGNORE_LABEL,
                 dice_weight: float = 1.0) -> torch.Tensor:
    """CE (ignoring `ignore`) + `dice_weight` * soft Dice; weight 1 is the round-3 / LR-ASPP training loss."""
    return F.cross_entropy(logits, y, ignore_index=ignore) + dice_weight * dice_loss(logits, y, ignore)


@torch.no_grad()
def boundary_band(y: torch.Tensor, n_classes: int, band_px: int, ignore: int = IGNORE_LABEL) -> torch.Tensor:
    """`(B, H, W)` bool: labelled pixels within `band_px` (chessboard) of an edge between two labelled classes.

    Edges against ignored pixels are not edges, as in `uavloc.segmentation.boundary`.
    """
    onehot = valid_one_hot(y, n_classes, ignore)
    near = F.max_pool2d(onehot, NEIGHBOURHOOD_PX, stride=1, padding=NEIGHBOURHOOD_PX // 2)
    ok = y != ignore
    edge = (near.sum(1) > 1) & ok
    band = F.max_pool2d(edge[:, None].float(), 2 * band_px + 1, stride=1, padding=band_px)[:, 0] > 0
    return band & ok


def boundary_ce(logits: torch.Tensor, y: torch.Tensor, band_px: int, ignore: int = IGNORE_LABEL) -> torch.Tensor:
    """Cross-entropy averaged over the `boundary_band` pixels only; zero when the batch has no edge."""
    band = boundary_band(y, logits.shape[1], band_px, ignore)
    if not band.any():
        return logits.sum() * 0.0
    return F.cross_entropy(logits, y, ignore_index=ignore, reduction="none")[band].mean()



@torch.no_grad()
def building_ids(y: torch.Tensor) -> torch.Tensor:
    """`(B, H, W)` int32 CPU tensor: `instances.components` of the building class in each crop, 0 = no building."""
    return torch.from_numpy(np.stack([components(crop == BUILDING)[0] for crop in y.cpu().numpy()]))


@torch.no_grad()
def gap_band(y: torch.Tensor, radius_px: int, ignore: int = IGNORE_LABEL,
             ids: torch.Tensor | None = None) -> torch.Tensor:
    """`(B, H, W)` bool: labelled non-building pixels within `radius_px` (chessboard) of two different buildings.

    Buildings are `instances.components` of the building class in each crop of `y`, so the band follows the crop.
    Pass `ids` (from `building_ids`) to reuse components already computed for the batch.
    """
    ids = (building_ids(y) if ids is None else ids)[:, None]
    window, pad = 2 * radius_px + 1, radius_px
    highest = F.max_pool2d(ids.float(), window, stride=1, padding=pad)
    lowest = -F.max_pool2d(-ids.masked_fill(ids == 0, NO_COMPONENT).float(), window, stride=1, padding=pad)
    two = ((highest > 0) & (lowest < highest))[:, 0]
    return two.to(y.device) & (y != ignore) & (y != BUILDING)


def gap_loss_terms(logits: torch.Tensor, y: torch.Tensor, band: torch.Tensor, gap_weight: float,
                   ignore: int = IGNORE_LABEL) -> dict[str, torch.Tensor]:
    """`loss` = `mean(w * CE) + Dice`, `w = 1 + gap_weight * band`, and its parts: plain `ce`, `dice`, and
    `gap_ce` (CE on the band only; zero when the batch has no band)."""
    ce = F.cross_entropy(logits, y, ignore_index=ignore, reduction="none")
    n_labelled = (y != ignore).sum().clamp(min=1)
    dice = dice_loss(logits, y, ignore)
    w = 1.0 + gap_weight * band.float()
    return {"loss": (w * ce).sum() / n_labelled + dice, "ce": ce.sum() / n_labelled, "dice": dice,
            "gap_ce": ce[band].mean() if band.any() else ce.sum() * 0.0}


def gap_weighted_ce(logits: torch.Tensor, y: torch.Tensor, band: torch.Tensor, gap_weight: float,
                    ignore: int = IGNORE_LABEL) -> torch.Tensor:
    """`mean(w * CE) + Dice` over labelled pixels, `w = 1 + gap_weight * band`; `gap_weight = 0` is `ce_dice_loss`."""
    return gap_loss_terms(logits, y, band, gap_weight, ignore)["loss"]



@dataclass(frozen=True)
class StructureLossConfig:
    """Weights of `structure_loss_terms`. All extra weights 0 is the round-3 CE + Dice (`CURRENT_LOSS`).

    Buildings: `gap_*` is 019's up-weight of background between two buildings (merges). `instance_*` adds CE weight
    `instance_weight * min(instance_ref_px / area, instance_max_ratio)` on a building's pixels, so a small house counts
    about as much as a large one (misses). `tversky_*` is a Tversky loss on the building class; `beta > alpha`
    punishes missed pixels more than false ones.
    Tracks: `skeleton_*` is skeleton recall (Kirchhoff et al. 2024) on the labelled track centreline dilated by
    `skeleton_dilation_px`, so a break in a road costs more than a thin road. `track_tversky_*` as for buildings.
    """

    gap_weight: float = 0.0
    gap_radius_px: int = 10
    instance_weight: float = 0.0
    instance_ref_px: float = BUILDING_MEDIAN_PX
    instance_max_ratio: float = 10.0
    tversky_weight: float = 0.0
    tversky_alpha: float = 0.3
    tversky_beta: float = 0.7
    skeleton_weight: float = 0.0
    skeleton_dilation_px: int = 1
    track_tversky_weight: float = 0.0
    track_tversky_alpha: float = 0.3
    track_tversky_beta: float = 0.7
    dice_weight: float = 1.0


CURRENT_LOSS = StructureLossConfig()


@torch.no_grad()
def instance_weights(ids: torch.Tensor, ref_px: float, max_ratio: float) -> torch.Tensor:
    """`(B, H, W)` float: `min(ref_px / area, max_ratio)` on each building's pixels (area within its crop), 0 elsewhere."""
    out = torch.zeros(ids.shape, dtype=torch.float32)
    for b, crop in enumerate(ids):
        area = torch.bincount(crop.flatten().long()).float()
        ratio = (ref_px / area.clamp(min=1)).clamp(max=max_ratio)
        ratio[0] = 0.0
        out[b] = ratio[crop.long()]
    return out


@torch.no_grad()
def class_skeleton(y: torch.Tensor, cls: int, dilation_px: int, min_px: int = MIN_BUILDING_PX) -> torch.Tensor:
    """`(B, H, W)` bool: the centreline of class `cls` in each crop, dilated by `dilation_px` (chessboard) and kept
    inside the class mask, so a perfect prediction scores 0 (Kirchhoff et al.'s tubed skeleton). Pieces of the class
    under `min_px` are label slivers and get no centreline."""
    mask = torch.from_numpy(np.stack([components(crop == cls, min_px)[0] > 0 for crop in y.cpu().numpy()]))
    skeleton = torch.from_numpy(np.stack([skeletonize(crop) for crop in mask.numpy()]))
    if dilation_px:
        window = 2 * dilation_px + 1
        skeleton = F.max_pool2d(skeleton[:, None].float(), window, stride=1, padding=dilation_px)[:, 0] > 0
    return (skeleton & mask).to(y.device)


def skeleton_recall_loss(logits: torch.Tensor, skeleton: torch.Tensor, cls: int) -> torch.Tensor:
    """`1 - sum(p_cls * skeleton) / sum(skeleton)` over the batch; 0 when there is no skeleton."""
    if not skeleton.any():
        return logits.sum() * 0.0
    p = logits.softmax(1)[:, cls]
    return 1 - (p * skeleton).sum() / skeleton.sum()


def tversky_loss(logits: torch.Tensor, y: torch.Tensor, cls: int, alpha: float, beta: float,
                 ignore: int = IGNORE_LABEL) -> torch.Tensor:
    """`1 - TP / (TP + alpha FP + beta FN)` of class `cls` over the batch, soft; 0 when `cls` is absent and unpredicted."""
    ok = (y != ignore).float()
    p = logits.softmax(1)[:, cls] * ok
    t = (y == cls).float() * ok
    tp = (p * t).sum()
    fp = (p * (1 - t)).sum()
    fn = ((1 - p) * t).sum()
    return 1 - (tp + DICE_SMOOTH) / (tp + alpha * fp + beta * fn + DICE_SMOOTH)


def structure_loss_terms(logits: torch.Tensor, y: torch.Tensor, config: StructureLossConfig,
                         ignore: int = IGNORE_LABEL) -> dict[str, torch.Tensor]:
    """`loss` = `mean(w * CE) + dice_weight * Dice + tversky_weight * Tversky(building)
    + skeleton_weight * SkeletonRecall(track) + track_tversky_weight * Tversky(track)`, and every part.

    `w = 1 + gap_weight * gap_band + instance_weight * instance_weights`. Every term is returned whatever its
    weight, so runs with different losses log the same curves.
    """
    ids = building_ids(y)
    band = gap_band(y, config.gap_radius_px, ignore, ids).to(logits.device)
    inst = instance_weights(ids, config.instance_ref_px, config.instance_max_ratio).to(logits.device)
    skeleton = class_skeleton(y, TRACK, config.skeleton_dilation_px)
    ce = F.cross_entropy(logits, y, ignore_index=ignore, reduction="none")
    n_labelled = (y != ignore).sum().clamp(min=1)
    w = 1.0 + config.gap_weight * band.float() + config.instance_weight * inst
    dice = dice_loss(logits, y, ignore)
    tversky = tversky_loss(logits, y, BUILDING, config.tversky_alpha, config.tversky_beta, ignore)
    skeleton_recall = skeleton_recall_loss(logits, skeleton, TRACK)
    track_tversky = tversky_loss(logits, y, TRACK, config.track_tversky_alpha, config.track_tversky_beta, ignore)
    building, track = y == BUILDING, y == TRACK
    zero = ce.sum() * 0.0
    loss = ((w * ce).sum() / n_labelled + config.dice_weight * dice + config.tversky_weight * tversky
            + config.skeleton_weight * skeleton_recall + config.track_tversky_weight * track_tversky)
    return {"loss": loss, "ce": ce.sum() / n_labelled, "weighted_ce": (w * ce).sum() / n_labelled, "dice": dice,
            "tversky_building": tversky, "skeleton_recall_track": skeleton_recall, "tversky_track": track_tversky,
            "gap_ce": ce[band].mean() if band.any() else zero,
            "building_ce": ce[building].mean() if building.any() else zero,
            "track_ce": ce[track].mean() if track.any() else zero,
            "gap_share": band.float().mean(), "track_skeleton_share": skeleton.float().mean(),
            "building_weight_mean": w[building].mean() if building.any() else zero}
