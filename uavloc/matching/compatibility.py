"""Score an offset by class-pair compatibility instead of same-class NCC (experiment 024).

Two scorers over soft class maps, both registered as `CorrelatorConfig.matching` modes:

- `"pmi"`: a 6 x 6 table `W_ij = log P(i, j) / (P(i) P(j))` counted on aligned cross-date pairs; the score of an
  offset is `sum_ij W_ij (drone_i * sat_j)` over the drone window, per drone cell (Chan et al., CVPR 2003).
- `"mi"`: mutual information of the joint class distribution of the drone window and the satellite window at
  each offset; needs no counting.

Both drop the hand weights, `MIN_PRESENT_SHARE` and `MIN_WINDOW_STD`, and report coverage as NaN. Correlations go
through the FFT on CPU (`xcorr_valid`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from tqdm.auto import tqdm

from uavloc.matching.correlator import CorrelatorConfig, Matcher, register_matching, z_scored
from uavloc.matching.learned import xcorr_valid

if TYPE_CHECKING:
    from uavloc.data.match_pairs import PairBank

PMI_SMOOTHING = 1e-4        # added to every P(i, j) before the log, so an unseen pair gets a finite penalty
MI_FLOOR = 1e-8             # floor on p_ij and on the marginals inside the MI log
PMI_BATCH_PAIRS = 256
PMI_MODE = "pmi"
MI_MODE = "mi"


def aligned_joint(query: torch.Tensor, satellite: torch.Tensor, cells: torch.Tensor) -> torch.Tensor:
    """Sum over pairs and drone cells of `query_i * satellite_j` with the query at its true cell, `(C, C)` float64.

    `query` `(N, C, h, w)`, `satellite` `(N, C, H, W)`, `cells` `(N, 2)` fractional `(row, col)`, rounded to the
    nearest cell.
    """
    height, width = query.shape[-2:]
    joint = torch.zeros(query.shape[1], satellite.shape[1], dtype=torch.float64)
    for drone, sat, (row, col) in zip(query, satellite, cells.round().long().tolist()):
        window = sat[:, row:row + height, col:col + width]
        joint += torch.einsum("ihw,jhw->ij", drone.cpu().double(), window.cpu().double())
    return joint


def pmi_from_joint(joint: torch.Tensor | np.ndarray, smoothing: float = PMI_SMOOTHING) -> np.ndarray:
    """`W_ij = log P(i, j) / (P(i) P(j))`; P is the normalised joint plus `smoothing`, renormalised, and P(i), P(j)
    are its row and column sums."""
    p = np.asarray(joint, dtype=np.float64)
    p = p / p.sum() + smoothing
    p = p / p.sum()
    return np.log(p) - np.log(p.sum(1, keepdims=True)) - np.log(p.sum(0, keepdims=True))


def bank_joint(bank: PairBank, batch_size: int = PMI_BATCH_PAIRS, indices: np.ndarray | None = None) -> np.ndarray:
    """`aligned_joint` over the bank's pairs (all, or `indices`), `(C, C)` sum of soft co-occurrences."""
    indices = np.arange(len(bank)) if indices is None else np.asarray(indices)
    joint = None
    for start in tqdm(range(0, len(indices), batch_size), desc="pmi pairs", unit="batch"):
        batch = bank.batch(indices[start:start + batch_size])
        part = aligned_joint(batch["query"], batch["satellite"], batch["cell"])
        joint = part if joint is None else joint + part
    return joint.numpy()


def estimate_pmi(bank: PairBank, smoothing: float = PMI_SMOOTHING, batch_size: int = PMI_BATCH_PAIRS) -> np.ndarray:
    """The `(C, C)` PMI table of drone class i (rows) over satellite class j (columns) at the true offset."""
    return pmi_from_joint(bank_joint(bank, batch_size), smoothing)


def pmi_heatmap(drone: torch.Tensor, satellite: torch.Tensor, weights: np.ndarray,
                mask: np.ndarray | None = None) -> np.ndarray:
    """z-scored `sum_ij W_ij (drone_i * sat_j)` per drone cell; maps `(1, C, h, w)` and `(1, C, H, W)`.

    One correlation per satellite class: the kernel of class j is `sum_i W_ij drone_i`.
    """
    drone, satellite = drone.float().cpu(), satellite.float().cpu()
    w = torch.as_tensor(weights, dtype=torch.float32)
    kernel = torch.einsum("ij,nihw->njhw", w, drone)
    score = xcorr_valid(satellite, kernel)[0].sum(0) / drone[0, 0].numel()
    return z_scored(score, mask)


def mi_heatmap(drone: torch.Tensor, satellite: torch.Tensor, mask: np.ndarray | None = None) -> np.ndarray:
    """z-scored mutual information of the drone and satellite class maps at every offset.

    `p_ij(o)` is the mean over the drone window of `drone_i * sat_j(. + o)` (36 correlations); the marginals are
    its row and column sums.
    """
    # float64: MI differences between offsets are small, and float32 FFT noise shows up after z-scoring
    drone, satellite = drone[0].cpu().double(), satellite[0].cpu().double()
    n_cells = drone[0].numel()
    joint = xcorr_valid(satellite[None], drone[:, None]) / n_cells        # (C_drone, C_sat, H', W')
    joint = joint.clamp(min=MI_FLOOR)
    drone_marginal = joint.sum(1, keepdim=True).clamp(min=MI_FLOOR)
    sat_marginal = joint.sum(0, keepdim=True).clamp(min=MI_FLOOR)
    mi = (joint * (joint.log() - drone_marginal.log() - sat_marginal.log())).sum((0, 1))
    return z_scored(mi.float(), mask)


def make_pmi_matcher(weights: np.ndarray) -> Matcher:
    """A `Matcher` scoring with a fixed PMI table."""
    def match(drone: torch.Tensor, satellite: torch.Tensor, config: CorrelatorConfig,
              mask: np.ndarray | None) -> tuple[np.ndarray, float]:
        return pmi_heatmap(drone, satellite, weights, mask), float("nan")
    return match


def mi_matcher(drone: torch.Tensor, satellite: torch.Tensor, config: CorrelatorConfig,
               mask: np.ndarray | None) -> tuple[np.ndarray, float]:
    return mi_heatmap(drone, satellite, mask), float("nan")


def register_pmi(weights: np.ndarray) -> None:
    """Make `"pmi"` a matching mode scored with `weights`; call before building a config that uses it."""
    register_matching(PMI_MODE, make_pmi_matcher(weights))


register_matching(MI_MODE, mi_matcher)
