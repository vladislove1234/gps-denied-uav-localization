"""Building instances: connected components of the building class, and per-building separate / merged / missed status.

Moved from `pocs/building-merge-stride/merge.ipynb` on second use (experiment 019-building-gap-loss). A labelled
building is *missed* when no predicted component covers >= `MATCH_FRAC` of it, *merged* when a component that covers
it also covers another labelled building, else *separate*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from uavloc.data.annotated import IGNORE_LABEL
from uavloc.features.segmentation import CLASSES

BUILDING = CLASSES.index("building")
M_PER_PX = 0.385  # the village exports' ground sampling distance
MIN_BUILDING_PX = 16  # 2.4 m²
MATCH_FRAC = 0.2
CONNECTIVITY = 1  # 4-connected
GAP_MARGIN_PX = 40  # nearest-gap search radius, 15.4 m
STATUSES = ("separate", "merged", "missed")
STRUCTURE = ndi.generate_binary_structure(2, CONNECTIVITY)


def components(mask: np.ndarray, min_px: int = MIN_BUILDING_PX) -> tuple[np.ndarray, int]:
    """`(H, W)` int32 component ids 1..n of `mask`, components with fewer than `min_px` pixels removed; and n."""
    lab, n = ndi.label(mask, STRUCTURE)
    keep = np.bincount(lab.ravel(), minlength=n + 1) >= min_px
    keep[0] = False
    relabel = np.zeros(n + 1, np.int32)
    relabel[keep] = np.arange(1, keep.sum() + 1)
    return relabel[lab], int(keep.sum())


def overlap_fractions(gt: np.ndarray, n_gt: int, pred: np.ndarray, n_pred: int) -> np.ndarray:
    """`(n_gt, n_pred)`: share of each labelled building's area covered by each predicted component."""
    both = (gt > 0) & (pred > 0)
    pairs = np.bincount((gt[both] - 1) * n_pred + (pred[both] - 1), minlength=n_gt * n_pred)
    area = np.bincount(gt.ravel(), minlength=n_gt + 1)[1:]
    return pairs.reshape(n_gt, n_pred) / np.maximum(area, 1)[:, None]


def building_status(frac: np.ndarray, match_frac: float = MATCH_FRAC) -> list[str]:
    """Per labelled building (rows of `overlap_fractions`): missed, merged or separate."""
    hit = frac >= match_frac
    shared = hit.sum(0) >= 2
    return ["missed" if not row.any() else "merged" if shared[row].any() else "separate" for row in hit]


def nearest_gap_px(gt: np.ndarray, n_gt: int, margin_px: int = GAP_MARGIN_PX) -> np.ndarray:
    """Per labelled building: Euclidean distance to the nearest other building, capped at `margin_px`."""
    gaps = np.full(n_gt, float(margin_px))
    for i, box in enumerate(ndi.find_objects(gt)):
        rows = slice(max(box[0].start - margin_px, 0), box[0].stop + margin_px)
        cols = slice(max(box[1].start - margin_px, 0), box[1].stop + margin_px)
        crop = gt[rows, cols]
        other = (crop > 0) & (crop != i + 1)
        if other.any():
            gaps[i] = min(ndi.distance_transform_edt(~other)[crop == i + 1].min(), margin_px)
    return gaps


def building_masks(label: np.ndarray, maps: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Boolean building masks of `maps` (class-id arrays), restricted to labelled pixels."""
    valid = label != IGNORE_LABEL
    return {key: (ids == BUILDING) & valid for key, ids in maps.items()}


def building_table(label: np.ndarray, maps: dict[str, np.ndarray], m_per_px: float = M_PER_PX) -> pd.DataFrame:
    """One row per labelled building: id, area, gap to its nearest neighbour and its status under each of `maps`."""
    gt, n_gt = components(label == BUILDING)
    table = pd.DataFrame({"building": np.arange(1, n_gt + 1),
                          "area_m2": np.bincount(gt.ravel(), minlength=n_gt + 1)[1:] * m_per_px ** 2,
                          "gap_m": nearest_gap_px(gt, n_gt) * m_per_px})
    for key, mask in building_masks(label, maps).items():
        pred, n_pred = components(mask)
        table[key] = building_status(overlap_fractions(gt, n_gt, pred, n_pred))
    return table


def merge_scores(table: pd.DataFrame, model: str, oracle: str) -> dict[str, float]:
    """Status shares of `model`, plus its merge rate among buildings `oracle` keeps separate (separable) and among
    those `oracle` merges (resolution-bound)."""
    shares = table[model].value_counts(normalize=True)
    separable = table[oracle] == "separate"
    bound = table[oracle] == "merged"
    merged = table[model] == "merged"
    return {**{f"{s}_share": float(shares.get(s, 0.0)) for s in STATUSES},
            "separable_merge_rate": float(merged[separable].mean()) if separable.any() else float("nan"),
            "resolution_bound_merge_rate": float(merged[bound].mean()) if bound.any() else float("nan"),
            "separable_buildings": int(separable.sum()), "buildings": len(table)}


def merged_by_gap(table: pd.DataFrame, keys: list[str], bins_m: list[float]) -> pd.DataFrame:
    """Share of buildings merged under each of `keys`, by the gap to their nearest neighbour (bins in metres)."""
    gap_bin = pd.cut(table.gap_m, bins_m + [np.inf], right=False)
    out = pd.DataFrame({k: (table[k] == "merged").groupby(gap_bin, observed=False).mean() for k in keys})
    out["buildings"] = table.groupby(gap_bin, observed=False).size()
    return out


def status_shares(labels: list[np.ndarray], preds: list[np.ndarray]) -> dict[str, float]:
    """Separate / merged / missed shares of all labelled buildings over a set of `(label, pred)` pairs.

    Buildings cut by a crop edge count as they appear in the crop. No gaps are computed, so it is cheap enough for
    every log point of a training.
    """
    counts = dict.fromkeys(STATUSES, 0)
    for label, pred in zip(labels, preds):
        gt, n_gt = components(label == BUILDING)
        if not n_gt:
            continue
        pred_ids, n_pred = components((pred == BUILDING) & (label != IGNORE_LABEL))
        for status in building_status(overlap_fractions(gt, n_gt, pred_ids, n_pred)):
            counts[status] += 1
    total = sum(counts.values())
    return {f"building_{s}_share": counts[s] / total if total else float("nan") for s in STATUSES}


def status_by_area(table: pd.DataFrame, keys: list[str], bins_m2: list[float]) -> pd.DataFrame:
    """Share of buildings missed and merged under each of `keys`, by labelled building area (bins in m²)."""
    area_bin = pd.cut(table.area_m2, bins_m2 + [np.inf], right=False)
    out = pd.DataFrame({f"{k} {s}": (table[k] == s).groupby(area_bin, observed=False).mean()
                        for k in keys for s in ("missed", "merged")})
    out["buildings"] = table.groupby(area_bin, observed=False).size()
    return out
