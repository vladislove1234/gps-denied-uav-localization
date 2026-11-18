"""Which validation crops show what a loss term changes: score every window of a held-out image under each loss."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from uavloc.data.annotated import IGNORE_LABEL
from uavloc.segmentation.instances import BUILDING, building_status, components, overlap_fractions


@dataclass(frozen=True)
class CropRule:
    """What a crop must show: at least `min_buildings` labelled buildings, and every building-term loss merging at least
    `min_merge_drop` fewer of them than the base loss."""

    min_buildings: int
    min_merge_drop: int


def merged_buildings(label: np.ndarray, prediction: np.ndarray) -> int:
    """Labelled buildings that fall into a predicted building component shared with another labelled building."""
    gt, n_gt = components(label == BUILDING)
    if not n_gt:
        return 0
    pred, n_pred = components((prediction == BUILDING) & (label != IGNORE_LABEL))
    return building_status(overlap_fractions(gt, n_gt, pred, n_pred)).count("merged")


def score_windows(label: np.ndarray, predictions: Mapping[str, np.ndarray], size_px: int,
                  stride_px: int) -> pd.DataFrame:
    """One row per window free of unlabelled pixels: building count, and per loss the merged-building count."""
    rows = []
    for top in range(0, label.shape[0] - size_px + 1, stride_px):
        for left in range(0, label.shape[1] - size_px + 1, stride_px):
            window = label[top:top + size_px, left:left + size_px]
            if (window == IGNORE_LABEL).any():
                continue
            row = {"top": top, "left": left, "buildings": components(window == BUILDING)[1]}
            for loss, prediction in predictions.items():
                row[f"merged_{loss}"] = merged_buildings(window, prediction[top:top + size_px, left:left + size_px])
            rows.append(row)
    return pd.DataFrame(rows)


def disjoint(windows: pd.DataFrame, chosen: pd.DataFrame, size_px: int) -> pd.Series:
    """Windows that overlap none of the `chosen` windows (`place`, `top`, `left` corners)."""
    keep = pd.Series(True, index=windows.index)
    for place, top, left in chosen[["place", "top", "left"]].itertuples(index=False):
        apart = (windows["top"] - top).abs().ge(size_px) | (windows["left"] - left).abs().ge(size_px)
        keep &= (windows["place"] != place) | apart
    return keep


def pick_crops(windows: pd.DataFrame, rule: CropRule, size_px: int, count: int, base: str,
               term_losses: Sequence[str]) -> pd.DataFrame:
    """`count` disjoint windows where the building terms merge the fewest neighbouring buildings compared to `base`.

    `windows` comes from `score_windows` with a `place` column. Score: merged buildings under `base` minus under
    `term_losses[0]`, ties broken by the summed drop over all `term_losses`; every term loss must merge at least
    `rule.min_merge_drop` fewer buildings than `base`. Returns the picked rows with `merge_drop` and `merge_drop_total`.
    """
    drops = pd.DataFrame({loss: windows[f"merged_{base}"] - windows[f"merged_{loss}"] for loss in term_losses})
    scored = windows.assign(merge_drop=drops[term_losses[0]], merge_drop_total=drops.sum(axis=1))
    eligible = (scored["buildings"] >= rule.min_buildings) & drops.ge(rule.min_merge_drop).all(axis=1)
    ranked = scored[eligible].sort_values(["merge_drop", "merge_drop_total"], ascending=False)
    picks = ranked.iloc[:0]
    for index in ranked.index:
        if len(picks) == count:
            break
        if disjoint(ranked.loc[[index]], picks, size_px).iloc[0]:
            picks = pd.concat([picks, ranked.loc[[index]]])
    if len(picks) < count:
        raise ValueError(f"Only {len(picks)} of {count} disjoint windows pass {rule}")
    return picks.reset_index(drop=True)
