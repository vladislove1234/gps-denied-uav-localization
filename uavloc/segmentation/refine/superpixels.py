"""Superpixel voting: each pixel takes the class of its RGB superpixel's mean LR-ASPP probability.

The mask edge then follows superpixel edges, which follow colour edges. Superpixels come from skimage SLIC,
skimage Felzenszwalb, or OpenCV watershed flooded from a regular seed grid (the fast one).
`masked_superpixel_refiner` floods the watershed only inside a pixel mask (for `band.band_refiner` in pixel mode):
pixels outside the mask are pre-labelled as one frozen region, so superpixels stop at the mask edge.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.segmentation import felzenszwalb, slic

from uavloc.features.segmentation import CLASSES
from uavloc.segmentation.eval import MaskedRefineFn, ModelOutput, RefineFn

METHODS = ("slic", "felzenszwalb", "watershed")
VOTES = ("all", "area")
TRACK = CLASSES.index("track")
SLIC_ITERATIONS = 10
FELZENSZWALB_SIGMA = 0.5
NO_SEGMENT = -1


@dataclass(frozen=True)
class SuperpixelMethod:
    """How the RGB image is over-segmented. `cell_px` is the target superpixel side (SLIC, watershed) or the
    minimum segment side (Felzenszwalb: min size = cell_px²)."""

    method: str = "slic"
    cell_px: int = 10
    compactness: float = 10.0
    scale: float = 100.0


@dataclass(frozen=True)
class VoteConfig:
    """`vote="area"` leaves the model's track pixels as they are and never votes a pixel into track.
    `alpha` mixes `alpha * superpixel mean + (1 - alpha) * pixel probs`."""

    vote: str = "all"
    alpha: float = 1.0


def seed_grid(shape: tuple[int, int], cell_px: int) -> np.ndarray:
    """int32 markers, one seed per `cell_px` cell at its centre, ids from 1."""
    rows, cols = np.arange(cell_px // 2, shape[0], cell_px), np.arange(cell_px // 2, shape[1], cell_px)
    markers = np.zeros(shape, np.int32)
    markers[np.ix_(rows, cols)] = np.arange(1, rows.size * cols.size + 1).reshape(rows.size, cols.size)
    return markers


def superpixels(rgb: np.ndarray, method: SuperpixelMethod) -> np.ndarray:
    """`(H, W)` int segment ids from 0; `NO_SEGMENT` marks pixels without one (watershed ridge lines)."""
    h, w = rgb.shape[:2]
    if method.method == "slic":
        return slic(rgb, n_segments=max(1, h * w // method.cell_px ** 2), compactness=method.compactness,
                    max_num_iter=SLIC_ITERATIONS, start_label=0)
    if method.method == "felzenszwalb":
        return felzenszwalb(rgb, scale=method.scale, sigma=FELZENSZWALB_SIGMA, min_size=method.cell_px ** 2)
    if method.method == "watershed":
        segments = cv2.watershed(np.ascontiguousarray(rgb), seed_grid((h, w), method.cell_px))
        return np.where(segments > 0, segments - 1, NO_SEGMENT)
    raise ValueError(f"unknown superpixel method {method.method!r}; expected one of {METHODS}")


def segment_mean(probs: np.ndarray, segments: np.ndarray) -> np.ndarray:
    """`(C, H, W)` probabilities averaged within each segment; pixels with `NO_SEGMENT` keep their own."""
    inside = segments >= 0
    ids = np.where(inside, segments, 0).ravel()
    n = int(ids.max()) + 1
    weights = inside.ravel().astype(np.float64)
    counts = np.maximum(np.bincount(ids, weights=weights, minlength=n), 1.0)
    pooled = np.stack([(np.bincount(ids, weights=p.ravel() * weights, minlength=n) / counts)[ids]
                       for p in probs]).reshape(probs.shape).astype(probs.dtype)
    return np.where(inside[None], pooled, probs)


def vote(probs: np.ndarray, segments: np.ndarray, config: VoteConfig) -> np.ndarray:
    """`(H, W)` uint8 class ids after superpixel voting."""
    if config.vote not in VOTES:
        raise ValueError(f"unknown vote {config.vote!r}; expected one of {VOTES}")
    mixed = config.alpha * segment_mean(probs, segments) + (1 - config.alpha) * probs
    if config.vote == "all":
        return mixed.argmax(0).astype(np.uint8)
    mixed[TRACK] = -np.inf
    return np.where(probs.argmax(0) == TRACK, TRACK, mixed.argmax(0)).astype(np.uint8)


def superpixel_refiner(method: SuperpixelMethod, config: VoteConfig) -> RefineFn:
    """A harness `RefineFn` that over-segments the RGB and votes the model's full-res probabilities."""
    def refine(output: ModelOutput, rgb: np.ndarray) -> np.ndarray:
        return vote(output.probs, superpixels(rgb, method), config)
    return refine


def watershed_at(rgb: np.ndarray, mask: np.ndarray, cell_px: int) -> np.ndarray:
    """Grid-seeded watershed flooded only inside `mask`; `(H, W)` segment ids, `NO_SEGMENT` outside the mask,
    on ridge lines, and on mask pixels that no in-mask seed reached."""
    markers = seed_grid(mask.shape, cell_px)
    frozen = int(markers.max()) + 1
    markers[~mask] = frozen
    segments = cv2.watershed(np.ascontiguousarray(rgb), markers)
    return np.where((segments > 0) & (segments != frozen), segments - 1, NO_SEGMENT)


def masked_superpixel_refiner(method: SuperpixelMethod, config: VoteConfig) -> MaskedRefineFn:
    """`(output, rgb, mask) -> (N,)` voted classes of the mask pixels; watershed only, flooded inside the mask."""
    if method.method != "watershed":
        raise ValueError(f"per-pixel superpixels need method 'watershed', got {method.method!r}")

    def refine(output: ModelOutput, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        rows, cols = np.nonzero(mask)
        segments = watershed_at(rgb, mask, method.cell_px)[rows, cols]
        return vote(output.probs[:, rows, cols], segments, config)
    return refine
