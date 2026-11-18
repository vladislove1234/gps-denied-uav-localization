"""Band refinement: run an expensive refiner only in an uncertain band of the model's mask, keep argmax elsewhere.

The band is the pixels within `distance_px` of a predicted class boundary, and/or the pixels whose top-1 minus
top-2 probability is below `margin`. Only band pixels take the refined class; every other pixel keeps the
model's bilinear argmax. Three ways to run the base refiner (`BandConfig.crop`):
- `frame`: a `RefineFn` on the whole frame, output masked to the band (quality reference, no saving);
- `tiles`: a `RefineFn` on each `tile_px` square that touches the band, with `context_px` of extra context so
  its border effects stay outside the tile;
- `pixels`: a `MaskedRefineFn` given the band itself, for refiners that can work on a pixel subset.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from uavloc.segmentation.eval import STRIDE, MaskedRefineFn, ModelOutput, RefineFn

CROPS = ("frame", "tiles", "pixels")
DISTANCE_MASK_PX = 5   # cv2.distanceTransform mask size for the L2 metric


@dataclass(frozen=True)
class BandConfig:
    """Band definition and how the base refiner is cropped to it. Tile and context sides are multiples of the
    model stride (8 px) so the stride-8 logits crop exactly; 24-px multiples also keep a 6-px seed grid aligned."""

    distance_px: float | None = 6.0   # within this Euclidean distance of a predicted class boundary; None = off
    margin: float | None = None       # top-1 minus top-2 probability below this; None = off
    crop: str = "tiles"               # one of CROPS
    tile_px: int = 96
    context_px: int = 24


def predicted_boundary(ids: np.ndarray) -> np.ndarray:
    """Pixels of an `(H, W)` class-id map with a 4-neighbour of another class (both sides of each edge)."""
    edge = np.zeros(ids.shape, bool)
    vertical, horizontal = ids[1:] != ids[:-1], ids[:, 1:] != ids[:, :-1]
    edge[1:] |= vertical
    edge[:-1] |= vertical
    edge[:, 1:] |= horizontal
    edge[:, :-1] |= horizontal
    return edge


def distance_band(ids: np.ndarray, distance_px: float) -> np.ndarray:
    """Pixels within Euclidean `distance_px` of a class boundary of `ids`."""
    boundary = predicted_boundary(ids)
    if not boundary.any():
        return boundary
    distance = cv2.distanceTransform((~boundary).astype(np.uint8), cv2.DIST_L2, DISTANCE_MASK_PX)
    return distance <= distance_px


def margin_band(probs: np.ndarray, margin: float) -> np.ndarray:
    """Pixels whose top-1 minus top-2 class probability of `(C, H, W)` `probs` is below `margin`."""
    top2 = np.partition(probs, -2, axis=0)[-2:]
    return top2[1] - top2[0] < margin


def uncertain_band(output: ModelOutput, config: BandConfig, ids: np.ndarray | None = None) -> np.ndarray:
    """`(H, W)` bool: the union of the configured distance and margin bands. `ids` = the argmax, if already known."""
    band = np.zeros(output.probs.shape[1:], bool)
    if config.distance_px is not None:
        band |= distance_band(output.probs.argmax(0) if ids is None else ids, config.distance_px)
    if config.margin is not None:
        band |= margin_band(output.probs, config.margin)
    return band


def band_tiles(band: np.ndarray, tile_px: int) -> list[tuple[int, int]]:
    """Top-left `(row, col)` px of every `tile_px` square tile that contains a band pixel."""
    h, w = band.shape
    rows, cols = -(-h // tile_px), -(-w // tile_px)
    padded = np.zeros((rows * tile_px, cols * tile_px), bool)
    padded[:h, :w] = band
    hit = padded.reshape(rows, tile_px, cols, tile_px).any(axis=(1, 3))
    return [(int(r) * tile_px, int(c) * tile_px) for r, c in zip(*np.nonzero(hit))]


def crop_output(output: ModelOutput, top: int, left: int, bottom: int, right: int) -> ModelOutput:
    """The model output restricted to pixels `[top, bottom) x [left, right)`; `top`, `left` multiples of STRIDE."""
    logits = output.logits[:, top // STRIDE:-(-bottom // STRIDE), left // STRIDE:-(-right // STRIDE)]
    return ModelOutput(np.ascontiguousarray(logits), np.ascontiguousarray(output.probs[:, top:bottom, left:right]))


def check_config(config: BandConfig) -> None:
    if config.crop not in CROPS:
        raise ValueError(f"crop must be one of {CROPS}, got {config.crop!r}")
    if config.tile_px <= 0 or config.tile_px % STRIDE or config.context_px % STRIDE or config.tile_px < 0 or config.context_px < 0:
        raise ValueError(f"tile_px and context_px must be non-negative multiples of {STRIDE}, got {config}")


def refine_in_band(base_refine_fn: RefineFn | MaskedRefineFn, output: ModelOutput, rgb: np.ndarray,
                   ids: np.ndarray, band: np.ndarray, config: BandConfig) -> np.ndarray:
    """`(H, W)` uint8: the base refiner's classes on `band` pixels, the argmax `ids` elsewhere."""
    out = ids.astype(np.uint8)
    if config.crop == "frame":
        return np.where(band, base_refine_fn(output, rgb), out).astype(np.uint8)
    if config.crop == "pixels":
        out[band] = base_refine_fn(output, rgb, band)
        return out
    h, w = band.shape
    tile, context = config.tile_px, config.context_px
    for top, left in band_tiles(band, tile):
        bottom, right = min(h, top + tile), min(w, left + tile)
        c_top, c_left = max(0, top - context), max(0, left - context)
        c_bottom, c_right = min(h, bottom + context), min(w, right + context)
        refined = base_refine_fn(crop_output(output, c_top, c_left, c_bottom, c_right),
                                 np.ascontiguousarray(rgb[c_top:c_bottom, c_left:c_right]))
        inside = band[top:bottom, left:right]
        out[top:bottom, left:right][inside] = refined[top - c_top:bottom - c_top, left - c_left:right - c_left][inside]
    return out


def band_refiner(base_refine_fn: RefineFn | MaskedRefineFn, config: BandConfig) -> RefineFn:
    """A harness `RefineFn` that runs `base_refine_fn` only in the uncertain band of `config`.
    `base_refine_fn` is a `MaskedRefineFn` for `crop="pixels"`, a `RefineFn` otherwise."""
    check_config(config)

    def refine(output: ModelOutput, rgb: np.ndarray) -> np.ndarray:
        ids = output.probs.argmax(0)
        return refine_in_band(base_refine_fn, output, rgb, ids, uncertain_band(output, config, ids), config)
    return refine


def band_fractions(output: ModelOutput, config: BandConfig) -> tuple[float, float]:
    """Share of pixels in the band, and share of the frame the base refiner runs on (tiles for `tiles`, the
    band itself for `pixels`, 1 for `frame`); context is not counted."""
    band = uncertain_band(output, config)
    if config.crop != "tiles":
        return float(band.mean()), float(band.mean()) if config.crop == "pixels" else 1.0
    h, w = band.shape
    covered = np.zeros_like(band)
    for top, left in band_tiles(band, config.tile_px):
        covered[top:top + config.tile_px, left:left + config.tile_px] = True
    return float(band.mean()), float(covered[:h, :w].mean())
