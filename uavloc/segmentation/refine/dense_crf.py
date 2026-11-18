"""Fully-connected CRF refinement of LR-ASPP masks (Krähenbühl & Koltun 2011), via `pydensecrf2`.

Unary: `-log p` of the model's full-res probabilities. Pairwise: an appearance (bilateral) kernel over
position and RGB, plus a smoothness (Gaussian) kernel over position, both with Potts compatibility.
Mean-field inference returns per-pixel marginals; `refine` argmaxes them into a label map.

`crf_marginals` is the building block for callers that want to restrict or blend the CRF (e.g. only in an
uncertain band); `refine` is the `RefineFn`-shaped whole-frame version. `crf_marginals_at` builds the CRF
over a pixel subset only, so its cost scales with the subset (used by `band.band_refiner` in pixel mode).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np
import pydensecrf.densecrf as dcrf

from uavloc.segmentation.eval import MaskedRefineFn, ModelOutput, RefineFn

# Krähenbühl & Koltun's published defaults (their Pascal VOC setting).
DEFAULT_ITERATIONS = 5
DEFAULT_BILATERAL_SXY_PX = 80.0
DEFAULT_BILATERAL_SRGB = 13.0
DEFAULT_BILATERAL_COMPAT = 10.0
DEFAULT_GAUSSIAN_SXY_PX = 3.0
DEFAULT_GAUSSIAN_COMPAT = 3.0
# Keeps -log p finite where the model is certain; 1e-5 caps a unary at ~11.5 nats.
PROB_FLOOR = 1e-5


@dataclass(frozen=True)
class DenseCrfConfig:
    """Mean-field iterations and the two Potts pairwise kernels (widths in px, colour width in 0-255 units)."""

    iterations: int = DEFAULT_ITERATIONS
    bilateral_sxy_px: float = DEFAULT_BILATERAL_SXY_PX
    bilateral_srgb: float = DEFAULT_BILATERAL_SRGB
    bilateral_compat: float = DEFAULT_BILATERAL_COMPAT
    gaussian_sxy_px: float = DEFAULT_GAUSSIAN_SXY_PX
    gaussian_compat: float = DEFAULT_GAUSSIAN_COMPAT
    prob_floor: float = PROB_FLOOR


def crf_marginals(probs: np.ndarray, rgb: np.ndarray, config: DenseCrfConfig) -> np.ndarray:
    """Mean-field marginals `(C, H, W)` float32 for class probabilities `(C, H, W)` and uint8 `rgb` `(H, W, 3)`."""
    n_classes, h, w = probs.shape
    if rgb.shape[:2] != (h, w):
        raise ValueError(f"probs {probs.shape} and rgb {rgb.shape} disagree on size")
    crf = dcrf.DenseCRF2D(w, h, n_classes)
    unary = -np.log(np.clip(probs, config.prob_floor, 1.0)).reshape(n_classes, -1)
    crf.setUnaryEnergy(np.ascontiguousarray(unary, dtype=np.float32))
    crf.addPairwiseGaussian(sxy=config.gaussian_sxy_px, compat=config.gaussian_compat)
    crf.addPairwiseBilateral(sxy=config.bilateral_sxy_px, srgb=config.bilateral_srgb,
                             rgbim=np.ascontiguousarray(rgb, dtype=np.uint8), compat=config.bilateral_compat)
    return np.asarray(crf.inference(config.iterations), dtype=np.float32).reshape(n_classes, h, w)


def refine(output: ModelOutput, rgb: np.ndarray, config: DenseCrfConfig) -> np.ndarray:
    """Whole-frame dense-CRF label map `(H, W)` uint8 from the model's full-res probabilities."""
    return crf_marginals(output.probs, rgb, config).argmax(0).astype(np.uint8)


def dense_crf_refiner(config: DenseCrfConfig) -> RefineFn:
    """`refine` with `config` bound, for `evaluate` and `refiner_latency`."""
    return partial(refine, config=config)


def crf_marginals_at(probs: np.ndarray, rgb: np.ndarray, mask: np.ndarray, config: DenseCrfConfig) -> np.ndarray:
    """Mean-field marginals `(C, N)` of a dense CRF whose nodes are only the N `mask` pixels (row-major order).

    Same unary and kernels as `crf_marginals`; pixels outside `mask` are not in the graph at all, so the cost
    scales with N. On an all-true mask it matches `crf_marginals`.
    """
    n_classes = probs.shape[0]
    rows, cols = np.nonzero(mask)
    crf = dcrf.DenseCRF(rows.size, n_classes)
    unary = -np.log(np.clip(probs[:, rows, cols], config.prob_floor, 1.0))
    crf.setUnaryEnergy(np.ascontiguousarray(unary, dtype=np.float32))
    position = np.stack([cols, rows]).astype(np.float32)
    crf.addPairwiseEnergy(np.ascontiguousarray(position / config.gaussian_sxy_px), compat=config.gaussian_compat)
    colour = rgb[rows, cols].T.astype(np.float32) / config.bilateral_srgb
    bilateral = np.vstack([position / config.bilateral_sxy_px, colour])
    crf.addPairwiseEnergy(np.ascontiguousarray(bilateral, dtype=np.float32), compat=config.bilateral_compat)
    return np.asarray(crf.inference(config.iterations), dtype=np.float32).reshape(n_classes, rows.size)


def refine_at(output: ModelOutput, rgb: np.ndarray, mask: np.ndarray, config: DenseCrfConfig) -> np.ndarray:
    """`(N,)` uint8 dense-CRF classes of the `mask` pixels, from a CRF over those pixels only."""
    if not mask.any():
        return np.zeros(0, np.uint8)
    return crf_marginals_at(output.probs, rgb, mask, config).argmax(0).astype(np.uint8)


def masked_dense_crf_refiner(config: DenseCrfConfig) -> MaskedRefineFn:
    """`refine_at` with `config` bound: a `(output, rgb, mask)` refiner for `band.band_refiner` in pixel mode."""
    return partial(refine_at, config=config)
