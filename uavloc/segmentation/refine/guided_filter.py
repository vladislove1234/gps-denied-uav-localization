"""Guided-filter refiners: class probabilities filtered with the RGB image as guide (He et al. 2013).

Two plug-in points of the harness (`uavloc.segmentation.eval.RefineFn`):
- `full_res`: the guided filter on the model's full-res bilinear probabilities, radius in full-res px.
- `fast`: joint guided upsampling from the stride-8 grid (Fast Guided Filter, He & Sun 2015). The linear
  coefficients are fitted between the cell-averaged guide and the stride-8 softmax, bilinearly upsampled
  like the model's own logits, and applied to the full-res guide. Radius is in stride-8 cells.

`refine_at` evaluates the fast mode only at a pixel subset (for `band.band_refiner` in pixel mode).

Guide is RGB (3x3 covariance per window) or grey, scaled to [0, 1], so `eps` is in (intensity)^2 units.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

from uavloc.segmentation.eval import STRIDE, MaskedRefineFn, ModelOutput, RefineFn, upsample_logits

PIXEL_MAX = 255.0
GUIDES = ("rgb", "grey")
MODES = ("full_res", "fast")


@dataclass(frozen=True)
class GuidedFilterConfig:
    mode: str = "full_res"   # "full_res" | "fast"
    guide: str = "rgb"       # "rgb" | "grey"
    radius: int = 4          # full-res px ("full_res") or stride-8 cells ("fast")
    eps: float = 1e-2


def box(x: np.ndarray, radius: int) -> np.ndarray:
    """Mean over a `(2r + 1)^2` window, reflected at the border, of each channel of `(H, W, ...)`."""
    size = (2 * radius + 1, 2 * radius + 1)
    flat = np.ascontiguousarray(x.reshape(*x.shape[:2], -1))
    out = cv2.boxFilter(flat, -1, size, borderType=cv2.BORDER_REFLECT)
    return out.reshape(x.shape)


def guide_image(rgb: np.ndarray, guide: str) -> np.ndarray:
    """`(H, W, 3)` uint8 → float32 guide in [0, 1]: `(H, W, 3)` for rgb, `(H, W)` for grey."""
    if guide == "grey":
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / PIXEL_MAX
    if guide == "rgb":
        return rgb.astype(np.float32) / PIXEL_MAX
    raise ValueError(f"guide must be one of {GUIDES}, got {guide!r}")


def grey_coefficients(guide: np.ndarray, p: np.ndarray, radius: int, eps: float) -> tuple[np.ndarray, np.ndarray]:
    """Window-mean `(a, b)` of `q = a I + b` for a grey guide `(H, W)` and inputs `(H, W, C)`."""
    mean_i = box(guide, radius)[..., None]
    mean_p = box(p, radius)
    var_i = box(guide * guide, radius)[..., None] - mean_i ** 2
    a = (box(guide[..., None] * p, radius) - mean_i * mean_p) / (var_i + eps)
    b = mean_p - a * mean_i
    return box(a, radius), box(b, radius)


def symmetric_inverse(s: list[list[np.ndarray]]) -> list[list[np.ndarray]]:
    """Inverse of per-pixel symmetric 3x3 matrices given as a nested list of planes, by cofactors."""
    a, b, c, e, f, i = s[0][0], s[0][1], s[0][2], s[1][1], s[1][2], s[2][2]
    c00, c01, c02 = e * i - f * f, c * f - b * i, b * f - c * e
    c11, c12, c22 = a * i - c * c, b * c - a * f, a * e - b * b
    det = a * c00 + b * c01 + c * c02
    return [[c00 / det, c01 / det, c02 / det], [c01 / det, c11 / det, c12 / det], [c02 / det, c12 / det, c22 / det]]


def rgb_coefficients(guide: np.ndarray, p: np.ndarray, radius: int, eps: float) -> tuple[np.ndarray, np.ndarray]:
    """Window-mean `(a, b)` of `q = a . I + b` for an RGB guide `(H, W, 3)`; `a` is `(H, W, C, 3)`, `b` `(H, W, C)`.

    Planes are kept separate (3 guide channels x C classes) so every step is an elementwise op on `(H, W, C)`.
    """
    mean_i, mean_p = box(guide, radius), box(p, radius)
    pairs = np.stack([guide[..., k] * guide[..., j] for k in range(3) for j in range(k, 3)], axis=-1)
    second = box(pairs, radius)
    upper = {(k, j): n for n, (k, j) in enumerate((k, j) for k in range(3) for j in range(k, 3))}
    cov_ii = [[second[..., upper[min(k, j), max(k, j)]] - mean_i[..., k] * mean_i[..., j] + eps * (k == j)
               for j in range(3)] for k in range(3)]
    inv = [[plane[..., None] for plane in row] for row in symmetric_inverse(cov_ii)]
    cross = box(guide[..., :, None] * p[..., None, :], radius)   # (H, W, 3, C)
    cov_ip = [cross[..., k, :] - mean_i[..., k, None] * mean_p for k in range(3)]
    a = [sum(cov_ip[k] * inv[k][j] for k in range(3)) for j in range(3)]
    b = mean_p - sum(a[j] * mean_i[..., j, None] for j in range(3))
    return box(np.stack(a, axis=-1), radius), box(b, radius)


def coefficients(guide: np.ndarray, p: np.ndarray, radius: int, eps: float) -> tuple[np.ndarray, np.ndarray]:
    fit = rgb_coefficients if guide.ndim == 3 else grey_coefficients
    return fit(guide, p, radius, eps)


def apply_coefficients(a: np.ndarray, b: np.ndarray, guide: np.ndarray) -> np.ndarray:
    """`q = a . I + b`, `(H, W, C)`."""
    if guide.ndim == 3:
        return sum(a[..., j] * guide[..., j, None] for j in range(3)) + b
    return a * guide[..., None] + b


def guided_filter(guide: np.ndarray, p: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """He et al. guided filter of `p` `(H, W, C)` with `guide` `(H, W[, 3])`, both float32."""
    a, b = coefficients(guide, p, radius, eps)
    return apply_coefficients(a, b, guide)


def cell_means(x: np.ndarray, cells: tuple[int, int]) -> np.ndarray:
    """`(H, W[, 3])` reflect-padded to `cells * STRIDE` and averaged per `STRIDE x STRIDE` cell."""
    full = (cells[0] * STRIDE, cells[1] * STRIDE)
    pad = ((0, full[0] - x.shape[0]), (0, full[1] - x.shape[1])) + ((0, 0),) * (x.ndim - 2)
    padded = np.pad(x, pad, mode="reflect")
    return padded.reshape(cells[0], STRIDE, cells[1], STRIDE, *x.shape[2:]).mean(axis=(1, 3))


def softmax(logits: np.ndarray, axis: int = 0) -> np.ndarray:
    z = np.exp(logits - logits.max(axis, keepdims=True))
    return z / z.sum(axis, keepdims=True)


def upsample_channels_last(x: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Stride-8 `(h, w, ...)` → `(H, W, ...)` with the model's own bilinear x8 upsample."""
    flat = x.reshape(*x.shape[:2], -1).transpose(2, 0, 1)
    up = upsample_logits(np.ascontiguousarray(flat), size).transpose(1, 2, 0)
    return up.reshape(*size, *x.shape[2:])


def fast_guided_upsample(guide: np.ndarray, logits: np.ndarray, radius_cells: int, eps: float) -> np.ndarray:
    """Joint guided upsampling of stride-8 `(C, h, w)` logits to the guide's `(H, W, C)` probabilities."""
    low_guide = cell_means(guide, logits.shape[1:]).astype(np.float32)
    low_probs = softmax(logits).transpose(1, 2, 0).astype(np.float32)
    a, b = coefficients(low_guide, low_probs, radius_cells, eps)
    size = guide.shape[:2]
    return apply_coefficients(upsample_channels_last(a, size), upsample_channels_last(b, size), guide)


def refine(output: ModelOutput, rgb: np.ndarray, config: GuidedFilterConfig) -> np.ndarray:
    """Guided-filtered class probabilities, argmaxed to an `(H, W)` uint8 label map."""
    guide = guide_image(rgb, config.guide)
    if config.mode == "full_res":
        probs = guided_filter(guide, output.probs.transpose(1, 2, 0).astype(np.float32), config.radius, config.eps)
    elif config.mode == "fast":
        probs = fast_guided_upsample(guide, output.logits, config.radius, config.eps)
    else:
        raise ValueError(f"mode must be one of {MODES}, got {config.mode!r}")
    return probs.argmax(-1).astype(np.uint8)


def refiner(config: GuidedFilterConfig) -> RefineFn:
    """`refine` bound to `config`, as a harness `RefineFn`."""
    return lambda output, rgb: refine(output, rgb, config)


def bilinear_at(x: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> torch.Tensor:
    """Stride-8 `(h, w, K)` sampled at full-res pixels `(rows, cols)` exactly as the model's bilinear x8
    upsample (`align_corners=False`) would, without upsampling the whole map. Returns `(N, K)`.
    Torch rather than numpy because its gathers and lerps use every allowed thread."""
    def taps(index: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
        source = np.maximum((index + 0.5) / STRIDE - 0.5, 0.0)
        low = np.minimum(np.floor(source).astype(np.int64), size - 1)
        return low, np.minimum(low + 1, size - 1), torch.from_numpy((source - low).astype(np.float32)[:, None])
    h, w = x.shape[:2]
    flat = torch.from_numpy(np.ascontiguousarray(x.reshape(h * w, -1)))
    r0, r1, wr = taps(rows, h)
    c0, c1, wc = taps(cols, w)
    def at(r: np.ndarray, c: np.ndarray) -> torch.Tensor:
        return flat.index_select(0, torch.from_numpy(r * w + c))
    return torch.lerp(torch.lerp(at(r0, c0), at(r0, c1), wc), torch.lerp(at(r1, c0), at(r1, c1), wc), wr)


def refine_at(output: ModelOutput, rgb: np.ndarray, mask: np.ndarray, config: GuidedFilterConfig) -> np.ndarray:
    """`(N,)` uint8 classes of the `mask` pixels from fast-mode guided upsampling evaluated at those pixels only."""
    if config.mode != "fast":
        raise ValueError(f"per-pixel evaluation needs mode 'fast', got {config.mode!r}")
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return np.zeros(0, np.uint8)
    guide = guide_image(rgb, config.guide)
    low_guide = cell_means(guide, output.logits.shape[1:]).astype(np.float32)
    low_probs = softmax(output.logits).transpose(1, 2, 0).astype(np.float32)
    a, b = coefficients(low_guide, low_probs, config.radius, config.eps)
    n_classes = b.shape[-1]
    ab = bilinear_at(np.concatenate([a.reshape(*a.shape[:2], -1), b], axis=-1).astype(np.float32), rows, cols)
    guide_at = torch.from_numpy(np.ascontiguousarray(guide[rows, cols].reshape(rows.size, 1, -1)))
    a_at = ab[:, :-n_classes].reshape(rows.size, n_classes, guide_at.shape[-1])
    return ((a_at * guide_at).sum(-1) + ab[:, -n_classes:]).argmax(-1).numpy().astype(np.uint8)


def masked_refiner(config: GuidedFilterConfig) -> MaskedRefineFn:
    """`refine_at` bound to `config`: a `(output, rgb, mask)` refiner for `band.band_refiner` in pixel mode."""
    return lambda output, rgb, mask: refine_at(output, rgb, mask, config)
