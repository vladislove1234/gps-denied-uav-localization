"""Sub-pixel translation between two co-located images of the same ground.

Moved out of ``datasets/ua-village-multidate/build.py`` on its second use: the
Google imagery PoC co-registers against the same Esri reference the build does,
and the two must measure shift the same way to be comparable.
"""

from __future__ import annotations

import numpy as np

DEFAULT_MAX_SHIFT_PX = 48
MIN_APPLIED_SHIFT_PX = 0.05


def phase_shift(reference: np.ndarray, moving: np.ndarray,
                max_shift: int = DEFAULT_MAX_SHIFT_PX) -> tuple[float, float, float]:
    """Translation `(dx, dy, peak_ratio)` taking greyscale `moving` onto `reference`, sub-pixel."""
    fa = np.fft.rfft2(reference - reference.mean())
    fb = np.fft.rfft2(moving - moving.mean())
    cross = fa * np.conj(fb)
    magnitude = np.abs(cross)
    cross = np.divide(cross, magnitude, out=np.zeros_like(cross), where=magnitude > 1e-9)
    surface = np.fft.fftshift(np.fft.irfft2(cross, s=reference.shape))
    cy, cx = surface.shape[0] // 2, surface.shape[1] // 2
    w = surface[cy - max_shift:cy + max_shift + 1, cx - max_shift:cx + max_shift + 1]
    py, px = np.unravel_index(np.argmax(w), w.shape)

    def refine(c, l, r):
        den = l - 2 * c + r
        return 0.0 if den == 0 else 0.5 * (l - r) / den

    sy = refine(w[py, px], w[py - 1, px], w[py + 1, px]) if 0 < py < w.shape[0] - 1 else 0.0
    sx = refine(w[py, px], w[py, px - 1], w[py, px + 1]) if 0 < px < w.shape[1] - 1 else 0.0
    ratio = float(w.max() / (np.abs(w).mean() + 1e-12))
    return float(px + sx - max_shift), float(py + sy - max_shift), ratio


def apply_shift(rgb: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Move `rgb` (H, W, 3) by the shift `phase_shift` measured, so it lands on the reference."""
    from scipy.ndimage import shift as ndshift
    if abs(dx) < MIN_APPLIED_SHIFT_PX and abs(dy) < MIN_APPLIED_SHIFT_PX:
        return rgb
    out = np.empty_like(rgb)
    for c in range(3):
        out[..., c] = ndshift(rgb[..., c], (-dy, -dx), order=1, mode="nearest")
    return out
