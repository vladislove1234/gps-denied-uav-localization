"""How far is OSM from the imagery it is supposed to label?

OSM geometry and Google Earth imagery are digitised from different basemaps, so
they can sit several metres apart. At the 5-10 m scale this project measures
position error at, that is not a detail -- it is label noise large enough to
swamp the signal, and it has to be measured before anything is trained on it.

Method: buildings have the strongest edges in the scene, so we correlate the
OSM building outlines against the image gradient and find the shift that lines
them up. The peak offset is the systematic georeferencing disagreement; whether
it is OSM that is wrong or the orthophoto's corner coordinates is a question
this cannot answer on its own -- it needs a second, independently georeferenced
copy of the same ground.
"""

from __future__ import annotations

import numpy as np


def gradient_magnitude(grey: np.ndarray) -> np.ndarray:
    """Sobel edge magnitude, no SciPy dependency."""
    grey = grey.astype(np.float32)
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = kx.T

    def convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        padded = np.pad(image, 1, mode="edge")
        out = np.zeros_like(image)
        for i in range(3):
            for j in range(3):
                weight = kernel[i, j]
                if weight:
                    out += weight * padded[i : i + image.shape[0], j : j + image.shape[1]]
        return out

    return np.hypot(convolve(grey, kx), convolve(grey, ky))


def _whiten(array: np.ndarray) -> np.ndarray:
    array = array.astype(np.float32)
    array -= array.mean()
    std = array.std()
    return array / std if std > 0 else array


def binarise(edges: np.ndarray, percentile: float = 88.0) -> np.ndarray:
    """Keep only the strongest gradients.

    Correlating raw gradient magnitude does not work here: the surface is
    dominated by broad low-frequency agreement -- "this half of the image is
    busier than that half" -- and the true peak never rises above it. The first
    attempt put its global maximum 217 m away, which is not a georeferencing
    error, it is noise. Thresholding to a sparse binary edge map removes the
    low-frequency term and leaves only structure.
    """
    return (edges > np.percentile(edges, percentile)).astype(np.float32)


def crop_to(mask: np.ndarray, margin: int) -> tuple[slice, slice]:
    """Bounding box of the nonzero part of ``mask``, padded by ``margin``."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return slice(None), slice(None)
    return (
        slice(max(0, ys.min() - margin), min(mask.shape[0], ys.max() + margin)),
        slice(max(0, xs.min() - margin), min(mask.shape[1], xs.max() + margin)),
    )


def offset(
    image_edges: np.ndarray,
    label_edges: np.ndarray,
    mpp: float,
    max_shift_m: float = 15.0,
    percentile: float = 88.0,
) -> dict:
    """Shift that best aligns ``label_edges`` onto ``image_edges``.

    Both layers are reduced to sparse binary edges and cropped to the region the
    labels actually cover, then correlated. The reported ``peak_z`` is measured
    **within the search window** -- comparing against the whole surface was the
    bug in the first version, and it produced a negative z for a maximum.

    A ``peak_z`` below about 4 means the two layers share no structure, and the
    offset is then meaningless however precise it looks.
    """
    from scipy.signal import fftconvolve

    rows, cols = crop_to(label_edges, margin=int(round(2 * max_shift_m / mpp)))
    a = _whiten(binarise(image_edges[rows, cols], percentile))
    b = _whiten((label_edges[rows, cols] > 0).astype(np.float32))

    correlation = fftconvolve(a, b[::-1, ::-1], mode="same")
    height, width = correlation.shape
    centre_y, centre_x = height // 2, width // 2
    radius = int(round(max_shift_m / mpp))
    window = correlation[
        max(0, centre_y - radius) : centre_y + radius + 1,
        max(0, centre_x - radius) : centre_x + radius + 1,
    ]

    peak_y, peak_x = np.unravel_index(np.argmax(window), window.shape)
    dy, dx = peak_y - radius, peak_x - radius
    return {
        "dx_px": int(dx),
        "dy_px": int(dy),
        "dx_m": float(dx * mpp),
        "dy_m": float(dy * mpp),
        "distance_m": float(np.hypot(dx, dy) * mpp),
        "peak_z": float((window.max() - window.mean()) / window.std()),
        "zero_shift_z": float((correlation[centre_y, centre_x] - window.mean()) / window.std()),
        "at_search_limit": bool(abs(dx) >= radius - 1 or abs(dy) >= radius - 1),
    }


def outline(mask: np.ndarray) -> np.ndarray:
    """1-pixel outline of a binary mask, by 4-neighbour difference."""
    mask = (mask > 0).astype(np.float32)
    padded = np.pad(mask, 1, mode="edge")
    neighbours = (
        padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]
    )
    return ((mask > 0) & (neighbours < 4)).astype(np.float32)
