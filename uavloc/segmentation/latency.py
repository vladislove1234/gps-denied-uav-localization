"""CPU latency of the model and of refiners, on a standard 512 x 512 satellite tile.

This machine's i7-9750H stands in for the Raspberry-Pi-class target: threads are capped at the Pi's 4 cores,
but absolute times are not Pi times. Compare refiners against each other and against the model, not to a budget.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np
import torch
from torch import nn

from uavloc.features.segmentation import LRASPPLogits, class_probabilities
from uavloc.segmentation.eval import ModelOutput, RefineFn, lraspp_outputs

STANDARD_TILE_PX = 512
PI_CLASS_THREADS = 4
WARMUP_RUNS = 5
TIMED_RUNS = 30
MS_PER_S = 1000.0
LOW_PERCENTILE, HIGH_PERCENTILE = 10, 90


@dataclass(frozen=True)
class Latency:
    median_ms: float
    p10_ms: float
    p90_ms: float
    runs: int
    threads: int


@contextmanager
def cpu_threads(n: int) -> Iterator[None]:
    """Cap torch and OpenCV at `n` threads for the duration."""
    torch_before, cv_before = torch.get_num_threads(), cv2.getNumThreads()
    torch.set_num_threads(n)
    cv2.setNumThreads(n)
    try:
        yield
    finally:
        torch.set_num_threads(torch_before)
        cv2.setNumThreads(cv_before)


@torch.no_grad()
def cpu_latency(fn: Callable[[], object], warmup: int = WARMUP_RUNS, runs: int = TIMED_RUNS,
                threads: int = PI_CLASS_THREADS) -> Latency:
    """Median (and p10/p90) wall time of `fn()` over `runs` calls after `warmup` untimed ones."""
    with cpu_threads(threads):
        for _ in range(warmup):
            fn()
        times = []
        for _ in range(runs):
            start = time.perf_counter()
            fn()
            times.append((time.perf_counter() - start) * MS_PER_S)
    low, median, high = np.percentile(times, [LOW_PERCENTILE, 50, HIGH_PERCENTILE])
    return Latency(float(median), float(low), float(high), runs, threads)


def standard_tile(rgb: np.ndarray, top: int, left: int, size_px: int = STANDARD_TILE_PX) -> np.ndarray:
    """A `size_px` square of real imagery, so content-dependent refiners are timed on real content."""
    tile = rgb[top:top + size_px, left:left + size_px]
    if tile.shape[:2] != (size_px, size_px):
        raise ValueError(f"tile at ({top}, {left}) runs off a {rgb.shape[:2]} image")
    return np.ascontiguousarray(tile)


def model_latency(model: LRASPPLogits, tile: np.ndarray, **kwargs) -> Latency:
    """Model on CPU, input normalisation to full-res probabilities (`lraspp_outputs`)."""
    model = copy.deepcopy(model).cpu().eval()
    return cpu_latency(lambda: lraspp_outputs(model, tile, "cpu"), **kwargs)


def refiner_latency(refine_fn: RefineFn, output: ModelOutput, tile: np.ndarray, **kwargs) -> Latency:
    """The refiner alone, given the model's output for `tile` (compute it once with `lraspp_outputs`)."""
    return cpu_latency(lambda: refine_fn(output, tile), **kwargs)


def segmenter_latency(model: nn.Module, tile: np.ndarray, **kwargs) -> Latency:
    """Any segmenter on CPU, input normalisation to full-res probabilities (`class_probabilities`)."""
    model = copy.deepcopy(model).cpu().eval()
    return cpu_latency(lambda: class_probabilities(model, tile, "cpu"), **kwargs)
