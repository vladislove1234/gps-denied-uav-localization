"""Class probabilities from an exported `.onnx` segmenter, the same file the Raspberry Pi runs."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import onnxruntime as ort

from uavloc.features.imagenet import normalise
from uavloc.features.tiling import TILE_PX, TILE_STEP_PX, tiled_predict

MODEL_ID_HEX_CHARS = 12
HASH_CHUNK_BYTES = 1 << 20


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


class OnnxSegmenter:
    """Tiled like `features.segmentation.class_probabilities`, with the model's fixed input size as the tile."""

    def __init__(self, model_path: Path, threads: int | None = None) -> None:
        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
        self.model_path = model_path
        self.session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
        model_input = self.session.get_inputs()[0]
        self._input_name = model_input.name
        batch, _, height, width = model_input.shape
        self.batch = batch if isinstance(batch, int) else 1
        self.tile_px = height if isinstance(height, int) and height == width else TILE_PX
        self.tile_step_px = self.tile_px * TILE_STEP_PX // TILE_PX
        self.model_id = file_sha256(model_path)[:MODEL_ID_HEX_CHARS]
        self.n_classes = self.logits(np.zeros((self.batch, 3, self.tile_px, self.tile_px), np.float32)).shape[1]

    def logits(self, tiles: np.ndarray) -> np.ndarray:
        """`(B, 3, P, P)` normalised float32 → `(B, C, P, P)`."""
        return self.session.run(None, {self._input_name: tiles})[0]

    def class_probabilities(self, rgb: np.ndarray) -> np.ndarray:
        """`(H, W, 3)` uint8 → `(C, H, W)` float32 softmax probabilities."""
        x = normalise(rgb)
        logits = tiled_predict(self.logits, x, self.tile_px, self.tile_step_px, self.n_classes, self.batch)
        logits -= logits.max(0, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(0, keepdims=True)
