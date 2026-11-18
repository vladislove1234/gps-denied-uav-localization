"""The satellite class map of a planning area, segmented on demand and cached per block of tiles."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Protocol

import numpy as np
from tqdm.auto import tqdm

from uavloc.data.satellite import GeoImage
from uavloc.data.tiles import TILE_PX, TileRange, TileSource, mercator_to_geoimage, mosaic_px
from uavloc.geo import GeoBounds

BLOCK_TILES = 4          # a block is 4 x 4 tiles, 1024 px: the unit that is segmented and cached
CONTEXT_TILES = 1        # tiles of context on each side, so block edges are segmented like their interior
PROBABILITY_LEVELS = 255

ProgressHook = Callable[[str, int, int], None]   # (stage, done, total)


class Segmenter(Protocol):
    model_id: str

    def class_probabilities(self, rgb: np.ndarray) -> np.ndarray: ...


def block_range(bounds: GeoBounds, zoom: int) -> TileRange:
    """The tiles covering `bounds`, grown to whole blocks."""
    tiles = TileRange.covering(bounds, zoom)
    bx0, by0 = tiles.x0 // BLOCK_TILES, tiles.y0 // BLOCK_TILES
    bx1, by1 = -(-(tiles.x0 + tiles.nx) // BLOCK_TILES), -(-(tiles.y0 + tiles.ny) // BLOCK_TILES)
    return TileRange(zoom, bx0 * BLOCK_TILES, by0 * BLOCK_TILES, (bx1 - bx0) * BLOCK_TILES, (by1 - by0) * BLOCK_TILES)


def block_probabilities(source: TileSource, segmenter: Segmenter, zoom: int, block_x: int, block_y: int,
                        tile_cache: Path, class_cache: Path) -> np.ndarray:
    """`(1024, 1024, C)` uint8 probabilities (x 255) of one block, segmented once and then read from disk."""
    path = class_cache / f"{source.key}-{segmenter.model_id}" / str(zoom) / f"{block_x}_{block_y}.npz"
    if path.exists():
        return np.load(path)["probabilities"]
    context = TileRange(zoom, block_x * BLOCK_TILES - CONTEXT_TILES, block_y * BLOCK_TILES - CONTEXT_TILES,
                        BLOCK_TILES + 2 * CONTEXT_TILES, BLOCK_TILES + 2 * CONTEXT_TILES)
    probabilities = segmenter.class_probabilities(mosaic_px(source, context, tile_cache))
    inner = slice(CONTEXT_TILES * TILE_PX, (CONTEXT_TILES + BLOCK_TILES) * TILE_PX)
    quantised = np.round(probabilities[:, inner, inner] * PROBABILITY_LEVELS).astype(np.uint8)
    quantised = np.ascontiguousarray(quantised.transpose(1, 2, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, probabilities=quantised)
    return quantised


def corridor_class_map(bounds: GeoBounds, zoom: int, source: TileSource, segmenter: Segmenter,
                       tile_cache: Path, class_cache: Path, progress: ProgressHook | None = None) -> GeoImage:
    """`(H, W, C)` uint8 class probabilities (x 255) over whole blocks covering `bounds`, rows linear in latitude."""
    blocks = block_range(bounds, zoom)
    block_px = BLOCK_TILES * TILE_PX
    n_bx, n_by = blocks.nx // BLOCK_TILES, blocks.ny // BLOCK_TILES
    out: np.ndarray | None = None
    coords = [(bx, by) for by in range(n_by) for bx in range(n_bx)]
    for done, (bx, by) in enumerate(tqdm(coords, desc="class map blocks")):
        block = block_probabilities(source, segmenter, zoom, blocks.x0 // BLOCK_TILES + bx,
                                    blocks.y0 // BLOCK_TILES + by, tile_cache, class_cache)
        if out is None:
            out = np.zeros((n_by * block_px, n_bx * block_px, block.shape[2]), np.uint8)
        out[by * block_px:(by + 1) * block_px, bx * block_px:(bx + 1) * block_px] = block
        if progress:
            progress("class map", done + 1, len(coords))
    return mercator_to_geoimage(out, blocks)
