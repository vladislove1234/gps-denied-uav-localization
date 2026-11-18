"""Esri World Imagery Wayback: the same tile, as it looked on 196 past dates.

This is the reason the PoC is worth running. Wayback serves archived releases of
World Imagery under the *same* XYZ scheme, so two dates of the same tile are
co-registered by construction -- no fitting, no control points, no residual.
That turns "the same ground photographed years apart in a different season by a
different sensor" from a data-collection problem into a URL change.

The catch, and what has to be measured: a Wayback *release* is global, but rural
ground only changes when Esri actually refreshed imagery there. Most of the 196
releases serve byte-identical tiles for a given village. ``distinct_releases``
finds the ones that are genuinely different by hashing a few probe tiles.
"""

from __future__ import annotations

import json
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tiles import TileSource, tile_hash, USER_AGENT

CONFIG_URL = "https://s3-us-west-2.amazonaws.com/config.maptiles.arcgis.com/waybackconfig.json"


def releases(cache_dir: Path) -> list[tuple[str, TileSource]]:
    """Every Wayback release as ``(date, TileSource)``, oldest first."""
    path = cache_dir / "waybackconfig.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(CONFIG_URL, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=60) as response:
            path.write_bytes(response.read())
    config = json.loads(path.read_text())

    out = []
    for item in config.values():
        match = re.search(r"(\d{4}-\d{2}-\d{2})", item["itemTitle"])
        if not match:
            continue
        out.append((match.group(1), TileSource(f"wayback-{match.group(1)}", item["itemURL"])))
    out.sort(key=lambda pair: pair[0])
    return out


def distinct_releases(grid, cache_dir: Path, every: int = 1,
                      probes: int = 3, workers: int = 8) -> list[tuple[str, TileSource]]:
    """The releases whose imagery over ``grid`` actually differs.

    Hashes ``probes`` tiles spread across the grid rather than one, because a
    single tile can be unchanged while the rest of the area was reflown. Keeps
    the *first* release showing each distinct combination -- the date the
    imagery was published, not the date it was superseded.
    """
    all_releases = releases(cache_dir)[::every]
    coords = _probe_tiles(grid, probes)

    def signature(entry):
        date, source = entry
        try:
            return date, source, "|".join(
                tile_hash(source, grid.z, tx, ty, cache_dir) for tx, ty in coords)
        except Exception as exc:
            return date, source, f"error:{type(exc).__name__}"

    with ThreadPoolExecutor(workers) as pool:
        signed = list(pool.map(signature, all_releases))

    seen, keep = set(), []
    for date, source, sig in signed:
        if sig.startswith("error:") or sig == "|".join(["missing"] * len(coords)):
            continue
        if sig in seen:
            continue
        seen.add(sig)
        keep.append((date, source))
    return keep


def _probe_tiles(grid, probes: int) -> list[tuple[int, int]]:
    """A few well-spread tiles from the grid."""
    picks = [(grid.x0 + grid.nx // 2, grid.y0 + grid.ny // 2)]
    if probes > 1:
        picks.append((grid.x0 + grid.nx // 4, grid.y0 + grid.ny // 4))
    if probes > 2:
        picks.append((grid.x0 + 3 * grid.nx // 4, grid.y0 + 3 * grid.ny // 4))
    return picks[:probes]
