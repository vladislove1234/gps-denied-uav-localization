"""Turn the stored mosaics into training crops.

The build stores whole 1284 x 988 m mosaics, not pre-cut tiles, because cutting
at build time bakes in a window size and multiplies the bytes. This module is
the other half: it enumerates crops lazily so the window can change without
rebuilding a byte.

    from crops import CropIndex
    index = CropIndex(size=256, stride=128, min_labelled=0.02)
    print(len(index))                      # (location, date, window) triples
    image, label = index.load(0)

A crop of 256 px at z18 is 99 m on the ground, which is the footprint of the
2025 camera at about 100 m altitude -- so one crop is roughly one drone frame.

Pairs for an invariance objective come from ``index.pairs()``: the same window
on two different dates shares a label mask and, after the build's
co-registration, the same pixels.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

SRC = Path(__file__).resolve().parents[2]
ROOT = SRC / "data/derived/ua-village-multidate"

# osm.CLASSES: background 0, building 1, road 2, water 3, wood 4.
# Buildings and roads are the classes whose *shape* is informative at 99 m.
# Wood and water are region classes -- kept in the label, not used to select.
STRUCTURAL = (1, 2)


def _class_edges(label: np.ndarray) -> np.ndarray:
    """Pixels where the class changes -- the part of a label that can localise."""
    out = np.zeros(label.shape, dtype=bool)
    out[:-1, :] |= label[:-1, :] != label[1:, :]
    out[:, :-1] |= label[:, :-1] != label[:, 1:]
    return out


@dataclass(frozen=True)
class Crop:
    location: str
    date: str
    row: int
    col: int


class CropIndex:
    def __init__(self, size: int = 256, stride: int = 128,
                 min_edge: float = 0.006, min_structure: float = 0.01,
                 root: Path = ROOT,
                 max_label_offset_m: float | None = 4.0,
                 min_label_peak_z: float | None = 4.0):
        """Windows are selected by *boundary* content, not by labelled area.

        Selecting on labelled fraction looks reasonable and is wrong here. The
        classes that cover the most ground are ``wood`` and ``water``, and a 99 m
        crop inside a forest polygon is a uniform green mask: it passes a
        "2% labelled" test at 89% labelled while carrying no structure to
        correlate against. The earlier PoC reached the same conclusion about
        farmland -- segment boundaries, not regions -- and the same trap is
        waiting one level down in the crop sampler.

        So a window must have ``min_edge`` of its pixels on a class boundary and
        ``min_structure`` of them in a structural class (building or road).

        ``max_label_offset_m`` / ``min_label_peak_z`` drop bad locations.

        The build measures, per location, how far OSM sits from that location's
        own imagery. It is not uniform: the median is 2.6 m but a quarter of
        locations are worse than 5 m, which at 0.386 m/px is 13 px of label
        error on a 256 px crop. Excluding them here rather than at build time is
        deliberate -- the bytes are fine and the threshold is a training
        decision, so it belongs to whoever is training. Pass ``None`` to keep
        everything.
        """
        self.size, self.stride, self.root = size, stride, Path(root)
        self.min_edge, self.min_structure = min_edge, min_structure
        index = json.loads((self.root / "index.json").read_text())
        candidates = index["built"]
        self.meta, self.locations, self.excluded = {}, [], []
        for key in candidates:
            meta = json.loads((self.root / key / "meta.json").read_text())
            offset = meta["label_offset"]
            if max_label_offset_m is not None and offset["distance_m"] > max_label_offset_m:
                self.excluded.append((key, f"label offset {offset['distance_m']:.1f} m"))
                continue
            if min_label_peak_z is not None and offset["peak_z"] < min_label_peak_z:
                self.excluded.append((key, f"label peak_z {offset['peak_z']:.1f}"))
                continue
            self.meta[key] = meta
            self.locations.append(key)
        self._labels: dict[str, np.ndarray] = {}
        self.crops: list[Crop] = []
        self.windows: dict[str, list[tuple[int, int]]] = {}

        for key in self.locations:
            meta = self.meta[key]
            label = self.label(key)
            edges = _class_edges(label)
            structure = np.isin(label, STRUCTURAL)
            inset = meta.get("inset_px", 4)
            height, width = label.shape
            wins = []
            for r in range(inset, height - size - inset + 1, stride):
                for c in range(inset, width - size - inset + 1, stride):
                    if edges[r:r + size, c:c + size].mean() < min_edge:
                        continue
                    if structure[r:r + size, c:c + size].mean() < min_structure:
                        continue
                    wins.append((r, c))
            self.windows[key] = wins
            for date in meta["dates"]:
                self.crops.extend(Crop(key, date, r, c) for r, c in wins)

    def label(self, key: str) -> np.ndarray:
        if key not in self._labels:
            self._labels[key] = np.array(Image.open(self.root / key / "label.png"))
        return self._labels[key]

    def __len__(self) -> int:
        return len(self.crops)

    def load(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        crop = self.crops[i]
        s = self.size
        image = np.array(Image.open(self.root / crop.location / f"{crop.date}.jpg")
                         .crop((crop.col, crop.row, crop.col + s, crop.row + s)))
        label = self.label(crop.location)[crop.row:crop.row + s, crop.col:crop.col + s]
        return image, label

    def pairs(self) -> list[tuple[Crop, Crop]]:
        """Every same-window, different-date pair. The invariance training signal."""
        out = []
        for key in self.locations:
            dates = sorted(self.meta[key]["dates"])
            for r, c in self.windows[key]:
                for i in range(len(dates)):
                    for j in range(i + 1, len(dates)):
                        out.append((Crop(key, dates[i], r, c), Crop(key, dates[j], r, c)))
        return out

    def split(self, holdout: float = 0.2) -> tuple[list[str], list[str]]:
        """Split **by location**, never by crop.

        Crops from one village overlap and share imagery; splitting across them
        leaks the test set into training and reports a number that is not real.
        """
        n = max(1, int(round(len(self.locations) * holdout)))
        ordered = sorted(self.locations)
        return ordered[n:], ordered[:n]


if __name__ == "__main__":
    index = CropIndex()
    print(f"{len(index.locations)} locations kept, {len(index.excluded)} excluded, "
          f"{len(index)} crops, {len(index.pairs())} same-window date pairs")
    for key, why in index.excluded[:5]:
        print(f"  excluded {key:38} {why}")
    train, test = index.split()
    print(f"split by location: {len(train)} train / {len(test)} test")
    image, label = index.load(0)
    print(f"crop 0: image {image.shape} {image.dtype}, label {label.shape} "
          f"classes {sorted(np.unique(label).tolist())}")
