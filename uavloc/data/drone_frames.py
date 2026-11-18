"""Drone frames with their logged pose, from a `data/derived/<dataset>/<frame>/` folder."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image

from uavloc.geo import LatLon
from uavloc.track import REPO_ROOT

DERIVED = REPO_ROOT / "data" / "derived"


@dataclass(frozen=True)
class DroneFrame:
    name: str
    image_path: Path
    position: LatLon
    altitude_m: float
    yaw_deg: float
    mpp: float
    picked_from: tuple[str, ...]

    def image(self) -> Image.Image:
        return Image.open(self.image_path).convert("RGB")


def load_frames(dataset: str) -> list[DroneFrame]:
    """Every frame of `dataset`, ordered by timestamp."""
    frames = []
    for meta_path in sorted((DERIVED / dataset).glob("*/meta.json")):
        meta = json.loads(meta_path.read_text())
        frames.append(DroneFrame(
            name=meta_path.parent.name,
            image_path=meta_path.parent / "frame.jpg",
            position=LatLon(meta["lat"], meta["lon"]),
            altitude_m=meta["alt_m"],
            yaw_deg=meta["yaw_deg"],
            mpp=meta["mpp"],
            picked_from=tuple(meta.get("picked_from", ())),
        ))
    if not frames:
        raise FileNotFoundError(f"no frames under {DERIVED / dataset}")
    return sorted(frames, key=lambda f: f.name)


def frame_place(frame: DroneFrame) -> str:
    """The `place` a simulated frame was cut from, as its `meta.json` records it."""
    return json.loads((frame.image_path.parent / "meta.json").read_text())["place"]


def time_blocks(names: list[str], gap_ms: int) -> pd.Series:
    """Pass index per frame name (`drone-<epoch ms>`): a new block starts after a gap longer than `gap_ms`."""
    stamps = pd.Series({n: int(n.split("-")[1]) for n in names}).sort_values()
    return (stamps.diff() > gap_ms).cumsum()
