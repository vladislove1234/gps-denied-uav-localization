"""Hand-corrected satellite labels from the annotator's export, as georeferenced rasters or image/label pairs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from uavloc.data.drone_frames import DERIVED
from uavloc.data.satellite import GeoImage
from uavloc.geo import LatLon

VILLAGE_EXPORT = "ua-village-annotated"
ORTHO_EXPORT = "ua-ortho-annotated"
IGNORE_LABEL = 255


@dataclass(frozen=True)
class AnnotatedImage:
    """One exported date: imagery `(H, W, 3)` uint8 and its class-id label `(H, W)` uint8 on the same pixels."""

    location: str
    date: str
    rgb: np.ndarray
    label: np.ndarray


def label_records(dataset: str = VILLAGE_EXPORT, derived_dir: Path = DERIVED) -> list[dict]:
    path = derived_dir / dataset / "samples.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def uncovered(shape: tuple[int, int], offset_px: tuple[int, int]) -> np.ndarray:
    """Strip of the label frame that a date's alignment offset leaves without imagery."""
    dx, dy = offset_px
    h, w = shape
    mask = np.zeros(shape, bool)
    if dx > 0:
        mask[:, w - dx:] = True
    if dx < 0:
        mask[:, :-dx] = True
    if dy > 0:
        mask[h - dy:, :] = True
    if dy < 0:
        mask[:-dy, :] = True
    return mask


def load_annotated_images(dataset: str = VILLAGE_EXPORT, derived_dir: Path = DERIVED) -> list[AnnotatedImage]:
    """Every exported date, sorted by (location, date); the strip its offset leaves uncovered is `IGNORE_LABEL`."""
    records = sorted(label_records(dataset, derived_dir), key=lambda r: (r["location"], r["date"]))
    images = []
    for r in records:
        rgb = np.array(Image.open(derived_dir / r["image_file"]).convert("RGB"))
        label = np.array(Image.open(derived_dir / dataset / r["label_file"]))
        if rgb.shape[:2] != label.shape:
            raise ValueError(f"{r['location']} / {r['date']}: image {rgb.shape[:2]} vs label {label.shape}")
        label[uncovered(label.shape, r["image_offset_px"])] = IGNORE_LABEL
        images.append(AnnotatedImage(r["location"], r["date"], rgb, label))
    return images


def load_label_map(location: str, date: str, dataset: str = VILLAGE_EXPORT) -> GeoImage:
    """Class-id raster `(H, W)` of one exported date, georeferenced by its corners.

    Satellite exports carry the Web Mercator `bounds` (linear lat/lon is within 5 cm of
    Web Mercator over 1 km). Ortho exports carry `bounds_approx`, the image's own corner
    fit, which is exactly the georeference the harness uses for that image.
    Only dates on the reference grid (`image_offset_px == [0, 0]`) are accepted: other
    dates' labels are shifted to follow their own imagery, not the ground.
    """
    matches = [r for r in label_records(dataset) if r["location"] == location and r["date"] == date]
    if not matches:
        raise FileNotFoundError(f"no exported label for {location} / {date} in {dataset}")
    record = matches[0]
    if any(record["image_offset_px"]):
        raise ValueError(f"{location} / {date} is shifted by {record['image_offset_px']} px; use a reference date")
    bounds = record.get("bounds") or record["bounds_approx"]
    labels = np.asarray(Image.open(DERIVED / dataset / record["label_file"]))
    return GeoImage(labels, LatLon(bounds["top"], bounds["left"]), LatLon(bounds["bottom"], bounds["right"]))
