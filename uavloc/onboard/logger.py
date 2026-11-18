"""Flight log on the Pi's disk: fixes and states as JSON lines, every Nth frame as a JPEG, the manifest."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import IO

from PIL import Image

from uavloc.onboard.contracts import Fix, FlightState, Frame

FIXES_NAME = "fixes.jsonl"
STATES_NAME = "states.jsonl"
FRAMES_DIR = "frames"
MANIFEST_NAME = "manifest.json"
FRAME_JPEG_QUALITY = 90
DEFAULT_FRAME_EVERY = 5
MS_PER_S = 1000


class JsonlFlightLogger:
    """`FlightLogger` writing into `directory`. Every line is flushed, so a power cut loses at most one."""

    def __init__(self, directory: Path, frame_every: int = DEFAULT_FRAME_EVERY, manifest: Path | None = None) -> None:
        if frame_every < 1:
            raise ValueError(f"frame_every must be at least 1, got {frame_every}")
        self.directory = directory
        self.frame_every = frame_every
        (directory / FRAMES_DIR).mkdir(parents=True, exist_ok=True)
        if manifest is not None:
            shutil.copyfile(manifest, directory / MANIFEST_NAME)
        self._fixes = (directory / FIXES_NAME).open("a")
        self._states = (directory / STATES_NAME).open("a")
        self._frames_seen = 0

    def log_fix(self, fix: Fix) -> None:
        position = fix.position
        self._write_line(self._fixes, {"time_s": fix.time_s, "lat": position.lat if position else None,
                                       "lon": position.lon if position else None,
                                       "uncertainty_m": fix.uncertainty_m, "quality": fix.quality,
                                       "reason": fix.reason})

    def log_state(self, state: FlightState) -> None:
        record = asdict(state)
        record["position"] = list(state.position) if state.position else None
        self._write_line(self._states, record)

    def log_frame(self, frame: Frame) -> None:
        index = self._frames_seen
        self._frames_seen += 1
        if index % self.frame_every:
            return
        name = f"frame-{index:06d}-{round(frame.time_s * MS_PER_S):09d}ms.jpg"
        Image.fromarray(frame.image).save(self.directory / FRAMES_DIR / name, quality=FRAME_JPEG_QUALITY)

    def close(self) -> None:
        for handle in (self._fixes, self._states):
            handle.close()

    @staticmethod
    def _write_line(handle: IO[str], record: dict) -> None:
        handle.write(json.dumps(record) + "\n")
        handle.flush()
