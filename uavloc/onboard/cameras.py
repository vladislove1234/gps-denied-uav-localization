"""Camera sources. v1 has no real camera: the stub localizer needs no image."""

from __future__ import annotations

from uavloc.onboard.contracts import Frame


class NullCamera:
    """A camera that never has a frame."""

    def next_frame(self) -> Frame | None:
        return None
