"""change_filter — the gate that keeps the VLM affordable. backend.md 3.3.

Downscale to 64x64 grayscale, mean absolute diff against the previous kept
frame, pass only above the threshold, then hold a cooldown so one event is not
described ten times. This drops roughly nine out of ten frames, which is what
makes real-time work on an Orin.
"""

from __future__ import annotations

import time
from typing import Any

DOWNSCALE = (64, 64)


def downscale(mat: Any) -> Any:
    """Grayscale 64x64 float32. Synchronous — call inside asyncio.to_thread."""
    import cv2

    gray = cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, DOWNSCALE, interpolation=cv2.INTER_AREA)
    return small.astype("float32")


def mean_abs_diff(a: Any, b: Any) -> float:
    import numpy as np

    return float(np.abs(a - b).mean())


class ChangeFilter:
    """Stateful gate; one instance per pipeline."""

    def __init__(self, threshold: float, cooldown_ms: int) -> None:
        self.threshold = threshold
        self.cooldown_s = cooldown_ms / 1000.0
        self._previous: Any | None = None
        self._last_pass: float | None = None
        self.last_diff = 0.0

    def should_pass(self, small: Any, *, now: float | None = None) -> bool:
        """`small` is the output of `downscale`. Returns True to admit the frame."""
        now = time.monotonic() if now is None else now
        previous, self._previous = self._previous, small
        if previous is None:
            # First frame is always admitted: it establishes the baseline and
            # gives the timeline something to show immediately.
            self._last_pass = now
            return True
        self.last_diff = mean_abs_diff(small, previous)
        if self.last_diff < self.threshold:
            return False
        if self._last_pass is not None and (now - self._last_pass) < self.cooldown_s:
            return False
        self._last_pass = now
        return True
