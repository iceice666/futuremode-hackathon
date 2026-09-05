"""Camera → Frame. docs/backend.md 3.3.

Two sources, same output:

* ``cv2.VideoCapture`` on ``--camera`` (default),
* an external writer started by ``--camera-cmd`` that drops JPEGs into
  ``<data-dir>/incoming``; we read each file then delete it, and forward the
  bytes undecoded. That is the documented escape hatch for Jetsons where cv2
  cannot open the device (spec.md 7).

Every cv2 call goes through ``asyncio.to_thread``: cv2 releases the GIL, but a
synchronous call still parks the event loop and makes SSE heartbeats and
``/api/health`` stutter.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import shlex
import time
from collections import deque
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .store import Frame

log = logging.getLogger(__name__)

FPS_WINDOW_S = 10.0
"""spec.md 2.1: capture_fps is a measured 10s sliding window, not the setting."""


class FpsMeter:
    def __init__(self, window_s: float = FPS_WINDOW_S) -> None:
        self.window_s = window_s
        self._marks: deque[float] = deque()

    def mark(self) -> None:
        now = time.monotonic()
        self._marks.append(now)
        self._trim(now)

    def value(self) -> float:
        now = time.monotonic()
        self._trim(now)
        if len(self._marks) < 2:
            return 0.0
        span = now - self._marks[0]
        if span <= 0:
            return 0.0
        return round(len(self._marks) / span, 2)

    def _trim(self, now: float) -> None:
        cutoff = now - self.window_s
        while self._marks and self._marks[0] < cutoff:
            self._marks.popleft()


MAX_BACKLOG = 4
"""Incoming JPEGs we are willing to be behind before dropping the oldest."""

JPEG_EOI = b"\xff\xd9"


def _read_jpeg(path: Path) -> bytes | None:
    """Bytes of a finished JPEG; None while the writer still has it open.

    multifilesink closes each file before opening the next, so a complete file
    ends in the end-of-image marker. Checking for it is cheaper and more certain
    than racing on mtime, and a file that never completes is dropped rather than
    retried forever.
    """
    try:
        data = path.read_bytes()
    except OSError:
        return b""
    if not data.endswith(JPEG_EOI):
        return None
    with contextlib.suppress(OSError):
        path.unlink()
    return data


def require_cv2() -> Any:
    """Import cv2 eagerly at startup so a bad install fails now, not at the
    first sample (backend.md 8.2)."""
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "cv2 is unavailable. JetPack ships it in system site-packages: create "
            "the venv with `python -m venv --system-site-packages .venv`. Never "
            "`pip install opencv-python` on arm64."
        ) from exc
    return cv2


async def camera_frames(device: str, fps: float) -> AsyncIterator[Frame]:
    """Frames from cv2.VideoCapture, paced at `fps`."""
    cv2 = require_cv2()
    source: Any = int(device) if device.isdigit() else device
    cap = await asyncio.to_thread(cv2.VideoCapture, source)
    try:
        if not await asyncio.to_thread(cap.isOpened):
            raise RuntimeError(
                f"cv2.VideoCapture could not open {device!r}; "
                "use --camera-cmd as the documented fallback (spec.md 7)"
            )
        interval = 1.0 / fps if fps > 0 else 0.5
        while True:
            started = time.monotonic()
            ok, mat = await asyncio.to_thread(cap.read)
            if not ok or mat is None:
                log.warning("camera read failed; retrying")
                await asyncio.sleep(interval)
                continue
            yield Frame(ts=datetime.now(timezone.utc), mat=mat)
            await asyncio.sleep(max(0.0, interval - (time.monotonic() - started)))
    finally:
        await asyncio.to_thread(cap.release)


async def _drain_stderr(process: asyncio.subprocess.Process) -> None:
    """A camera command that dies must say why: unread stderr on a PIPE both
    hides the reason and can eventually block the child on a full buffer."""
    stream = process.stderr
    if stream is None:
        return
    while line := await stream.readline():
        log.warning("camera-cmd: %s", line.decode(errors="replace").rstrip())


async def incoming_frames(
    command: str, incoming_dir: Path, fps: float
) -> AsyncIterator[Frame]:
    """Run `--camera-cmd` and consume the JPEGs it writes.

    The bytes are forwarded without decoding: at 21fps the live view is the only
    consumer of most frames and it wants JPEG anyway, so a decode happens later
    and only for the frames the change filter actually looks at.
    """
    incoming_dir.mkdir(parents=True, exist_ok=True)
    process = await asyncio.create_subprocess_exec(
        *shlex.split(command),
        cwd=incoming_dir,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    stderr_task = asyncio.create_task(_drain_stderr(process), name="camera-cmd-stderr")
    poll_interval = 1.0 / fps if fps > 0 else 0.5
    try:
        while True:
            if process.returncode is not None:
                raise RuntimeError(f"--camera-cmd exited with {process.returncode}")
            files = sorted(p for p in incoming_dir.glob("*.jpg") if p.is_file())
            if not files:
                await asyncio.sleep(poll_interval)
                continue
            # A live view that replays a backlog is not live. If the writer got
            # ahead of us -- a slow pass, a paused event loop -- throw the stale
            # middle away and carry on from the newest frames.
            if len(files) > MAX_BACKLOG:
                for path in files[:-MAX_BACKLOG]:
                    with contextlib.suppress(OSError):
                        path.unlink()
                log.debug("dropped %d stale incoming frames", len(files) - MAX_BACKLOG)
                files = files[-MAX_BACKLOG:]
            served = 0
            for path in files:
                data = await asyncio.to_thread(_read_jpeg, path)
                if data is None:
                    # Still being written (no end-of-image marker yet). Leave it
                    # on disk; the next pass will find it closed.
                    break
                if not data:
                    continue
                served += 1
                yield Frame(ts=datetime.now(timezone.utc), jpeg=data)
            if not served:
                # Everything on disk was half-written or unreadable; waiting is
                # the only useful thing to do, and not waiting is a spin.
                await asyncio.sleep(poll_interval)
    finally:
        if process.returncode is None:
            process.terminate()
            with contextlib.suppress(asyncio.TimeoutError, ProcessLookupError):
                await asyncio.wait_for(process.wait(), timeout=3.0)
        stderr_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await stderr_task


def open_source(config) -> AsyncIterator[Frame]:
    """Pick the frame source from config. `--camera-cmd` wins over `--camera`."""
    if config.camera_cmd:
        return incoming_frames(config.camera_cmd, config.incoming_dir, config.capture_fps)
    return camera_frames(config.camera, config.capture_fps)
