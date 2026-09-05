"""Persistence + fan-out. Owns the only write path into SQLite.

Responsibilities: encode frame JPEG/thumbnail, insert frames/events/embeddings,
append to the in-memory search index, and broadcast the new event to SSE
subscribers.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .db import Database, EventRow, FrameRow, encode_vec, iso, new_id
from .search import SearchIndex, l2_normalize

log = logging.getLogger(__name__)

THUMB_WIDTH = 320
"""spec.md 0: thumbnails are always JPEG, 320px wide."""

JPEG_QUALITY = 85

LIVE_WIDTH = 960
LIVE_QUALITY = 70
"""The live MJPEG view is a picture of the room, not the archive copy; it is
sent 21 times a second, so it is worth half the bytes."""

BROADCAST_CAPACITY = 64
"""spec.md 2.5: slow clients drop events, we never resend."""


@dataclass(slots=True)
class Frame:
    """A captured image on its way through the pipeline.

    `jpeg` is the camera's own bytes when the source already produced JPEG
    (`--camera-cmd`), and `mat` is decoded from it only when something actually
    needs pixels. At 21fps the live view forwards those bytes untouched, so a
    frame nobody looks at closely never pays for a decode.
    """

    ts: datetime
    mat: Any = None  # numpy ndarray (BGR) from cv2; typed loosely to keep cv2 optional
    jpeg: bytes | None = None

    def decode(self) -> Any:
        """Pixels, decoding the JPEG once and caching. Call inside a thread."""
        if self.mat is None:
            if self.jpeg is None:
                raise ValueError("frame has neither pixels nor JPEG bytes")
            import cv2
            import numpy as np

            self.mat = cv2.imdecode(np.frombuffer(self.jpeg, dtype="uint8"), cv2.IMREAD_COLOR)
            if self.mat is None:
                raise ValueError("cv2.imdecode failed for a captured frame")
        return self.mat


class LiveStream:
    """Latest-frame fan-out for the MJPEG view (spec.md 2.8).

    Deliberately not the SSE `Broadcaster`: a viewer who cannot keep up wants
    the newest frame, not the oldest undelivered one, so every subscriber queue
    holds exactly one frame and publishing overwrites it. A stalled browser tab
    therefore costs one buffer, not a growing backlog.
    """

    def __init__(self) -> None:
        self.latest: bytes | None = None
        self.latest_ts: datetime | None = None
        self._subscribers: set[asyncio.Queue[bytes]] = set()

    def subscribe(self) -> asyncio.Queue[bytes]:
        queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=1)
        self._subscribers.add(queue)
        if self.latest is not None:
            queue.put_nowait(self.latest)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[bytes]) -> None:
        self._subscribers.discard(queue)

    @property
    def viewer_count(self) -> int:
        return len(self._subscribers)

    def publish(self, jpeg: bytes, ts: datetime) -> None:
        self.latest = jpeg
        self.latest_ts = ts
        for queue in self._subscribers:
            with contextlib.suppress(asyncio.QueueEmpty):
                queue.get_nowait()
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(jpeg)


def encode_frame(frame: Frame) -> tuple[bytes, bytes, int, int]:
    """(full jpeg, thumb jpeg, width, height). Synchronous — call in a thread.

    A frame that arrived as JPEG is stored as the camera encoded it: re-encoding
    would cost a decode and a generation of quality to produce the same picture.
    """
    import cv2

    mat = frame.decode()
    height, width = mat.shape[:2]
    params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    if frame.jpeg is not None:
        full_bytes = frame.jpeg
    else:
        ok, full = cv2.imencode(".jpg", mat, params)
        if not ok:
            raise RuntimeError("cv2.imencode failed for full frame")
        full_bytes = full.tobytes()
    thumb_height = max(1, round(height * THUMB_WIDTH / width))
    small = cv2.resize(mat, (THUMB_WIDTH, thumb_height), interpolation=cv2.INTER_AREA)
    ok, thumb = cv2.imencode(".jpg", small, params)
    if not ok:
        raise RuntimeError("cv2.imencode failed for thumbnail")
    return full_bytes, thumb.tobytes(), width, height


def encode_live(mat: Any) -> bytes:
    """A JPEG for the live view from raw pixels. Synchronous — call in a thread.

    Only the `cv2.VideoCapture` source needs this: `--camera-cmd` already hands
    us JPEG, and re-encoding what the camera encoded would be pure waste.
    """
    import cv2

    height, width = mat.shape[:2]
    if width > LIVE_WIDTH:
        target = (LIVE_WIDTH, max(1, round(height * LIVE_WIDTH / width)))
        mat = cv2.resize(mat, target, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", mat, [int(cv2.IMWRITE_JPEG_QUALITY), LIVE_QUALITY])
    if not ok:
        raise RuntimeError("cv2.imencode failed for the live frame")
    return buf.tobytes()


def write_frame_files(
    data_dir: Path, frame_id: str, full: bytes, thumb: bytes
) -> tuple[str, str]:
    """Writes both files and returns paths relative to --data-dir (spec.md 0)."""
    rel_full = Path("frames") / f"{frame_id}.jpg"
    rel_thumb = Path("thumbs") / f"{frame_id}.jpg"
    for rel, payload in ((rel_full, full), (rel_thumb, thumb)):
        target = data_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    return rel_full.as_posix(), rel_thumb.as_posix()


class Broadcaster:
    """Fan-out to SSE subscribers. Bounded per-subscriber, drops on overflow."""

    def __init__(self, capacity: int = BROADCAST_CAPACITY) -> None:
        self.capacity = capacity
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=self.capacity)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self._subscribers.discard(queue)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    def publish(self, payload: dict[str, Any]) -> None:
        for queue in self._subscribers:
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                log.debug("dropping SSE event for a slow subscriber")


class Store:
    def __init__(
        self,
        *,
        db: Database,
        index: SearchIndex,
        data_dir: Path,
        broadcaster: Broadcaster,
    ) -> None:
        self.db = db
        self.index = index
        self.data_dir = data_dir
        self.broadcaster = broadcaster

    async def save_frame(self, frame: Frame) -> FrameRow:
        frame_id = new_id("frm")
        full, thumb, width, height = await asyncio.to_thread(encode_frame, frame)
        rel_full, rel_thumb = await asyncio.to_thread(
            write_frame_files, self.data_dir, frame_id, full, thumb
        )
        row = FrameRow(
            id=frame_id,
            ts=iso(frame.ts),
            path=rel_full,
            thumb_path=rel_thumb,
            width=width,
            height=height,
        )
        await self.db.insert_frame(row)
        return row

    async def save_event(
        self,
        *,
        frame: FrameRow,
        summary: str,
        objects: list[str],
        confidence: float,
        source: str,
        vec: np.ndarray | None,
    ) -> EventRow:
        """Insert the event (plus its normalised vector) and publish it."""
        event = EventRow(
            id=new_id("evt"),
            ts=frame.ts,
            frame_id=frame.id,
            summary=summary,
            objects=objects,
            confidence=confidence,
            source=source,
        )
        normalized: np.ndarray | None = None
        blob: bytes | None = None
        if vec is not None:
            try:
                normalized = l2_normalize(vec)
            except ValueError:
                log.error("event %s got a zero-norm embedding; storing without vector", event.id)
            else:
                if normalized.shape[0] != self.index.dim:
                    raise ValueError(
                        f"embedding dim {normalized.shape[0]} != index dim {self.index.dim}"
                    )
                blob = encode_vec(normalized)
        await self.db.insert_event(event, dim=self.index.dim, blob=blob)
        if normalized is not None and blob is not None:
            self.index.add(event.id, normalized, event.ts)
        self.broadcaster.publish(event.to_json())
        return event
