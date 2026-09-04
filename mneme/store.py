"""Persistence + fan-out. Owns the only write path into SQLite.

Responsibilities: encode frame JPEG/thumbnail, insert frames/events/embeddings,
append to the in-memory search index, and broadcast the new event to SSE
subscribers.
"""

from __future__ import annotations

import asyncio
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
BROADCAST_CAPACITY = 64
"""spec.md 2.5: slow clients drop events, we never resend."""


@dataclass(slots=True)
class Frame:
    """A captured image on its way through the pipeline."""

    ts: datetime
    mat: Any  # numpy ndarray (BGR) from cv2; typed loosely to keep cv2 optional


def encode_frame(mat: Any) -> tuple[bytes, bytes, int, int]:
    """(full jpeg, thumb jpeg, width, height). Synchronous — call in a thread."""
    import cv2

    height, width = mat.shape[:2]
    params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    ok, full = cv2.imencode(".jpg", mat, params)
    if not ok:
        raise RuntimeError("cv2.imencode failed for full frame")
    thumb_height = max(1, round(height * THUMB_WIDTH / width))
    small = cv2.resize(mat, (THUMB_WIDTH, thumb_height), interpolation=cv2.INTER_AREA)
    ok, thumb = cv2.imencode(".jpg", small, params)
    if not ok:
        raise RuntimeError("cv2.imencode failed for thumbnail")
    return full.tobytes(), thumb.tobytes(), width, height


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
        full, thumb, width, height = await asyncio.to_thread(encode_frame, frame.mat)
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
            self.index.add(event.id, normalized)
        self.broadcaster.publish(event.to_json())
        return event
