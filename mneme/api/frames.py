"""GET /api/frames/{id}/thumb — spec.md 2.3 — and the live view, spec.md 2.8.

Frame content is immutable once written, so we can cache hard.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Query, Request
from fastapi.responses import FileResponse, StreamingResponse

from . import ApiError

log = logging.getLogger(__name__)

router = APIRouter()

CACHE_CONTROL = "public, max-age=31536000, immutable"

BOUNDARY = "mnemeframe"
LIVE_IDLE_TIMEOUT_S = 20.0
"""How long to wait for a frame before giving up on the stream. Longer than any
gap the camera should produce; a client that finds the stream closed reconnects
and gets whatever the camera is doing then."""


@router.get("/frames/live.mjpg")
async def live_stream(request: Request) -> StreamingResponse:
    """The camera as a motion-JPEG stream — spec.md 2.8.

    `multipart/x-mixed-replace` rather than a websocket or HLS: an `<img src>`
    plays it with no JavaScript, no codec, and no buffering delay, which is what
    "live" has to mean when the point of the demo is that the room on screen is
    the room in front of you. The frames are the camera's own JPEG bytes,
    forwarded undecoded (capture.py), so the stream costs the Orin almost
    nothing while vLLM has the GPU.
    """
    runtime = request.app.state.runtime
    live = runtime.live
    if live.latest is None and runtime.config.no_camera:
        raise ApiError("FRAME_NOT_FOUND", "no camera; nothing to stream")

    async def frames():
        queue = live.subscribe()
        try:
            while True:
                if await request.is_disconnected():
                    return
                try:
                    jpeg = await asyncio.wait_for(queue.get(), LIVE_IDLE_TIMEOUT_S)
                except asyncio.TimeoutError:
                    log.debug("live stream idle for %ss; closing", LIVE_IDLE_TIMEOUT_S)
                    return
                yield (
                    f"--{BOUNDARY}\r\n"
                    f"Content-Type: image/jpeg\r\n"
                    f"Content-Length: {len(jpeg)}\r\n\r\n"
                ).encode("ascii") + jpeg + b"\r\n"
        finally:
            live.unsubscribe(queue)

    return StreamingResponse(
        frames(),
        media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


# Declared before /frames/{frame_id}/thumb so "latest" is not captured as an id.
@router.api_route("/frames/latest/thumb", methods=["GET", "HEAD"])
async def latest_thumb(request: Request, full: str | None = Query(default=None)) -> FileResponse:
    """The newest frame the change filter kept — the live view.

    Deliberately not cached: unlike a frame addressed by id, what this resolves
    to changes every second or so, and the whole point of the panel is that it
    keeps up. It is still not a video stream: the CSI sensor allows one Argus
    session and the capture pipeline holds it, so this is as live as it gets.
    """
    runtime = request.app.state.runtime
    row = await runtime.db.latest_frame()
    if row is None:
        raise ApiError("FRAME_NOT_FOUND", "no frames captured yet")
    want_full = full not in (None, "", "0", "false", "False")
    path = runtime.config.data_dir / (row.path if want_full else row.thumb_path)
    if not path.is_file():
        raise ApiError("FRAME_NOT_FOUND", f"frame {row.id} has no file")
    return FileResponse(
        path,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store", "X-Frame-Id": row.id, "X-Frame-Ts": row.ts},
    )


# FastAPI's APIRoute does not imply HEAD from GET the way Starlette's Route
# does, and `curl -sI` is part of the acceptance list (backend.md 8.8).
@router.api_route("/frames/{frame_id}/thumb", methods=["GET", "HEAD"])
async def frame_thumb(
    request: Request, frame_id: str, full: str | None = Query(default=None)
) -> FileResponse:
    runtime = request.app.state.runtime
    row = await runtime.db.get_frame(frame_id)
    if row is None:
        raise ApiError("FRAME_NOT_FOUND", f"no frame {frame_id}")
    want_full = full not in (None, "", "0", "false", "False")
    relative = row.path if want_full else row.thumb_path
    # DB holds paths relative to --data-dir (spec.md 0); resolve here.
    path = runtime.config.data_dir / relative
    if not path.is_file():
        raise ApiError("FRAME_NOT_FOUND", f"frame {frame_id} has no file at {relative}")
    return FileResponse(
        path, media_type="image/jpeg", headers={"Cache-Control": CACHE_CONTROL}
    )
