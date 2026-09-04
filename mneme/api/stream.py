"""GET /api/stream — SSE, spec.md 2.5.

`observed` payloads are the exact same shape as an /api/events element, so the
frontend reuses one parser. Heartbeat every 15s. `retry: 3000` lets EventSource
reconnect on its own. Dropped events are never resent: a reconnecting client
backfills via /api/events.
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Request
from sse_starlette.event import ServerSentEvent
from sse_starlette.sse import EventSourceResponse

from ..db import iso_now

log = logging.getLogger(__name__)

router = APIRouter()

HEARTBEAT_S = 15
RETRY_MS = 3000


def heartbeat_event() -> ServerSentEvent:
    """The library's own keepalive is an SSE comment, which EventSource never
    surfaces to the client. spec.md 2.5 promises a named `heartbeat` event, so
    we replace the ping payload instead of racing it with a second timer."""
    return ServerSentEvent(
        event="heartbeat", data=json.dumps({"ts": iso_now()}, ensure_ascii=False)
    )


@router.get("/stream")
async def stream(request: Request) -> EventSourceResponse:
    runtime = request.app.state.runtime
    queue = runtime.broadcaster.subscribe()

    async def publisher():
        # sse-starlette 2.4 dropped the `retry=` constructor kwarg; retry is
        # now a field on an emitted event, so send it as the first frame.
        yield {"retry": RETRY_MS}
        try:
            while True:
                payload = await queue.get()
                yield {
                    "event": "observed",
                    "data": json.dumps(payload, ensure_ascii=False),
                }
        finally:
            runtime.broadcaster.unsubscribe(queue)

    return EventSourceResponse(
        publisher(), ping=HEARTBEAT_S, ping_message_factory=heartbeat_event
    )
