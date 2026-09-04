"""GET /api/health — spec.md 2.1. Always HTTP 200; the client reads `status`."""

from __future__ import annotations

import time

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
async def health(request: Request) -> dict[str, object]:
    state = request.app.state
    runtime = state.runtime
    sidecar = runtime.sidecar
    sidecar_status = sidecar.status
    return {
        # degraded = sidecar is gone but old data still reads.
        "status": "degraded" if sidecar_status == "down" else "ok",
        "device": runtime.device,
        "vlm_model": sidecar.vlm_model,
        "llm_model": sidecar.llm_model,
        "embed_model": sidecar.embed_model,
        "embed_dim": runtime.config.embed_dim,
        # Measured by the background probe (backend.md 8.4); never computed
        # here, a handler must not wait on the network.
        "offline": runtime.offline,
        "uptime_s": int(time.monotonic() - runtime.started_at),
        "capture_fps": runtime.fps.value(),
        "queue_depth": runtime.queue_depth(),
        "event_count": await runtime.db.count_events(),
        "sidecar": sidecar_status,
        "mode": runtime.config.mode,
    }
