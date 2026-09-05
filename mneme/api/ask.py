"""POST /api/ask — spec.md 2.4. The whole retrieval + refusal behaviour.

Flow: embed the question, cosine over the whole index, take top_k. If the best
score is below `--ask-min-score` we refuse *without* calling the LLM. Otherwise
we build the numbered Taipei-time context (sidecar.md 3.2) from at most three
above-threshold hits and hand it to the LLM verbatim.

Never fabricate: the LLM may also decide the context is insufficient and refuse
on its own, and in that case the citations are still returned as retrieved.
"""

from __future__ import annotations

import time
from typing import Any
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from ..db import parse_iso
from ..search import l2_normalize
from ..sidecar import REFUSAL, SidecarFailed, SidecarTimeout, SidecarUnavailable
from . import ApiError

router = APIRouter()

TAIPEI = ZoneInfo("Asia/Taipei")
MAX_CITATIONS = 3
"""top_k only widens retrieval; the response is capped at 3 (spec.md 2.4)."""


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)


def context_line(index: int, ts_iso: str, summary: str) -> str:
    """`[1] 2026-09-05 22:03(台北時間) <summary>`

    The single documented exception to the UTC-everywhere rule: the LLM must
    never do timezone maths, so the conversion happens here.
    """
    local = parse_iso(ts_iso).astimezone(TAIPEI)
    return f"[{index}] {local.strftime('%Y-%m-%d %H:%M')}(台北時間) {summary}"


@router.post("/ask")
async def ask(request: Request, body: AskRequest) -> dict[str, Any]:
    runtime = request.app.state.runtime
    started = time.monotonic()
    question = body.question.strip()
    if not question:
        raise ApiError("INVALID_PARAM", "question must not be blank")

    if not runtime.index.count:
        # Nothing indexed: refuse without reserving the sidecar at all, so an
        # empty database never queues behind a capture-pipeline describe.
        return await _respond(runtime, question, REFUSAL, [], started)

    # embed + answer as one reservation, so the pipeline cannot interleave a
    # describe between them (see SocketSidecar.session).
    async with runtime.sidecar.session():
        answer, citations = await _retrieve(runtime, question, body.top_k, started)
    # The `queries` row is written outside the session: it is our own DB, and
    # holding the sidecar while we touch it would stall the pipeline for no
    # reason.
    return await _respond(runtime, question, answer, citations, started)


async def _retrieve(
    runtime, question: str, top_k: int, started: float
) -> tuple[str, list[dict[str, Any]]]:
    """Embed, search, and answer. Called with the sidecar already reserved."""
    threshold = runtime.config.ask_min_score
    vec = await _guarded(runtime, question, started, runtime.sidecar.embed(question))
    try:
        query = l2_normalize(vec)
    except ValueError as exc:
        raise ApiError("INTERNAL", f"embedding for the question is unusable: {exc}") from exc
    hits = runtime.index.search(query, top_k)

    if not hits or hits[0][1] < threshold:
        # Below threshold: refuse without calling the LLM at all.
        return REFUSAL, []

    kept = [(eid, score) for eid, score in hits if score >= threshold][:MAX_CITATIONS]
    rows = await runtime.db.events_by_ids([eid for eid, _ in kept])
    citations: list[dict[str, Any]] = []
    context: list[str] = []
    for position, (event_id, score) in enumerate(kept, start=1):
        row = rows.get(event_id)
        if row is None:
            continue
        citations.append(
            {
                "event_id": row.id,
                "ts": row.ts,
                "summary": row.summary,
                "thumb_url": f"/api/frames/{row.frame_id}/thumb",
                "score": score,
            }
        )
        context.append(context_line(position, row.ts, row.summary))

    answer = await _guarded(
        runtime, question, started, runtime.sidecar.answer(question, context)
    )
    return answer, citations


async def _record(
    runtime, question: str, answer: str, citations: list[dict[str, Any]], started: float
) -> int:
    """Every call, success or failure, writes one `queries` row."""
    latency_ms = int((time.monotonic() - started) * 1000)
    await runtime.db.insert_query(
        question=question,
        answer=answer,
        cited=[c["event_id"] for c in citations],
        latency_ms=latency_ms,
    )
    return latency_ms


async def _respond(
    runtime,
    question: str,
    answer: str,
    citations: list[dict[str, Any]],
    started: float,
) -> dict[str, Any]:
    latency_ms = await _record(runtime, question, answer, citations, started)
    return {"answer": answer, "citations": citations, "latency_ms": latency_ms}


async def _guarded(runtime, question: str, started: float, coro):
    """Sidecar failures still owe a `queries` row (spec.md 2.4) before the
    error propagates to the taxonomy handlers in `api/__init__.py`."""
    try:
        return await coro
    except (SidecarUnavailable, SidecarTimeout, SidecarFailed) as exc:
        await _record(runtime, question, f"[error] {exc}", [], started)
        raise
