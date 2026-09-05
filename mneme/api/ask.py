"""POST /api/ask — spec.md 2.4. The whole retrieval + refusal behaviour.

Flow: embed the question, cosine over the index, take top_k, build the numbered
Taipei-time context (sidecar.md 3.2) from at most three hits and hand it to the
LLM verbatim.

Scope: `since`/`until` in the request restrict retrieval to a time window, and
a question about *now* with no explicit window searches only the newest
`RECENT_WINDOW` events. See `asks_about_now` for why cosine alone cannot answer
one. Both are slices of the chronological index (search.py), so scoping makes
retrieval cheaper, not more expensive.

Refusal has two paths and they are not interchangeable:

- `--ask-min-score` is a *floor guard*. Below it we refuse without calling the
  LLM and return no citations. It catches an empty index, a broken vector, and
  the mock sidecar's near-zero cosine for unrelated text. On real bge-m3 it
  practically never fires: Chinese cosine has a floor around 0.7 and the
  separation between witnessed and unwitnessed is ~0.1 (sidecar.md 8.9).
- Semantic refusal is the LLM's job, per the sidecar.md 3.2 system prompt. That
  is the normal path on real models, and there the citations are still returned
  as retrieved — the judges get to see what we found, we just do not make
  anything up.

So a refusal answer may come back with a non-empty `citations`. That is the
contract, not a bug: never key any logic off `citations == []`.
"""

from __future__ import annotations

import re
import time
from typing import Any
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field, field_validator

from ..db import parse_iso
from ..search import l2_normalize
from ..sidecar import REFUSAL, SidecarFailed, SidecarTimeout, SidecarUnavailable
from . import ApiError

router = APIRouter()

TAIPEI = ZoneInfo("Asia/Taipei")
MAX_CITATIONS = 3
"""top_k only widens retrieval; the response is capped at 3 (spec.md 2.4)."""


MIN_TOP_K = 1
MAX_TOP_K = 20

RECENT_WINDOW = 40
"""How many of the newest events a question about *now* is allowed to see.

A count, not a duration, and deliberately: when the room has sat still for an
hour, "現在桌上有什麼" should still answer from the last thing the camera saw
rather than refuse. With the change filter's cooldown a busy room produces an
event every few seconds, so 40 is the last few minutes of activity."""

NOW_TERMS = (
    "現在", "现在", "目前", "此刻", "當下", "当下", "眼前",
    "剛剛", "刚刚", "剛才", "刚才", "方才", "最近", "正在",
    "這會兒", "这会儿",
)
NOW_PATTERN = re.compile(
    r"\b(right now|just now|at the moment|currently|now|latest)\b", re.IGNORECASE
)


def asks_about_now(question: str) -> bool:
    """Is this a question about the present moment rather than about the day?

    It has to be asked in the question's own words, because the embedding
    cannot answer it. Event summaries describe objects and actions, never
    times, so "目前" contributes nothing to the query vector -- and on real
    bge-m3 the Chinese cosine floor is ~0.7 with only ~0.1 separating a
    witnessed thing from an unwitnessed one (sidecar.md 8.9). A frame from
    eight hours ago that happens to show the same desk therefore outranks the
    one from a minute ago, which is exactly the wrong answer to "現在".

    Both character sets are listed: the VLM writes simplified, people type
    traditional, and the question is matched as raw text.
    """
    if any(term in question for term in NOW_TERMS):
        return True
    return NOW_PATTERN.search(question) is not None


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    top_k: int = 5
    since: str | None = None
    until: str | None = None
    """UTC ISO 8601 bounds (spec.md 2.4). Either may be omitted for an open end.
    `until` covers the whole second it names."""

    @field_validator("since", "until")
    @classmethod
    def _check_iso(cls, value: str | None) -> str | None:
        """Reject a malformed bound instead of silently answering about the
        whole day: a question scoped to a window the caller cannot see is worse
        than an error they can."""
        if value is None:
            return None
        value = value.strip()
        if not value:
            return None
        try:
            parse_iso(value)
        except ValueError as exc:
            raise ValueError(f"not a UTC ISO 8601 timestamp: {value!r}") from exc
        return value

    @field_validator("top_k")
    @classmethod
    def _clamp_top_k(cls, value: int) -> int:
        """spec.md 2.4: top_k clamps to 1-20, it does not error. Same shape as
        the `limit` param in 2.2 -- an out-of-range retrieval width is not
        something the caller needs to hear about."""
        return min(max(value, MIN_TOP_K), MAX_TOP_K)


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
        answer, citations = await _retrieve(
            runtime, question, body.top_k, started, body.since, body.until
        )
    # The `queries` row is written outside the session: it is our own DB, and
    # holding the sidecar while we touch it would stall the pipeline for no
    # reason.
    return await _respond(runtime, question, answer, citations, started)


async def _retrieve(
    runtime,
    question: str,
    top_k: int,
    started: float,
    since: str | None = None,
    until: str | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Embed, search, and answer. Called with the sidecar already reserved."""
    threshold = runtime.config.ask_min_score
    vec = await _guarded(runtime, question, started, runtime.sidecar.embed(question))
    try:
        query = l2_normalize(vec)
    except ValueError as exc:
        raise ApiError("INTERNAL", f"embedding for the question is unusable: {exc}") from exc
    # An explicit window is the caller telling us when to look, so it wins;
    # the "現在" shortcut only applies when they did not say.
    scoped = since is not None or until is not None
    rows = runtime.index.rows_between(since, until) if scoped else None
    # "現在" is a filter on time, not a direction in embedding space.
    recent = None if scoped or not asks_about_now(question) else RECENT_WINDOW
    hits = runtime.index.search(query, top_k, recent=recent, rows=rows)

    if not hits or hits[0][1] < threshold:
        # Floor guard only (see module docstring): nothing retrievable is even
        # close, so refuse without spending an LLM call. Semantic refusal is
        # the prompt's job and happens further down, with citations attached.
        # An empty window lands here too, which is the honest answer: we did
        # not see anything then.
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
