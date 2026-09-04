"""GET /api/events — spec.md 2.2.

Ordering is fixed `ts DESC, id DESC`. `cursor` is exclusive, `from`/`to` are
inclusive, `next_cursor` is always the last id on the page (or null on a short
page). `q` is a case-insensitive LIKE over `summary` only.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query, Request

from ..db import iso, parse_iso, valid_id
from . import ApiError

router = APIRouter()

DEFAULT_LIMIT = 50
MAX_LIMIT = 200


def _limit(raw: str | None) -> int:
    if raw is None or raw == "":
        return DEFAULT_LIMIT
    try:
        value = int(raw)
    except ValueError as exc:
        raise ApiError("INVALID_PARAM", f"limit must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise ApiError("INVALID_PARAM", "limit must be > 0")
    return min(value, MAX_LIMIT)  # over 200 clamps silently, never errors


def _timestamp(raw: str | None, name: str) -> str | None:
    if raw is None or raw == "":
        return None
    try:
        parsed = parse_iso(raw)
    except ValueError as exc:
        raise ApiError("INVALID_PARAM", f"{name} must be ISO 8601, got {raw!r}") from exc
    return iso(parsed)


@router.get("/events")
async def list_events(
    request: Request,
    from_: str | None = Query(default=None, alias="from"),
    to: str | None = Query(default=None),
    limit: str | None = Query(default=None),
    cursor: str | None = Query(default=None),
    q: str | None = Query(default=None),
) -> dict[str, Any]:
    runtime = request.app.state.runtime
    page_size = _limit(limit)
    if cursor is not None and cursor != "" and not valid_id(cursor):
        raise ApiError("INVALID_CURSOR", f"not a valid event id: {cursor!r}")
    rows = await runtime.db.list_events(
        limit=page_size,
        from_ts=_timestamp(from_, "from"),
        to_ts=_timestamp(to, "to"),
        cursor=cursor or None,
        q=q or None,
    )
    if rows is None:
        # Well-formed cursor pointing at nothing: treat as end of stream.
        return {"events": [], "next_cursor": None}
    events = [row.to_json() for row in rows]
    next_cursor = events[-1]["id"] if len(events) == page_size else None
    return {"events": events, "next_cursor": next_cursor}
