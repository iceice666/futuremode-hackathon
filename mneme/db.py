"""SQLite access. Schema is docs/backend.md 1; concurrency rules are 8.6.

Shape of the concurrency model:

* one dedicated write connection, serialised behind an ``asyncio.Lock``;
  every write runs in ``asyncio.to_thread`` so the event loop never blocks.
* reads run inline on the event loop. They are sub-millisecond, and
  ``_sqlite3`` releases the GIL around every step, so a thread pool turns
  them into a lock convoy that costs an order of magnitude more than the
  query — see ``Database._read``.
* one read connection per thread via ``threading.local()``. In practice that
  is a single connection on the loop thread; the per-thread indirection keeps
  a read that does get wrapped in ``to_thread`` correct instead of unsafe.
* ``isolation_level=None`` plus explicit ``BEGIN``/``COMMIT``. The implicit
  mode leaves ``SELECT`` outside the transaction, which misleads debugging.
* ``PRAGMA foreign_keys`` is per connection, so every new connection re-runs
  the pragma block.
"""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from ulid import ULID

SCHEMA_VERSION = 1

VEC_DTYPE = "<f4"
"""Explicit little-endian f32. Never the `float32` alias: the byte order in
the schema comment must not depend on the host machine."""

SCHEMA = """
CREATE TABLE IF NOT EXISTS frames (
    id          TEXT PRIMARY KEY,
    ts          TEXT NOT NULL,
    path        TEXT NOT NULL,
    thumb_path  TEXT NOT NULL,
    width       INTEGER NOT NULL,
    height      INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_frames_ts ON frames(ts);

CREATE TABLE IF NOT EXISTS events (
    id          TEXT PRIMARY KEY,
    ts          TEXT NOT NULL,
    frame_id    TEXT NOT NULL REFERENCES frames(id),
    summary     TEXT NOT NULL,
    objects     TEXT NOT NULL DEFAULT '[]',
    confidence  REAL NOT NULL DEFAULT 1.0,
    source      TEXT NOT NULL DEFAULT 'vlm'
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);

CREATE TABLE IF NOT EXISTS embeddings (
    event_id    TEXT PRIMARY KEY REFERENCES events(id) ON DELETE CASCADE,
    dim         INTEGER NOT NULL,
    vec         BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS queries (
    id          TEXT PRIMARY KEY,
    ts          TEXT NOT NULL,
    question    TEXT NOT NULL,
    answer      TEXT NOT NULL,
    cited       TEXT NOT NULL DEFAULT '[]',
    latency_ms  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL
);
"""

ID_RE = re.compile(r"^(evt|frm)_[0-9A-HJKMNP-TV-Z]{26}$", re.IGNORECASE)


def new_id(prefix: str) -> str:
    """`evt_01JBQ...` / `frm_01JBQ...` — ULID, sortable (spec.md 0)."""
    return f"{prefix}_{ULID()}"


def valid_id(value: str) -> bool:
    return bool(ID_RE.match(value))


def iso(ts: datetime) -> str:
    """ISO 8601 UTC with milliseconds and a `Z` suffix (spec.md 0)."""
    ts = ts.astimezone(timezone.utc)
    return f"{ts.strftime('%Y-%m-%dT%H:%M:%S')}.{ts.microsecond // 1000:03d}Z"


def iso_now() -> str:
    return iso(datetime.now(timezone.utc))


def parse_iso(value: str) -> datetime:
    """Accepts the shapes we hand out plus plain `+00:00` offsets."""
    text = value.strip()
    if text.endswith(("z", "Z")):
        text = f"{text[:-1]}+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def encode_vec(vec: np.ndarray) -> bytes:
    return np.ascontiguousarray(vec, dtype=VEC_DTYPE).tobytes()


def decode_vec(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=VEC_DTYPE)


@dataclass(frozen=True, slots=True)
class FrameRow:
    id: str
    ts: str
    path: str
    thumb_path: str
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class EventRow:
    id: str
    ts: str
    frame_id: str
    summary: str
    objects: list[str]
    confidence: float
    source: str

    def to_json(self) -> dict[str, Any]:
        """The one event shape shared by /api/events and SSE `observed`."""
        return {
            "id": self.id,
            "ts": self.ts,
            "summary": self.summary,
            "objects": self.objects,
            "confidence": self.confidence,
            "source": self.source,
            "thumb_url": f"/api/frames/{self.frame_id}/thumb",
        }


def _event_row(row: sqlite3.Row) -> EventRow:
    return EventRow(
        id=row["id"],
        ts=row["ts"],
        frame_id=row["frame_id"],
        summary=row["summary"],
        objects=json.loads(row["objects"]),
        confidence=row["confidence"],
        source=row["source"],
    )


def apply_pragmas(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")


def connect(path: Path, *, check_same_thread: bool = True) -> sqlite3.Connection:
    conn = sqlite3.connect(
        path, isolation_level=None, check_same_thread=check_same_thread, timeout=5.0
    )
    conn.row_factory = sqlite3.Row
    apply_pragmas(conn)
    return conn


class EmbedDimMismatch(RuntimeError):
    """`meta.embed_dim` disagrees with `--embed-dim` (backend.md 8.3)."""


def create_schema(conn: sqlite3.Connection, *, embed_model: str, embed_dim: int) -> None:
    # executescript() commits any open transaction before it runs, so the DDL
    # stays outside our explicit transaction. CREATE ... IF NOT EXISTS is
    # idempotent, which is all we need here.
    conn.executescript(SCHEMA)
    # Check before the upsert: overwriting meta first would erase the very
    # value we are supposed to validate against and turn the guard into a
    # no-op. Mixed dimensions make scores garbage, so refuse outright.
    row = conn.execute("SELECT value FROM meta WHERE key = 'embed_dim'").fetchone()
    if row is not None and int(row["value"]) != embed_dim:
        raise EmbedDimMismatch(
            f"meta.embed_dim is {row['value']} but --embed-dim is {embed_dim}. "
            "Mixed dimensions produce garbage scores; re-seed the DB instead."
        )
    conn.execute("BEGIN")
    try:
        conn.executemany(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            [
                ("schema_version", str(SCHEMA_VERSION)),
                ("embed_model", embed_model),
                ("embed_dim", str(embed_dim)),
            ],
        )
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


class Database:
    """Async-facing facade. Every method here is safe to call from handlers."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._write = connect(path, check_same_thread=False)
        self._write_lock = asyncio.Lock()
        self._local = threading.local()

    # -- connections ----------------------------------------------------

    def reader(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = connect(self.path)
            self._local.conn = conn
        return conn

    def close(self) -> None:
        self._write.close()

    async def _read(self, fn, /, *args):
        """Reads run inline on the event loop, deliberately.

        `_sqlite3` releases the GIL around every `sqlite3_step`, so dispatching
        sub-millisecond queries to a thread pool costs far more in GIL
        handoffs than the query itself: measured with ab, GET
        /api/events?limit=50 fell from 1558 rps at c=1 to 277 rps at c=32
        (0.46 -> 12.9 ms CPU per request, ~3.7 cores burnt) purely in lock
        convoy. Inline the same endpoint holds ~2450 rps at c=32.

        A read is 0.07-1.1 ms of work (the 200-row page over 20k events is the
        worst case), which is cheaper than one handoff, so blocking the loop
        for it is the correct trade. Writes keep `_mutate`: they are serialized
        by `_write_lock` anyway, and one thread never convoys.
        """
        return fn(*args)

    async def _mutate(self, fn, /, *args):
        async with self._write_lock:
            return await asyncio.to_thread(fn, *args)

    # -- setup ----------------------------------------------------------

    def init_schema(self, *, embed_model: str, embed_dim: int) -> None:
        create_schema(self._write, embed_model=embed_model, embed_dim=embed_dim)

    def get_meta(self, key: str) -> str | None:
        row = self.reader().execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    # -- reads ----------------------------------------------------------

    def _count_events(self) -> int:
        return int(self.reader().execute("SELECT count(*) AS n FROM events").fetchone()["n"])

    async def count_events(self) -> int:
        return await self._read(self._count_events)

    def _list_events(
        self,
        limit: int,
        from_ts: str | None,
        to_ts: str | None,
        cursor: str | None,
        q: str | None,
    ) -> list[EventRow] | None:
        """None means "cursor row does not exist" — caller returns an empty page."""
        conn = self.reader()
        where: list[str] = []
        params: list[Any] = []
        if from_ts is not None:
            where.append("ts >= ?")
            params.append(from_ts)
        if to_ts is not None:
            where.append("ts <= ?")
            params.append(to_ts)
        if q:
            where.append("summary LIKE ? ESCAPE '\\'")
            params.append(f"%{_like_escape(q)}%")
        if cursor is not None:
            row = conn.execute("SELECT ts FROM events WHERE id = ?", (cursor,)).fetchone()
            if row is None:
                return None
            # Sort key is (ts, id) DESC, so "strictly after the cursor" is
            # everything ordered below it — exclusive, never repeating a row.
            where.append("(ts < ? OR (ts = ? AND id < ?))")
            params.extend([row["ts"], row["ts"], cursor])
        sql = "SELECT * FROM events"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY ts DESC, id DESC LIMIT ?"
        params.append(limit)
        return [_event_row(r) for r in conn.execute(sql, params)]

    async def list_events(
        self,
        *,
        limit: int,
        from_ts: str | None = None,
        to_ts: str | None = None,
        cursor: str | None = None,
        q: str | None = None,
    ) -> list[EventRow] | None:
        return await self._read(self._list_events, limit, from_ts, to_ts, cursor, q)

    def _events_by_ids(self, ids: Sequence[str]) -> dict[str, EventRow]:
        if not ids:
            return {}
        placeholders = ",".join("?" * len(ids))
        rows = self.reader().execute(
            f"SELECT * FROM events WHERE id IN ({placeholders})", tuple(ids)
        )
        return {r["id"]: _event_row(r) for r in rows}

    async def events_by_ids(self, ids: Sequence[str]) -> dict[str, EventRow]:
        return await self._read(self._events_by_ids, ids)

    def _get_frame(self, frame_id: str) -> FrameRow | None:
        row = self.reader().execute("SELECT * FROM frames WHERE id = ?", (frame_id,)).fetchone()
        if row is None:
            return None
        return FrameRow(
            id=row["id"],
            ts=row["ts"],
            path=row["path"],
            thumb_path=row["thumb_path"],
            width=row["width"],
            height=row["height"],
        )

    async def get_frame(self, frame_id: str) -> FrameRow | None:
        return await self._read(self._get_frame, frame_id)

    def load_embeddings(self) -> list[tuple[str, int, bytes]]:
        """Startup-only full scan; runs before the server accepts traffic."""
        rows = self.reader().execute(
            "SELECT e.event_id, e.dim, e.vec FROM embeddings e "
            "JOIN events ev ON ev.id = e.event_id ORDER BY ev.ts, ev.id"
        )
        return [(r["event_id"], r["dim"], r["vec"]) for r in rows]

    # -- writes ---------------------------------------------------------

    def _insert_frame(self, frame: FrameRow) -> None:
        self._write.execute("BEGIN")
        try:
            self._write.execute(
                "INSERT INTO frames(id, ts, path, thumb_path, width, height) "
                "VALUES(?, ?, ?, ?, ?, ?)",
                (
                    frame.id,
                    frame.ts,
                    frame.path,
                    frame.thumb_path,
                    frame.width,
                    frame.height,
                ),
            )
        except BaseException:
            self._write.execute("ROLLBACK")
            raise
        self._write.execute("COMMIT")

    async def insert_frame(self, frame: FrameRow) -> None:
        await self._mutate(self._insert_frame, frame)

    def _insert_event(self, event: EventRow, dim: int, blob: bytes | None) -> None:
        self._write.execute("BEGIN")
        try:
            self._write.execute(
                "INSERT INTO events(id, ts, frame_id, summary, objects, confidence, source) "
                "VALUES(?, ?, ?, ?, ?, ?, ?)",
                (
                    event.id,
                    event.ts,
                    event.frame_id,
                    event.summary,
                    json.dumps(event.objects, ensure_ascii=False),
                    event.confidence,
                    event.source,
                ),
            )
            if blob is not None:
                self._write.execute(
                    "INSERT INTO embeddings(event_id, dim, vec) VALUES(?, ?, ?)",
                    (event.id, dim, blob),
                )
        except BaseException:
            self._write.execute("ROLLBACK")
            raise
        self._write.execute("COMMIT")

    async def insert_event(self, event: EventRow, *, dim: int, blob: bytes | None) -> None:
        await self._mutate(self._insert_event, event, dim, blob)

    def _insert_query(
        self, query_id: str, ts: str, question: str, answer: str, cited: str, latency_ms: int
    ) -> None:
        self._write.execute("BEGIN")
        try:
            self._write.execute(
                "INSERT INTO queries(id, ts, question, answer, cited, latency_ms) "
                "VALUES(?, ?, ?, ?, ?, ?)",
                (query_id, ts, question, answer, cited, latency_ms),
            )
        except BaseException:
            self._write.execute("ROLLBACK")
            raise
        self._write.execute("COMMIT")

    async def insert_query(
        self,
        *,
        question: str,
        answer: str,
        cited: Iterable[str],
        latency_ms: int,
    ) -> None:
        await self._mutate(
            self._insert_query,
            new_id("qry"),
            iso_now(),
            question,
            answer,
            json.dumps(list(cited), ensure_ascii=False),
            latency_ms,
        )


def _like_escape(value: str) -> str:
    """`q` is a literal substring, so LIKE wildcards inside it must not match."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
