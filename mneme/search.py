"""In-memory cosine retrieval. docs/backend.md 8.7.

Every vector is L2-normalised before it lands here, so cosine degenerates to
a dot product and retrieval is a single matmul over a contiguous
``(capacity, dim)`` float32 buffer.

Rows are chronological -- appended as events happen, and ``load_embeddings``
orders by ``ts`` at startup -- so any time range is a *contiguous slice*, found
by bisecting the parallel list of timestamps. That is what lets ``/api/ask``
scope a question to a window without a second index or a scan.

Concurrency: the store appends (single writer), handlers read. Appends write
``buf[count]`` *first* and only then publish the new ``(buf, count)`` tuple,
so a reader either misses the row entirely or sees it fully written. Under the
GIL that tuple swap is atomic — no lock needed.
"""

from __future__ import annotations

import bisect

import numpy as np

INITIAL_CAPACITY = 256


class DimensionMismatch(ValueError):
    """Mixed dimensions are a bug we refuse to tolerate (backend.md 8.7)."""


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """Guard `norm == 0`: an all-zero vector would produce a row of `nan`
    scores and silently disable the refusal threshold."""
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0 or not np.isfinite(norm):
        raise ValueError("cannot L2 normalize a zero or non-finite vector")
    return arr / norm


class SearchIndex:
    def __init__(self, dim: int, capacity: int = INITIAL_CAPACITY) -> None:
        self.dim = dim
        self._ids: list[str] = []
        self._ts: list[str] = []  # ISO 8601 UTC, sorted; parallel to _ids
        self._state: tuple[np.ndarray, int] = (
            np.zeros((max(capacity, 1), dim), dtype=np.float32),
            0,
        )

    def __len__(self) -> int:
        return self._state[1]

    @property
    def count(self) -> int:
        return self._state[1]

    def add(self, event_id: str, vec: np.ndarray, ts: str = "") -> None:
        """`vec` must already be normalised and of the right dimension.

        `ts` is the event's UTC ISO 8601 timestamp. It is only used to find
        time windows, and callers that never ask for one may leave it empty --
        an empty string sorts before every real timestamp, so it lands outside
        any window rather than inside the wrong one.
        """
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        if arr.shape[0] != self.dim:
            raise DimensionMismatch(f"expected dim {self.dim}, got {arr.shape[0]}")
        buf, count = self._state
        if count == buf.shape[0]:
            grown = np.zeros((buf.shape[0] * 2, self.dim), dtype=np.float32)
            grown[:count] = buf[:count]
            buf = grown
        buf[count] = arr
        self._ids.append(event_id)
        self._ts.append(ts)
        self._state = (buf, count + 1)

    def rows_between(self, since: str | None, until: str | None) -> tuple[int, int]:
        """`[lo, hi)` row range covering a UTC ISO 8601 window; both ends optional.

        `until` is inclusive of the whole second it names -- a range that ends
        at 14:30 should contain 14:30:59, because nobody means otherwise -- so
        it bisects on the timestamp plus a suffix that sorts after any
        fractional part.

        ISO 8601 UTC sorts lexicographically, which is why this is a bisect on
        strings and not a parse of a few thousand timestamps per question.
        """
        count = self._state[1]
        stamps = self._ts[:count]
        lo = 0 if since is None else bisect.bisect_left(stamps, since)
        hi = count if until is None else bisect.bisect_right(stamps, until + "\uffff")
        return lo, max(lo, hi)

    def search(
        self,
        query: np.ndarray,
        top_k: int,
        *,
        recent: int | None = None,
        rows: tuple[int, int] | None = None,
    ) -> list[tuple[str, float]]:
        """Returns `(event_id, cosine)` sorted by score descending.

        `rows` restricts the search to a `[lo, hi)` row range (from
        `rows_between`); `recent` further keeps only the newest N of whatever
        is left. Both are slices of a chronological buffer, so a scoped search
        costs less than an unscoped one rather than more.

        An empty index or an empty range short-circuits: the caller's refusal
        path handles it and numpy must not see a zero-row matmul.
        """
        buf, count = self._state
        if count == 0 or top_k <= 0:
            return []
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.dim:
            raise DimensionMismatch(f"expected dim {self.dim}, got {q.shape[0]}")
        lo, hi = (0, count) if rows is None else rows
        lo, hi = max(0, lo), min(count, hi)
        if recent is not None:
            lo = max(lo, hi - max(recent, 1))
        window = hi - lo
        if window <= 0:
            return []
        start = lo
        scores = buf[start:hi] @ q
        k = min(top_k, window)
        # kth must be clamped: argpartition(-scores, top_k) raises
        # "kth out of bounds" whenever count <= top_k, which is exactly the
        # freshly-seeded / near-empty case.
        candidates = np.argpartition(-scores, k - 1)[:k]
        # Ties break towards the newest row. Two frames that match a question
        # equally well are not equally good answers to it -- "where is my mug"
        # means where it is now -- and rows are chronological, so a higher index
        # is a later moment. lexsort takes its primary key last.
        ordered = candidates[np.lexsort((-candidates, -scores[candidates]))]
        # float(...) so json never sees a np.float32 (not serializable).
        return [
            (self._ids[start + int(i)], float(scores[int(i)])) for i in ordered
        ]


def load_index(rows: list[tuple[str, int, bytes, str]], dim: int) -> SearchIndex:
    """Build the index from `(event_id, dim, blob, ts)` rows read at startup.

    The rows arrive ordered by `ts` (db.load_embeddings), which is what makes
    `rows_between` a bisect rather than a scan.
    """
    from .db import decode_vec

    index = SearchIndex(dim, capacity=max(len(rows), INITIAL_CAPACITY))
    for event_id, row_dim, blob, ts in rows:
        if row_dim != dim:
            raise DimensionMismatch(
                f"event {event_id} has embedding dim {row_dim}, expected {dim}"
            )
        vec = decode_vec(blob)
        if vec.shape[0] != dim:
            raise DimensionMismatch(
                f"event {event_id} blob holds {vec.shape[0]} floats, expected {dim}"
            )
        index.add(event_id, vec, ts)
    return index
