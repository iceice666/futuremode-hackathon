"""In-memory cosine retrieval. docs/backend.md 8.7.

Every vector is L2-normalised before it lands here, so cosine degenerates to
a dot product and retrieval is a single matmul over a contiguous
``(capacity, dim)`` float32 buffer.

Concurrency: the store appends (single writer), handlers read. Appends write
``buf[count]`` *first* and only then publish the new ``(buf, count)`` tuple,
so a reader either misses the row entirely or sees it fully written. Under the
GIL that tuple swap is atomic — no lock needed.
"""

from __future__ import annotations

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
        self._state: tuple[np.ndarray, int] = (
            np.zeros((max(capacity, 1), dim), dtype=np.float32),
            0,
        )

    def __len__(self) -> int:
        return self._state[1]

    @property
    def count(self) -> int:
        return self._state[1]

    def add(self, event_id: str, vec: np.ndarray) -> None:
        """`vec` must already be normalised and of the right dimension."""
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
        self._state = (buf, count + 1)

    def search(self, query: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        """Returns `(event_id, cosine)` sorted by score descending.

        Empty index short-circuits: the caller's refusal path handles it and
        numpy must not see a zero-row matmul.
        """
        buf, count = self._state
        if count == 0 or top_k <= 0:
            return []
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.dim:
            raise DimensionMismatch(f"expected dim {self.dim}, got {q.shape[0]}")
        scores = buf[:count] @ q
        k = min(top_k, count)
        # kth must be clamped: argpartition(-scores, top_k) raises
        # "kth out of bounds" whenever count <= top_k, which is exactly the
        # freshly-seeded / near-empty case.
        candidates = np.argpartition(-scores, k - 1)[:k]
        ordered = candidates[np.argsort(-scores[candidates], kind="stable")]
        # float(...) so json never sees a np.float32 (not serializable).
        return [(self._ids[int(i)], float(scores[int(i)])) for i in ordered]


def load_index(rows: list[tuple[str, int, bytes]], dim: int) -> SearchIndex:
    """Build the index from `(event_id, dim, blob)` rows read at startup."""
    from .db import decode_vec

    index = SearchIndex(dim, capacity=max(len(rows), INITIAL_CAPACITY))
    for event_id, row_dim, blob in rows:
        if row_dim != dim:
            raise DimensionMismatch(
                f"event {event_id} has embedding dim {row_dim}, expected {dim}"
            )
        vec = decode_vec(blob)
        if vec.shape[0] != dim:
            raise DimensionMismatch(
                f"event {event_id} blob holds {vec.shape[0]} floats, expected {dim}"
            )
        index.add(event_id, vec)
    return index
