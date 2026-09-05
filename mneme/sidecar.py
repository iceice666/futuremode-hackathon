"""Inference sidecar client. docs/sidecar.md 3.1 (wire) and 8.5 (mock).

The real sidecar is a separate process with its own venv and CUDA. We speak
line-delimited JSON over a unix socket: one connection, one in-flight request
at a time, reconnect every 2 seconds after a drop. ``req_id`` exists for log
correlation, not multiplexing.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import hashlib
import json
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from ulid import ULID

log = logging.getLogger(__name__)

RECONNECT_INTERVAL_S = 2.0
READ_LIMIT = 1 << 20
"""1MiB per line. 1024-dim vec is ~22KB, but a bigger embedding model must
widen the stream limit rather than change the protocol (sidecar.md 3.1)."""

VLM_MODEL = "SmolVLM2-2.2B-Instruct"
LLM_MODEL = "qwen2.5-7b-instruct-q4"
EMBED_MODEL = "bge-m3"

REFUSAL = "我沒有看到相關的畫面。"
"""The fixed refusal sentence (spec.md 2.4 / sidecar.md 3.2)."""

_IN_SESSION: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "sidecar_session", default=False
)
"""Set while `Sidecar.session()` holds the RPC lock for the current task.

A ContextVar, not an attribute: each request runs in its own asyncio task and
therefore its own context copy, so one handler's session can never be mistaken
for another's.
"""


class SidecarUnavailable(RuntimeError):
    """No connection -> 503 SIDECAR_UNAVAILABLE."""


class SidecarTimeout(TimeoutError):
    """No answer within the deadline -> 504 SIDECAR_TIMEOUT."""


class SidecarFailed(RuntimeError):
    """`{"kind":"failed"}` -> 502 SIDECAR_FAILED."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@runtime_checkable
class Sidecar(Protocol):
    """Both the socket client and the mock implement exactly this."""

    @property
    def status(self) -> str:
        """`up` | `down` | `mock` (spec.md 2.1)."""

    @property
    def vlm_model(self) -> str: ...

    @property
    def llm_model(self) -> str: ...

    @property
    def embed_model(self) -> str: ...

    async def describe(self, image_path: str) -> tuple[str, list[str]]: ...

    async def embed(self, text: str) -> np.ndarray: ...

    async def answer(self, question: str, context: list[str]) -> str: ...

    def session(self) -> contextlib.AbstractAsyncContextManager[Sidecar]:
        """Reserve the sidecar for one logical request's whole RPC sequence."""

    async def start(self) -> None: ...

    async def stop(self) -> None: ...


class SocketSidecar:
    """Unix-socket client with a background reconnect loop."""

    def __init__(
        self,
        socket_path: Path,
        *,
        timeout_ms: int,
        embed_timeout_ms: int,
        embed_dim: int,
    ) -> None:
        self.socket_path = socket_path
        self.timeout_s = timeout_ms / 1000.0
        self.embed_timeout_s = embed_timeout_ms / 1000.0
        self.embed_dim = embed_dim
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        # One in-flight request: the lock *is* the concurrency contract.
        self._rpc_lock = asyncio.Lock()
        self._connect_task: asyncio.Task[None] | None = None
        self._closing = False

    # -- lifecycle ------------------------------------------------------

    @property
    def status(self) -> str:
        return "up" if self._writer is not None else "down"

    @property
    def vlm_model(self) -> str:
        return VLM_MODEL

    @property
    def llm_model(self) -> str:
        return LLM_MODEL

    @property
    def embed_model(self) -> str:
        return EMBED_MODEL

    async def start(self) -> None:
        self._closing = False
        self._connect_task = asyncio.create_task(self._reconnect_loop(), name="sidecar-connect")

    async def stop(self) -> None:
        self._closing = True
        if self._connect_task is not None:
            self._connect_task.cancel()
            try:
                await self._connect_task
            except asyncio.CancelledError:
                pass
            self._connect_task = None
        await self._drop_connection()

    async def _reconnect_loop(self) -> None:
        while not self._closing:
            if self._writer is None:
                try:
                    reader, writer = await asyncio.open_unix_connection(
                        str(self.socket_path), limit=READ_LIMIT
                    )
                except (OSError, asyncio.TimeoutError) as exc:
                    log.debug("sidecar connect failed: %s", exc)
                else:
                    self._reader, self._writer = reader, writer
                    log.info("sidecar connected at %s", self.socket_path)
            await asyncio.sleep(RECONNECT_INTERVAL_S)

    async def _drop_connection(self) -> None:
        writer, self._writer, self._reader = self._writer, None, None
        if writer is None:
            return
        writer.close()
        try:
            await writer.wait_closed()
        except (OSError, asyncio.CancelledError):
            pass

    # -- rpc ------------------------------------------------------------

    @contextlib.asynccontextmanager
    async def session(self) -> AsyncIterator[SocketSidecar]:
        """Hold the RPC lock across every call made inside the block.

        /api/ask issues two RPCs (embed, then answer). Acquiring the lock
        separately for each lets the capture pipeline slip a ~800ms describe
        into the gap, which measured as +708ms on the ask's embed wait. Taking
        the lock once removes that interleaving: ask latency under a saturated
        pipeline went 2.36s -> 1.64s, against a 1.56s floor of pure inference,
        with pipeline throughput unchanged at 1.15 events/s.

        Still exactly one in-flight request, so the wire protocol in
        sidecar.md 3.1 is unchanged; the sidecar cannot tell the difference.
        """
        async with self._rpc_lock:
            token = _IN_SESSION.set(True)
            try:
                yield self
            finally:
                _IN_SESSION.reset(token)

    async def _rpc(self, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        if _IN_SESSION.get():
            # `session()` already holds the lock for this whole exchange.
            reply = await self._exchange(payload, timeout_s)
        else:
            async with self._rpc_lock:
                reply = await self._exchange(payload, timeout_s)
        if reply.get("kind") == "failed":
            raise SidecarFailed(
                str(reply.get("code", "UNKNOWN")), str(reply.get("message", ""))
            )
        if reply.get("req_id") != payload["req_id"]:
            raise SidecarFailed(
                "REQ_ID_MISMATCH", f"expected {payload['req_id']}, got {reply.get('req_id')}"
            )
        return reply

    async def _exchange(self, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        """One request/response on the wire. The caller owns the lock."""
        reader, writer = self._reader, self._writer
        if reader is None or writer is None:
            raise SidecarUnavailable(f"sidecar not connected at {self.socket_path}")
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        try:
            writer.write(line.encode("utf-8"))
            await writer.drain()
            raw = await asyncio.wait_for(reader.readline(), timeout=timeout_s)
        except asyncio.TimeoutError as exc:
            # A hung sidecar poisons the stream framing: drop the socket
            # so the next request starts from a clean connection.
            await self._drop_connection()
            raise SidecarTimeout(f"sidecar timed out after {timeout_s:.1f}s") from exc
        except (OSError, ConnectionError) as exc:
            await self._drop_connection()
            raise SidecarUnavailable(f"sidecar connection lost: {exc}") from exc
        except ValueError as exc:
            # readline() raises ValueError when READ_LIMIT is reached
            # before a newline. The unread tail of that reply is still
            # queued, so every later read would resync mid-line: the
            # framing is poisoned and the socket has to go.
            await self._drop_connection()
            raise SidecarFailed(
                "OVERSIZED_REPLY",
                f"reply exceeded the {READ_LIMIT} byte line limit: {exc}",
            ) from exc
        if not raw:
            await self._drop_connection()
            raise SidecarUnavailable("sidecar closed the connection")
        try:
            reply = json.loads(raw)
        except json.JSONDecodeError as exc:
            await self._drop_connection()
            raise SidecarFailed("BAD_JSON", f"unparseable reply: {exc}") from exc
        if not isinstance(reply, dict):
            # Valid JSON but not an object: every check in _rpc is a `.get`,
            # so this would surface as AttributeError -> 500. The line was
            # framed correctly, so the connection survives.
            raise SidecarFailed(
                "BAD_REPLY", f"expected a JSON object, got {type(reply).__name__}"
            )
        return reply

    async def describe(self, image_path: str) -> tuple[str, list[str]]:
        reply = await self._rpc(
            {"kind": "describe", "req_id": str(ULID()), "image_path": image_path},
            self.timeout_s,
        )
        _expect(reply, "described")
        summary = _require(reply, "summary")
        objects = reply.get("objects", [])
        if not isinstance(objects, list):
            raise SidecarFailed(
                "BAD_REPLY", f"objects must be a list, got {type(objects).__name__}"
            )
        return str(summary), [str(o) for o in objects]

    async def embed(self, text: str) -> np.ndarray:
        reply = await self._rpc(
            {"kind": "embed", "req_id": str(ULID()), "text": text}, self.embed_timeout_s
        )
        _expect(reply, "embedded")
        try:
            vec = np.asarray(_require(reply, "vec"), dtype=np.float32)
        except (TypeError, ValueError) as exc:
            # A ragged or non-numeric `vec` is the sidecar's bug, not ours.
            raise SidecarFailed("BAD_REPLY", f"vec is not a float array: {exc}") from exc
        if vec.ndim != 1 or vec.size == 0 or not np.isfinite(vec).all():
            # A vector with a NaN would poison every cosine score it touches
            # and silently disable the refusal threshold, so refuse it here.
            raise SidecarFailed(
                "BAD_REPLY",
                f"vec must be a non-empty finite 1-D array, got shape {vec.shape}",
            )
        if vec.size != self.embed_dim:
            # sidecar.md 3.1: len(vec) == --embed-dim. Otherwise the handler
            # would hand it to SearchIndex.search, whose DimensionMismatch is
            # raised outside ask.py's guard -> 500 with no `queries` row.
            raise SidecarFailed(
                "BAD_REPLY", f"vec has {vec.size} dims, expected {self.embed_dim}"
            )
        return vec

    async def answer(self, question: str, context: list[str]) -> str:
        reply = await self._rpc(
            {"kind": "answer", "req_id": str(ULID()), "question": question, "context": context},
            self.timeout_s,
        )
        _expect(reply, "answered")
        return str(_require(reply, "answer"))


def _expect(reply: dict[str, Any], kind: str) -> None:
    if reply.get("kind") != kind:
        raise SidecarFailed("UNEXPECTED_KIND", f"expected {kind}, got {reply.get('kind')!r}")


def _require(reply: dict[str, Any], field: str) -> Any:
    """A reply of the right `kind` still has to carry its payload. Missing it
    would be a bare KeyError, i.e. a 500 outside the spec 2.6 taxonomy."""
    if field not in reply:
        raise SidecarFailed("BAD_REPLY", f"{reply.get('kind')} reply has no {field!r} field")
    return reply[field]


MOCK_SENTENCES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("桌面沒有變化,只有窗外光線變暗", ("desk", "window")),
    ("有人把馬克杯放到桌子右側", ("person", "mug", "desk")),
    ("一個人把黑色後背包放在桌子左側,然後離開", ("person", "backpack", "desk")),
    ("桌上有一條白色充電線,旁邊放著一個馬克杯", ("cable", "mug", "desk")),
    ("有人坐到椅子上開始使用筆記型電腦", ("person", "chair", "laptop")),
    ("門被打開,一個人走進房間", ("person", "door")),
    ("桌上的水瓶被拿走,只剩下鍵盤和滑鼠", ("bottle", "keyboard", "mouse")),
    ("房間裡沒有人,燈是關著的", ("room",)),
)


def token_vector(token: str, dim: int) -> np.ndarray:
    """The primitive from sidecar.md 8.5, verbatim: sha256 -> rng -> gaussian.

    Never the builtin `hash()`: it is per-process seeded, so the vector would
    differ between runs and machines.
    """
    seed = int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:8], "big")
    return np.random.default_rng(seed).standard_normal(dim, dtype=np.float32)


PUNCTUATION = "。,,、??!!::「」()()"


def mock_tokens(text: str) -> list[str]:
    """Character bigrams. Chinese has no spaces, and a bigram is the cheapest
    unit that still carries a word: 馬克杯 -> 馬克, 克杯.

    Deliberately no single characters: filler like 有 / 人 / 在 appears in
    almost every sentence and pushes unrelated pairs over --ask-min-score,
    which would break the refusal test in backend.md 8.8.
    """
    chars = [c for c in text.strip() if not c.isspace() and c not in PUNCTUATION]
    if len(chars) < 2:
        return chars
    return [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]


def mock_embed(text: str, dim: int) -> np.ndarray:
    """Deterministic bag-of-bigrams embedding (sidecar.md 8.5).

    Hashing the *whole* sentence would make every string orthogonal, so
    cosine would be ~0 for every pair and retrieval, ranking and the refusal
    threshold could not be exercised at all — which is the entire stated
    purpose of the mock. Summing per-token vectors instead turns lexical
    overlap into cosine: identical text scores 1.0, shared words score high,
    unrelated text lands near 0 and correctly falls below --ask-min-score.
    Same text -> same vector, on any machine and any Python version.
    """
    tokens = mock_tokens(text)
    if not tokens:
        return token_vector(text, dim)
    acc = np.zeros(dim, dtype=np.float32)
    for token in tokens:
        acc += token_vector(token, dim)
    norm = float(np.linalg.norm(acc))
    if norm == 0.0:
        return token_vector(text, dim)
    return acc / norm


class MockSidecar:
    """In-process fake so the whole API runs on a machine without CUDA.

    Not a demo path: `/api/health` reports `sidecar: "mock"` verbatim.
    """

    def __init__(self, embed_dim: int, data_dir: Path | None = None) -> None:
        self.embed_dim = embed_dim
        self.data_dir = data_dir

    @property
    def status(self) -> str:
        return "mock"

    @property
    def vlm_model(self) -> str:
        return "mock-vlm"

    @property
    def llm_model(self) -> str:
        return "mock-llm"

    @property
    def embed_model(self) -> str:
        return "mock-embed"

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    @contextlib.asynccontextmanager
    async def session(self) -> AsyncIterator[MockSidecar]:
        """Nothing to reserve: the mock runs in-process with no shared socket,
        so there is no lock a pipeline call could contend for."""
        yield self

    async def describe(self, image_path: str) -> tuple[str, list[str]]:
        index = await asyncio.to_thread(_mock_sentence_index, image_path, self.data_dir)
        summary, objects = MOCK_SENTENCES[index]
        await asyncio.sleep(0.3)  # mirrors the reported ms: 300
        return summary, list(objects)

    async def embed(self, text: str) -> np.ndarray:
        return mock_embed(text, self.embed_dim)

    async def answer(self, question: str, context: list[str]) -> str:
        if not context:
            return REFUSAL
        head = context[0]
        # "[1] 2026-09-05 22:03(台北時間) <summary>" -> time + summary
        body = head.split("] ", 1)[-1]
        when, _, summary = body.partition(")")
        return f"最後一次是{when.strip()}),{summary.strip()}。"


def _mock_sentence_index(image_path: str, data_dir: Path | None) -> int:
    """Grayscale mean picks the sentence, mirroring sidecar.md 8.5.

    `image_path` is relative to --data-dir (sidecar.md 3.1), so it must be
    joined here exactly as the real sidecar does; resolving it against the
    process cwd would always miss and silently degrade to the fallback.
    Unreadable images (a seeded DB whose frames were pruned) still get a
    deterministic sentence from the path digest.
    """
    candidate = str(data_dir / image_path) if data_dir is not None else image_path
    try:
        import cv2

        img = cv2.imread(candidate, cv2.IMREAD_GRAYSCALE)
    except (ImportError, OSError) as exc:
        log.debug("mock describe could not read %s: %s", candidate, exc)
    else:
        if img is not None:
            return int(img.mean()) % len(MOCK_SENTENCES)
        log.debug("mock describe found no image at %s", candidate)
    digest = hashlib.sha256(image_path.encode("utf-8")).digest()
    return digest[0] % len(MOCK_SENTENCES)


def build_sidecar(config, embed_timeout_ms: int) -> Sidecar:
    if config.mock_sidecar:
        return MockSidecar(config.embed_dim, config.data_dir)
    return SocketSidecar(
        config.sidecar,
        timeout_ms=config.sidecar_timeout_ms,
        embed_timeout_ms=embed_timeout_ms,
        embed_dim=config.embed_dim,
    )
