"""End-to-end sidecar verification against real inference, runnable on macOS.

`--mock-sidecar` proves the API layer works; it proves nothing about the wire
protocol, because the mock never touches a socket. The Orin is the only machine
that runs the real thing, and `docs/sidecar.md` closes with the requirement
that "真 sidecar 換上去之後,backend.md 8.8 的驗收清單必須照樣全過". This
script is how that claim gets checked *before* the models reach the Orin: it
starts `sidecar/server.py --backend mlx`, which loads the same three models
(SmolVLM2 / Qwen2.5-7B / bge-m3, MLX-quantised) on Apple silicon, and drives it
through every clause of the contract.

    python3 scripts/verify_sidecar.py

Checks, in order:

  §3.1 framing      one JSON object per line, `\\n`-terminated, `req_id` echoed
  §3.1 embed        `len(vec) == --embed-dim`, finite, deterministic
  §3.1 describe     one-sentence `summary`, `objects` is a list, `ms` >= 0
  §3.1 errors       an unknown `kind` yields Failed/UNKNOWN_KIND, not a crash
  §3.1 concurrency  the backend's one-in-flight rule is honoured under load
  §3.2 prompt       empty `context` returns the exact refusal sentence
  §3.2 answer       a grounded question is answered from `context` only
  §8.8 refusal      an unwitnessed event is refused (the hard acceptance test)
  client parity     `mneme.sidecar.SocketSidecar` talks to it unmodified
  retrieval         real cosine ranking over the seeded corpus

Exit status is 0 only when every check passes. Two venvs are involved by
design (backend.md 8.2): this script runs in the *main* venv and launches the
sidecar in `sidecar/.venv`, so a torch/MLX import can never leak into the
backend's environment.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from mneme.api.ask import context_line
from mneme.capture import require_cv2
from mneme.search import l2_normalize, load_index
from mneme.sidecar import REFUSAL, SidecarFailed, SocketSidecar

EMBED_DIM = 1024
EMBED_TIMEOUT_MS = 5000
SIDECAR_TIMEOUT_MS = 120_000
"""Deliberately far above the production 20s. A cold 4-bit 7B answer on an M-series
GPU is seconds, not milliseconds, and this script measures latency rather than
enforcing it — a timeout here would report "sidecar broken" for a slow laptop."""

READY_TIMEOUT_S = 900.0
"""Model load, including a first-run download of ~9GB."""


@dataclass
class Report:
    passed: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        if ok:
            self.passed.append(name)
            print(f"  \033[32mPASS\033[0m {name}" + (f" — {detail}" if detail else ""))
        else:
            self.failed.append((name, detail))
            print(f"  \033[31mFAIL\033[0m {name} — {detail}")
        return ok

    def note(self, text: str) -> None:
        self.notes.append(text)
        print(f"  \033[36mnote\033[0m {text}")


def section(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m")


# -- sidecar process -----------------------------------------------------


class SidecarProcess:
    """Launches `sidecar/server.py` in its own venv and waits for readiness.

    Readiness comes over a pipe (`--ready-fd`) rather than a sleep: loading
    three models takes tens of seconds and a fixed sleep is either wasteful or
    flaky. The pipe closes if the process dies, so a crashed load surfaces as
    EOF instead of a hang.
    """

    def __init__(self, socket_path: Path, data_dir: Path, python: Path) -> None:
        self.socket_path = socket_path
        self.data_dir = data_dir
        self.python = python
        self.proc: asyncio.subprocess.Process | None = None

    async def start(self) -> None:
        read_fd, write_fd = os.pipe()
        os.set_inheritable(write_fd, True)
        argv = [
            str(self.python),
            "server.py",
            "--backend",
            "mlx",
            "--socket",
            str(self.socket_path),
            "--data-dir",
            str(self.data_dir),
            "--embed-dim",
            str(EMBED_DIM),
            "--ready-fd",
            str(write_fd),
        ]
        print(f"  launching {' '.join(argv[1:])}")
        self.proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(REPO / "sidecar"),
            pass_fds=(write_fd,),
        )
        os.close(write_fd)
        reader = asyncio.StreamReader()
        loop = asyncio.get_running_loop()
        pipe = os.fdopen(read_fd, "rb", 0)
        await loop.connect_read_pipe(
            lambda: asyncio.StreamReaderProtocol(reader), pipe
        )
        started = time.monotonic()
        line = await asyncio.wait_for(reader.readline(), timeout=READY_TIMEOUT_S)
        if not line:
            raise RuntimeError("sidecar exited before it became ready")
        print(f"  models resident after {time.monotonic() - started:.1f}s")

    async def stop(self) -> None:
        if self.proc is None or self.proc.returncode is not None:
            return
        self.proc.terminate()
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self.proc.wait(), timeout=30)
        if self.proc.returncode is None:
            self.proc.kill()
            await self.proc.wait()


# -- raw wire client -----------------------------------------------------


class RawClient:
    """Speaks the wire by hand, so framing bugs cannot hide behind the real
    client's parsing. Everything §3.1 promises is observable here."""

    def __init__(self, socket_path: Path) -> None:
        self.socket_path = socket_path

    async def __aenter__(self) -> Self:
        self.reader, self.writer = await asyncio.open_unix_connection(
            str(self.socket_path), limit=1 << 20
        )
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.writer.close()
        with contextlib.suppress(OSError):
            await self.writer.wait_closed()

    async def call(self, payload: dict[str, object], timeout: float = 180.0) -> tuple[dict, bytes]:
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        self.writer.write(line.encode("utf-8"))
        await self.writer.drain()
        raw = await asyncio.wait_for(self.reader.readline(), timeout=timeout)
        return json.loads(raw), raw


# -- camera --------------------------------------------------------------


class CameraUnavailable(RuntimeError):
    """No camera, or the OS refused access."""


class CameraCapture:
    """Grabs real frames through the same cv2 path the pipeline uses.

    Deliberately `mneme.capture.require_cv2`, not a bare import: if cv2 is
    broken this must fail with the project's own diagnostic, the same one
    `python -m mneme` prints at startup.

    On macOS the first open triggers a TCC permission prompt. A denied camera
    reports `isOpened() == False` with `not authorized to capture video` on
    stderr, which is a machine setup problem rather than a contract failure —
    hence CameraUnavailable and a skip, not a FAIL.
    """

    WARMUP_FRAMES = 5
    """AVFoundation hands out a few dark frames while auto-exposure settles;
    describing those would test the sensor's warm-up, not the VLM."""

    def __init__(self, device: str) -> None:
        self.cv2 = require_cv2()
        source: int | str = int(device) if device.isdigit() else device
        self.cap = self.cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise CameraUnavailable(
                f"cv2.VideoCapture could not open {device!r}. On macOS grant the "
                "terminal camera access (System Settings > Privacy & Security > "
                "Camera) and re-run; the pipeline's documented fallback is "
                "--camera-cmd (spec.md 7)."
            )
        for _ in range(self.WARMUP_FRAMES):
            self.cap.read()

    @property
    def resolution(self) -> str:
        return (
            f"{int(self.cap.get(self.cv2.CAP_PROP_FRAME_WIDTH))}x"
            f"{int(self.cap.get(self.cv2.CAP_PROP_FRAME_HEIGHT))}"
        )

    def grab(self, data_dir: Path, count: int) -> list[str]:
        """Write `count` frames and return paths relative to --data-dir.

        Relative because that is what a Describe request carries (sidecar.md
        3.1); handing over an absolute path would let a sidecar that forgot to
        join --data-dir pass anyway.
        """
        out_dir = data_dir / "frames"
        out_dir.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []
        for index in range(count):
            if index:
                # Give the scene time to change. The point of two frames is
                # that they differ, so back-to-back reads would be a weaker
                # test than a person simply moving between them.
                print(f"    move or change the scene... capturing {index + 1}/{count}")
                time.sleep(2.0)
            ok, mat = self.cap.read()
            if not ok or mat is None:
                raise CameraUnavailable("camera opened but returned no frame")
            path = out_dir / f"verify_live_{index}.jpg"
            self.cv2.imwrite(str(path), mat)
            paths.append(path.relative_to(data_dir).as_posix())
        return paths

    def close(self) -> None:
        self.cap.release()

    def cleanup(self, data_dir: Path, paths: list[str]) -> None:
        """These frames are verification scratch, not events: nothing was
        written to SQLite for them, so leaving them in data/frames/ would be
        litter that the next `--no-camera` run still serves over HTTP."""
        for rel in paths:
            with contextlib.suppress(OSError):
                (data_dir / rel).unlink()


# -- checks --------------------------------------------------------------


async def check_wire(report: Report, socket_path: Path, frame_rel: str) -> None:
    section("§3.1 wire protocol")
    async with RawClient(socket_path) as client:
        reply, raw = await client.call({"kind": "embed", "req_id": "req-1", "text": "測試"})
        report.check("reply is newline-terminated", raw.endswith(b"\n"))
        report.check(
            "reply carries exactly one JSON object per line",
            raw.count(b"\n") == 1 and isinstance(reply, dict),
        )
        report.check("kind == embedded", reply.get("kind") == "embedded", str(reply.get("kind")))
        report.check("req_id is echoed", reply.get("req_id") == "req-1")
        vec = reply.get("vec")
        report.check(
            f"len(vec) == {EMBED_DIM}",
            isinstance(vec, list) and len(vec) == EMBED_DIM,
            f"got {len(vec) if isinstance(vec, list) else type(vec).__name__}",
        )
        arr = np.asarray(vec, dtype=np.float32)
        report.check(
            "vec is finite (no NaN/inf)",
            bool(np.isfinite(arr).all()),
            "a NaN would silently disable the refusal threshold",
        )
        report.check(
            "raw line contains no bare NaN/Infinity",
            b"NaN" not in raw and b"Infinity" not in raw,
        )
        ms = reply.get("ms")
        report.check("ms is a non-negative int", isinstance(ms, int) and ms >= 0, str(ms))

        # Determinism: sidecar.md calls the mock deterministic, and a real
        # embedding model at temperature 0 must be too, or the same summary
        # would land at two points in the index.
        again, _ = await client.call({"kind": "embed", "req_id": "req-2", "text": "測試"})
        drift = float(np.abs(np.asarray(again["vec"], dtype=np.float32) - arr).max())
        report.check("embed is deterministic", drift == 0.0, f"max abs drift {drift:.2e}")

        reply, _ = await client.call(
            {"kind": "describe", "req_id": "req-3", "image_path": frame_rel}, timeout=300.0
        )
        report.check("kind == described", reply.get("kind") == "described", str(reply)[:120])
        summary = reply.get("summary", "")
        report.check(
            "summary is a non-empty single line",
            isinstance(summary, str) and bool(summary) and "\n" not in summary,
            repr(summary)[:100],
        )
        report.check("objects is a list", isinstance(reply.get("objects"), list),
                     repr(reply.get("objects"))[:80])
        report.note(f"VLM summary: {summary}")
        report.note(f"VLM objects: {reply.get('objects')}")

        # §3.1: an unrecognised kind is a Failed, never a dropped connection.
        reply, _ = await client.call({"kind": "sing", "req_id": "req-4"})
        report.check(
            "unknown kind -> failed/UNKNOWN_KIND",
            reply.get("kind") == "failed" and reply.get("code") == "UNKNOWN_KIND",
            str(reply)[:120],
        )
        report.check("failed reply echoes req_id", reply.get("req_id") == "req-4")

        # The connection has to survive that Failed, or every later request
        # would pay a 2s reconnect.
        reply, _ = await client.call({"kind": "embed", "req_id": "req-5", "text": "還活著"})
        report.check("connection survives a Failed", reply.get("kind") == "embedded")

        # A missing payload field is the sidecar's own BAD_REQUEST, not a stack
        # trace that kills the connection.
        reply, _ = await client.call({"kind": "embed", "req_id": "req-6"})
        report.check(
            "missing field -> failed, connection alive",
            reply.get("kind") == "failed",
            str(reply)[:120],
        )


async def check_one_in_flight(report: Report, socket_path: Path) -> None:
    section("§3.1 concurrency (one in-flight request)")
    async with RawClient(socket_path) as client:
        # Pipeline three requests without reading, then read all three. Replies
        # must come back in order, one per line, with matching req_ids: that is
        # the only observable the protocol promises, and it is what breaks if
        # the sidecar ever answers out of order.
        for i in range(3):
            payload = json.dumps(
                {"kind": "embed", "req_id": f"pipe-{i}", "text": f"併發測試{i}"},
                ensure_ascii=False,
            )
            client.writer.write(payload.encode("utf-8") + b"\n")
        await client.writer.drain()
        ids = []
        for _ in range(3):
            raw = await asyncio.wait_for(client.reader.readline(), timeout=120.0)
            ids.append(json.loads(raw).get("req_id"))
        report.check(
            "pipelined replies stay in order",
            ids == ["pipe-0", "pipe-1", "pipe-2"],
            str(ids),
        )


async def check_live_camera(
    report: Report, sidecar: SocketSidecar, data_dir: Path, capture: CameraCapture
) -> list[str]:
    """Describe frames from the real camera, not the seeded placeholders.

    The seed images are solid colours with "SEED #059" painted on, so a VLM
    that ignored the pixels entirely and emitted a constant sentence would
    still pass every other check in this file. Two live frames with a real
    visual change between them cannot be faked that way: the summaries have to
    differ, and the objects have to name things that are actually in the room.
    """
    section("live camera -> describe (real pixels)")
    frames = capture.grab(data_dir, count=2)
    report.note(f"captured {len(frames)} frames at {capture.resolution}")

    results = []
    for rel in frames:
        started = time.monotonic()
        summary, objects = await sidecar.describe(rel)
        results.append((rel, summary, objects, time.monotonic() - started))
        report.note(f"{rel}: {summary}")
        report.note(f"{rel}: objects={objects}")

    for rel, summary, objects, _ in results:
        report.check(
            f"{Path(rel).name}: summary is one non-empty line",
            bool(summary) and "\n" not in summary,
            repr(summary)[:100],
        )
        report.check(
            f"{Path(rel).name}: summary is Chinese prose",
            # The prompt asks for 一句中文; an English fallback or a bare label
            # dump means the VLM ignored it.
            sum(1 for c in summary if "\u4e00" <= c <= "\u9fff") >= 5,
            repr(summary)[:100],
        )
        report.check(
            f"{Path(rel).name}: objects are short labels",
            isinstance(objects, list)
            and bool(objects)
            and all(isinstance(o, str) and 0 < len(o) <= 24 for o in objects),
            repr(objects)[:100],
        )
        report.check(
            f"{Path(rel).name}: summary is not the seed placeholder",
            "SEED" not in summary,
            repr(summary)[:100],
        )

    # The discriminative check: a constant-output VLM fails here.
    report.check(
        "two different scenes get different summaries",
        results[0][1] != results[1][1],
        f"both said {results[0][1]!r}" if results[0][1] == results[1][1] else "distinct",
    )
    mean_ms = sum(r[3] for r in results) / len(results) * 1000
    report.note(f"describe latency on live frames: {mean_ms:.0f}ms mean")
    report.check(
        "describe stays inside --sidecar-timeout-ms (20000)",
        mean_ms < 20_000,
        f"{mean_ms:.0f}ms",
    )
    return frames


async def check_prompt_contract(
    report: Report, sidecar: SocketSidecar, corpus: list[tuple[str, str]]
) -> None:
    section("§3.2 prompt contract")
    # Empty context must produce the exact refusal sentence. The backend
    # compares strings nowhere, but /api/ask hands this through untouched, so a
    # paraphrase would ship a different promise to the user.
    answer = await sidecar.answer("有人在跳舞嗎", [])
    report.check(
        "empty context -> exact refusal sentence",
        answer == REFUSAL,
        repr(answer),
    )

    # Pin the context to an event that actually mentions the mug: asking
    # about 馬克杯 while feeding an unrelated summary would test refusal, not
    # grounding, and which summary sits at corpus[0] depends on the seed.
    ts, summary = next(
        ((t, s) for t, s in corpus if "馬克杯" in s),
        corpus[0],
    )
    context = [context_line(1, ts, summary)]
    answer = await sidecar.answer("馬克杯放在哪", context)
    report.note(f"grounded context: {context[0]}")
    report.note(f"grounded answer: {answer}")
    report.check(
        "grounded answer is one or two Chinese sentences",
        bool(answer) and "\n" not in answer and len(answer) <= 120,
        repr(answer)[:120],
    )
    report.check(
        "grounded answer is not the refusal",
        answer != REFUSAL,
        "the LLM had the fact in context and must use it",
    )
    report.check(
        "grounded answer stays inside the context",
        any(token in answer for token in ("馬克杯", "桌")),
        repr(answer)[:120],
    )
    report.check(
        "grounded answer does not hedge",
        not any(word in answer for word in ("可能", "大概", "也許", "應該是")),
        repr(answer)[:120],
    )


async def check_retrieval_and_refusal(
    report: Report,
    sidecar: SocketSidecar,
    corpus: list[tuple[str, str]],
    min_score: float,
) -> None:
    section("§8.8 retrieval + hard refusal, on real embeddings")
    started = time.monotonic()
    vectors = []
    for _, summary in corpus:
        vectors.append(l2_normalize(await sidecar.embed(summary)))
    elapsed = time.monotonic() - started
    report.note(
        f"embedded {len(corpus)} summaries in {elapsed:.1f}s "
        f"({elapsed / len(corpus) * 1000:.0f}ms each)"
    )
    index = load_index(
        [(f"evt_{i}", EMBED_DIM, v.tobytes()) for i, v in enumerate(vectors)], EMBED_DIM
    )
    report.check("index accepted every real vector", index.count == len(corpus))

    # Grounded question: the right event has to rank first.
    query = l2_normalize(await sidecar.embed("馬克杯放在哪"))
    hits = index.search(query, 5)
    top_summary = corpus[int(hits[0][0].split("_")[1])][1]
    report.check(
        "retrieval ranks the mug event first",
        "馬克杯" in top_summary,
        f"top={hits[0][1]:.3f} {top_summary}",
    )
    report.check(
        "scores are sorted descending",
        all(hits[i][1] >= hits[i + 1][1] for i in range(len(hits) - 1)),
        ", ".join(f"{s:.3f}" for _, s in hits),
    )
    report.check(
        "floor guard does not eat a grounded question",
        hits[0][1] >= min_score,
        f"{hits[0][1]:.3f} >= {min_score}",
    )
    grounded_top = hits[0][1]

    # The hard acceptance test: something that never happened.
    unseen = "有沒有人在跳舞"
    query = l2_normalize(await sidecar.embed(unseen))
    hits = index.search(query, 5)
    unseen_top = hits[0][1]
    report.note(
        f"real bge-m3 top score: grounded {grounded_top:.3f} vs unwitnessed "
        f"{unseen_top:.3f} (separation {grounded_top - unseen_top:.3f})"
    )

    # spec.md 2.4 (v1.5) names the prompt as the primary refusal path, so this
    # exercises it unconditionally instead of branching on whether the floor
    # guard happened to fire. bge-m3's Chinese cosine has a floor near 0.7 and
    # the witnessed/unwitnessed separation is ~0.1, so --ask-min-score (tuned
    # against the mock's near-zero cosine for unrelated text) will not gate
    # this question -- and even if a future threshold did, the prompt still has
    # to hold on its own. Citations stay attached either way.
    if unseen_top < min_score:
        report.note(
            f"floor guard would also have caught this ({unseen_top:.3f} < {min_score}); "
            "verifying the §3.2 prompt path regardless"
        )
    else:
        report.note(
            f"--ask-min-score {min_score} does not gate real bge-m3 "
            f"(unwitnessed scores {unseen_top:.3f}); refusal comes from §3.2"
        )
    kept = [(eid, s) for eid, s in hits if s >= min_score][:3]
    context = []
    for position, (event_id, _) in enumerate(kept, start=1):
        ts, summary = corpus[int(event_id.split("_")[1])]
        context.append(context_line(position, ts, summary))
    answer = await sidecar.answer(unseen, context)
    report.note(f"answer to an unwitnessed question ({len(context)} cited): {answer}")

    report.check(
        "unwitnessed question is refused, not fabricated",
        REFUSAL in answer or "沒有看到" in answer,
        repr(answer)[:160],
    )
    report.check(
        "refusal does not invent a dancer",
        "跳舞" not in answer.replace(REFUSAL, "") or "沒有" in answer,
        repr(answer)[:160],
    )


async def check_client_parity(
    report: Report, sidecar: SocketSidecar, frame_rel: str
) -> None:
    section("client parity: mneme.sidecar.SocketSidecar, unmodified")
    report.check("client reports sidecar up", sidecar.status == "up", sidecar.status)

    summary, objects = await sidecar.describe(frame_rel)
    report.check(
        "describe() returns (summary, objects)",
        bool(summary) and isinstance(objects, list),
        f"{summary[:40]}... / {objects}",
    )

    vec = await sidecar.embed("有人把馬克杯放到桌子右側")
    report.check(
        "embed() returns the declared dtype and dim",
        vec.dtype == np.float32 and vec.shape == (EMBED_DIM,),
        f"{vec.dtype} {vec.shape}",
    )

    # session() must hold the socket across both RPCs of an ask, without
    # deadlocking on the reentrant _rpc path.
    async with sidecar.session():
        v = await sidecar.embed("鎖測試")
        a = await sidecar.answer("測試", [context_line(1, "2026-09-05T14:03:00.000Z", "測試記錄")])
    report.check(
        "session() holds the lock across embed+answer",
        v.shape == (EMBED_DIM,) and bool(a),
        f"answer={a[:40]}",
    )

    # A bad image path is a Failed, mapped to 502 by api/__init__.py, not a
    # hang and not a 500.
    try:
        await sidecar.describe("frames/does-not-exist.jpg")
    except SidecarFailed as exc:
        report.check("missing image -> SidecarFailed", True, exc.code)
    else:
        report.check("missing image -> SidecarFailed", False, "no exception raised")


def load_corpus(db_path: Path, limit: int) -> list[tuple[str, str]]:
    """(ts, summary) from the seeded DB, newest first."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT ts, summary FROM events ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
    finally:
        conn.close()
    return [(r["ts"], r["summary"]) for r in rows]


def pick_frame(db_path: Path, data_dir: Path) -> str:
    """A frame path relative to --data-dir, as §3.1 specifies."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        for row in conn.execute("SELECT path FROM frames ORDER BY ts DESC"):
            if (data_dir / row["path"]).is_file():
                return row["path"]
    finally:
        conn.close()
    raise SystemExit(
        f"no readable frame under {data_dir}; run "
        "`python -m mneme.seed --out data/memory.db --data-dir ./data "
        "--hours 8 --count 60 --seed 42` first"
    )


async def run(args: argparse.Namespace) -> int:
    data_dir = args.data_dir.resolve()
    db_path = args.db or (data_dir / "memory.db")
    if not db_path.is_file():
        raise SystemExit(f"no database at {db_path}; run python -m mneme.seed first")

    python = REPO / "sidecar" / ".venv" / "bin" / "python"
    if not python.is_file():
        raise SystemExit(
            f"no sidecar venv at {python}\n"
            "  python3 -m venv sidecar/.venv\n"
            "  sidecar/.venv/bin/pip install -r sidecar/requirements-mlx.txt"
        )

    corpus = load_corpus(db_path, args.corpus)
    frame_rel = pick_frame(db_path, data_dir)
    report = Report()

    section("starting sidecar (real MLX inference)")
    process = SidecarProcess(args.socket, data_dir, python)
    await process.start()

    client = SocketSidecar(
        args.socket,
        timeout_ms=SIDECAR_TIMEOUT_MS,
        embed_timeout_ms=SIDECAR_TIMEOUT_MS,
        embed_dim=EMBED_DIM,
    )
    camera: CameraCapture | None = None
    live_frames: list[str] = []
    try:
        await check_wire(report, args.socket, frame_rel)
        await check_one_in_flight(report, args.socket)

        await client.start()
        # The client's reconnect loop polls every RECONNECT_INTERVAL_S.
        for _ in range(40):
            if client.status == "up":
                break
            await asyncio.sleep(0.25)
        await check_client_parity(report, client, frame_rel)

        if args.no_camera:
            print("\n\033[33mskipping the live camera check (--no-camera)\033[0m")
        else:
            try:
                camera = CameraCapture(args.camera)
            except (CameraUnavailable, RuntimeError) as exc:
                # A missing or unauthorised camera is this machine's setup, not
                # a broken contract, so it must not turn into a red FAIL that
                # hides a real one. Loud, and never silently counted as a pass.
                print(f"\n\033[33mlive camera check SKIPPED: {exc}\033[0m")
                report.note(f"live camera check skipped: {exc}")
            else:
                live_frames = await check_live_camera(report, client, data_dir, camera)

        await check_prompt_contract(report, client, corpus)
        await check_retrieval_and_refusal(report, client, corpus, args.ask_min_score)
    finally:
        if camera is not None:
            camera.close()
            if not args.keep_frames:
                camera.cleanup(data_dir, live_frames)
        await client.stop()
        await process.stop()

    section("summary")
    print(f"  {len(report.passed)} passed, {len(report.failed)} failed")
    for name, detail in report.failed:
        print(f"  \033[31m✗\033[0m {name}: {detail}")
    return 1 if report.failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python scripts/verify_sidecar.py")
    parser.add_argument("--data-dir", type=Path, default=REPO / "data")
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--socket", type=Path, default=Path("/tmp/vlm-verify.sock"))
    parser.add_argument(
        "--ask-min-score",
        type=float,
        default=0.35,
        help="the production floor guard from backend.md 8.3, not a semantic judge",
    )
    parser.add_argument(
        "--corpus",
        type=int,
        default=16,
        help="how many seeded summaries to embed for the retrieval check",
    )
    parser.add_argument(
        "--camera",
        default="0",
        help="cv2.VideoCapture index or device path for the live describe check",
    )
    parser.add_argument(
        "--no-camera",
        action="store_true",
        help="skip the live camera check (headless CI, or no camera attached)",
    )
    parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="keep the captured frames under <data-dir>/frames for inspection",
    )
    return asyncio.run(run(parser.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
