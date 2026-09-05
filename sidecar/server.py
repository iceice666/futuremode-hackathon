"""Inference sidecar. Implements docs/sidecar.md 3.1 over a unix socket.

Two backends, one wire protocol:

* ``--backend mlx``   — real VLM / LLM / embedding inference through MLX
  (Apple silicon). This is what makes the protocol verifiable on a Mac.
* ``--backend cuda``  — the Jetson Orin path (transformers + torch).

The backend only has to provide three synchronous callables (describe, embed,
answer). Everything about the wire — framing, `req_id` echo, the `Failed`
taxonomy, the refusal sentence — lives here so both backends cannot drift.

Why threads: MLX and torch are synchronous and hold one GPU. Running them on
the event loop would block the socket reader, so every inference goes through
``asyncio.to_thread``. sidecar.md 3.1 promises the backend only ever has one
in-flight request, but a *misbehaving* client must not be able to run two
generations on the same model concurrently, so a lock serialises them anyway.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import signal
import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from prompts import ANSWER_SYSTEM, DESCRIBE_PROMPT, OBJECTS_PROMPT, REFUSAL

log = logging.getLogger("sidecar")

READ_LIMIT = 1 << 20
"""Must match `mneme.sidecar.READ_LIMIT`: the backend drops the connection on
an oversized line, so writing one is unrecoverable, not merely rejected."""

MAX_SUMMARY_CHARS = 120
MAX_OBJECTS = 6

DEFAULT_MLX_VLM = "mlx-community/SmolVLM2-2.2B-Instruct-mlx"
DEFAULT_MLX_LLM = "mlx-community/Qwen2.5-7B-Instruct-4bit"
DEFAULT_MLX_EMBED = "mlx-community/bge-m3-mlx-4bit"

DEFAULT_CUDA_VLM = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
DEFAULT_CUDA_LLM = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_CUDA_EMBED = "BAAI/bge-m3"


class SidecarError(Exception):
    """Turned into `{"kind":"failed"}` with this `code` (sidecar.md 3.1)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class Backend(Protocol):
    """Three synchronous inference calls. Called on a worker thread."""

    vlm_model: str
    llm_model: str
    embed_model: str
    embed_dim: int

    def describe(self, image_path: Path) -> tuple[str, list[str]]: ...

    def embed(self, text: str) -> np.ndarray: ...

    def answer(self, question: str, context: list[str]) -> str: ...


# -- text post-processing ------------------------------------------------


LIST_MARKERS = "-*•0123456789.、) "
"""Character *set* stripped from the head of a line, not a prefix: models open
list items as "- ", "1. " or "1) " interchangeably."""

PREFIX_TAIL = ",,::的 "
"""Likewise a character set: whatever punctuation follows a dropped "圖片中"."""


def one_sentence(text: str) -> str:
    """`Described.summary` and `Answered.answer` go into SQLite and the HTTP
    response unmodified (sidecar.md 3.2), so trimming the model's habits has to
    happen here: markdown bullets, a leading "圖片中", a second paragraph.

    Newlines matter most. They are legal inside a JSON string, so nothing
    downstream rejects them, but a summary containing one renders as a broken
    timeline row and a two-paragraph answer is no longer "一到兩句".
    """
    lines = [line.strip().lstrip(LIST_MARKERS) for line in text.splitlines()]
    # A bulleted or multi-paragraph reply: keep the first non-empty line. The
    # contract asks for one sentence, so joining the rest with spaces would
    # only smuggle the list back in.
    flat = " ".join(next((line for line in lines if line), "").split())
    for prefix in ("這張圖片中", "這張圖片", "圖片中", "畫面中", "圖中", "照片中"):
        if flat.startswith(prefix):
            flat = flat[len(prefix) :].lstrip(PREFIX_TAIL)
            break
    if len(flat) > MAX_SUMMARY_CHARS:
        # Cut on a sentence boundary when there is one inside the budget.
        head = flat[:MAX_SUMMARY_CHARS]
        cut = max(head.rfind(c) for c in "。!?.!?")
        flat = head[: cut + 1] if cut > 0 else head
    return flat.strip()


OBJECT_PREAMBLE = (":", ":")


def parse_objects(text: str) -> list[str]:
    """`objects` is a list of short labels, so a model that answers with a
    sentence or a numbered list must still produce a clean list."""
    raw = text.replace("、", ",").replace(",", ",").replace("\n", ",")
    out: list[str] = []
    for chunk in raw.split(","):
        label = chunk.strip().strip("-*•.0123456789 ").lower()
        for sep in OBJECT_PREAMBLE:
            # "the objects are: person" — drop the lead-in, keep the label.
            if sep in label:
                label = label.rsplit(sep, 1)[-1].strip()
        if not label or label in out or len(label) > 24:
            continue
        out.append(label)
        if len(out) >= MAX_OBJECTS:
            break
    return out


# -- MLX backend ---------------------------------------------------------


class MLXBackend:
    """Apple-silicon inference. Same three models as the Orin, MLX-quantised.

    Loading is eager on purpose: a 4-bit 7B LLM takes tens of seconds to
    materialise, and paying that on the first `/api/ask` would blow the
    backend's 20s `--sidecar-timeout-ms` and look like a hung sidecar.
    """

    def __init__(self, *, vlm: str, llm: str, embed: str) -> None:
        import mlx_embeddings
        import mlx_lm
        import mlx_vlm

        self._mlx_vlm = mlx_vlm
        self._mlx_lm = mlx_lm
        self._mlx_embeddings = mlx_embeddings

        self.vlm_model = vlm
        self.llm_model = llm
        self.embed_model = embed

        log.info("loading embedding model %s", embed)
        self._embed_model, self._embed_tok = mlx_embeddings.load(embed)
        self.embed_dim = int(self._probe_embed_dim())
        log.info("embedding dim = %d", self.embed_dim)

        log.info("loading VLM %s", vlm)
        from mlx_vlm.utils import load_config

        self._vlm, self._vlm_processor = mlx_vlm.load(vlm)
        self._vlm_config = load_config(vlm)

        log.info("loading LLM %s", llm)
        self._llm, self._llm_tok = mlx_lm.load(llm)
        log.info("all models resident")

    def _probe_embed_dim(self) -> int:
        """The dimension is a startup fact the backend refuses to guess: it
        compares this against `--embed-dim` and aborts on a mismatch, which is
        far better than shipping a table of wrong-width vectors."""
        vec = self._encode("維度探測")
        return vec.shape[-1]

    def _encode(self, text: str) -> np.ndarray:
        out = self._mlx_embeddings.generate(self._embed_model, self._embed_tok, [text])
        # bge-m3 exposes the pooled, already-normalised sentence vector as
        # `text_embeds`; `last_hidden_state` would need pooling by hand.
        embeds = getattr(out, "text_embeds", None)
        if embeds is None:
            raise SidecarError("EMBED_FAILED", "model returned no text_embeds")
        return np.asarray(embeds, dtype=np.float32).reshape(-1)

    def embed(self, text: str) -> np.ndarray:
        return self._encode(text)

    def describe(self, image_path: Path) -> tuple[str, list[str]]:
        summary = one_sentence(self._vlm_generate(image_path, DESCRIBE_PROMPT, 96))
        if not summary:
            raise SidecarError("DESCRIBE_EMPTY", "VLM produced an empty summary")
        objects = parse_objects(self._vlm_generate(image_path, OBJECTS_PROMPT, 48))
        return summary, objects

    def _vlm_generate(self, image_path: Path, instruction: str, max_tokens: int) -> str:
        prompt = self._mlx_vlm.apply_chat_template(
            self._vlm_processor, self._vlm_config, instruction, num_images=1
        )
        result = self._mlx_vlm.generate(
            self._vlm,
            self._vlm_processor,
            prompt,
            image=[str(image_path)],
            max_tokens=max_tokens,
            temperature=0.0,
            verbose=False,
        )
        return getattr(result, "text", str(result))

    def answer(self, question: str, context: list[str]) -> str:
        if not context:
            return REFUSAL
        joined = "\n".join(context)
        messages = [
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": f"觀察記錄:\n{joined}\n\n問題:{question}"},
        ]
        prompt = self._llm_tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        text = self._mlx_lm.generate(
            self._llm, self._llm_tok, prompt, max_tokens=160, verbose=False
        )
        answer = one_sentence(text)
        return answer or REFUSAL


# -- CUDA backend --------------------------------------------------------


class CudaBackend:
    """Jetson Orin path: transformers + torch, same three models unquantised.

    Kept in the same file as MLXBackend so the wire handling below has exactly
    one implementation. Never imported on a Mac — torch is not in that venv.
    """

    def __init__(self, *, vlm: str, llm: str, embed: str, quantize: bool = False) -> None:
        import torch
        from transformers import (
            AutoModel,
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoProcessor,
            AutoTokenizer,
        )

        self._torch = torch
        self.vlm_model = vlm
        self.llm_model = llm
        self.embed_model = embed
        dtype = torch.float16

        # Jetson Orin has 15.6GB of memory shared between CPU and GPU, so the three
        # fp16 models (~21.6GB together) do not fit. NF4 brings that to roughly a
        # third. bitsandbytes must be one built for sm_87 -- the PyPI aarch64 wheel
        # ships kernels for datacenter ARM only and dies with "named symbol not
        # found" here.
        quant_config = None
        if quantize:
            from transformers import BitsAndBytesConfig

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        # transformers renamed torch_dtype -> dtype in v5. The Orin is pinned to
        # 4.51 because v5 rejects JetPack's torch 2.5, so ask this install which
        # spelling it takes rather than hardcoding one and breaking the other.
        import transformers

        dtype_kw = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"

        def load(factory, name: str, *, quantized: bool = True):
            """A 4-bit model is placed by device_map during construction; calling
            .to() on it afterwards is an error, hence the two paths."""
            kwargs: dict[str, Any] = {dtype_kw: dtype}
            if quant_config is None or not quantized:
                return factory.from_pretrained(name, **kwargs).to("cuda").eval()
            return factory.from_pretrained(
                name, quantization_config=quant_config, device_map="cuda", **kwargs
            ).eval()

        # The embedding model stays fp16 even under --quantize: it is only ~2.3GB,
        # so 4-bit saves little, and retrieval quality is what the whole product
        # rests on -- not the place to spend accuracy for memory.
        log.info("loading embedding model %s (fp16)", embed)
        self._embed_tok = AutoTokenizer.from_pretrained(embed)
        self._embed = load(AutoModel, embed, quantized=False)
        self.embed_dim = int(self._embed.config.hidden_size)

        log.info("loading VLM %s", vlm)
        self._vlm_processor = AutoProcessor.from_pretrained(vlm)
        self._vlm = load(AutoModelForImageTextToText, vlm)

        log.info("loading LLM %s", llm)
        self._llm_tok = AutoTokenizer.from_pretrained(llm)
        self._llm = load(AutoModelForCausalLM, llm)
        log.info("all models resident")

    def embed(self, text: str) -> np.ndarray:
        torch = self._torch
        inputs = self._embed_tok(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to("cuda")
        with torch.inference_mode():
            out = self._embed(**inputs)
        # bge-m3 uses the CLS token as the sentence representation.
        vec = out.last_hidden_state[0, 0].float().cpu().numpy()
        return np.asarray(vec, dtype=np.float32)

    def describe(self, image_path: Path) -> tuple[str, list[str]]:
        summary = one_sentence(self._vlm_generate(image_path, DESCRIBE_PROMPT, 96))
        if not summary:
            raise SidecarError("DESCRIBE_EMPTY", "VLM produced an empty summary")
        objects = parse_objects(self._vlm_generate(image_path, OBJECTS_PROMPT, 48))
        return summary, objects

    def _vlm_generate(self, image_path: Path, instruction: str, max_tokens: int) -> str:
        torch = self._torch
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "path": str(image_path)},
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        inputs = self._vlm_processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to("cuda", dtype=torch.float16)
        with torch.inference_mode():
            ids = self._vlm.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        trimmed = ids[0][inputs["input_ids"].shape[-1] :]
        return self._vlm_processor.decode(trimmed, skip_special_tokens=True)

    def answer(self, question: str, context: list[str]) -> str:
        if not context:
            return REFUSAL
        torch = self._torch
        joined = "\n".join(context)
        messages = [
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": f"觀察記錄:\n{joined}\n\n問題:{question}"},
        ]
        text = self._llm_tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = self._llm_tok(text, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            ids = self._llm.generate(**inputs, max_new_tokens=160, do_sample=False)
        trimmed = ids[0][inputs["input_ids"].shape[-1] :]
        answer = one_sentence(self._llm_tok.decode(trimmed, skip_special_tokens=True))
        return answer or REFUSAL


# -- server --------------------------------------------------------------


class Server:
    def __init__(self, backend: Backend, data_dir: Path) -> None:
        self.backend = backend
        self.data_dir = data_dir
        # One GPU: never run two generations at once, whatever the client does.
        self._gpu = asyncio.Lock()

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        peer = "backend"
        log.info("%s connected", peer)
        try:
            while True:
                try:
                    raw = await reader.readline()
                except ValueError:
                    # A request longer than READ_LIMIT poisons the framing:
                    # the unread tail would resync mid-line. Drop the socket
                    # and let the backend's 2s reconnect loop recover.
                    log.error("oversized request line, dropping connection")
                    return
                if not raw:
                    return
                reply = await self._dispatch(raw)
                writer.write(json.dumps(reply, ensure_ascii=False).encode("utf-8") + b"\n")
                await writer.drain()
        except (ConnectionResetError, BrokenPipeError):
            log.info("%s disconnected mid-request", peer)
        finally:
            log.info("%s disconnected", peer)
            writer.close()
            with contextlib.suppress(OSError, asyncio.CancelledError):
                await writer.wait_closed()

    async def _dispatch(self, raw: bytes) -> dict[str, Any]:
        try:
            request = json.loads(raw)
        except json.JSONDecodeError as exc:
            # No req_id to echo: the backend treats a mismatch as BAD_JSON's
            # sibling and drops the connection either way.
            return failed("", "BAD_JSON", f"unparseable request: {exc}")
        if not isinstance(request, dict):
            return failed("", "BAD_REQUEST", "expected a JSON object")
        req_id = str(request.get("req_id", ""))
        kind = request.get("kind")
        started = time.monotonic()
        try:
            if kind == "describe":
                summary, objects = await self._describe(_field(request, "image_path"))
                return {
                    "kind": "described",
                    "req_id": req_id,
                    "summary": summary,
                    "objects": objects,
                    "ms": _ms(started),
                }
            if kind == "embed":
                vec = await self._embed(_field(request, "text"))
                return {
                    "kind": "embedded",
                    "req_id": req_id,
                    "vec": vec,
                    "ms": _ms(started),
                }
            if kind == "answer":
                answer = await self._answer(
                    _field(request, "question"), _context(request)
                )
                return {
                    "kind": "answered",
                    "req_id": req_id,
                    "answer": answer,
                    "ms": _ms(started),
                }
            # sidecar.md 3.1: an unknown kind is a Failed, never a crash.
            return failed(req_id, "UNKNOWN_KIND", f"unsupported kind {kind!r}")
        except SidecarError as exc:
            log.warning("request %s failed: %s", req_id or "?", exc)
            return failed(req_id, exc.code, exc.message)
        except Exception as exc:  # inference blew up: report, do not die
            log.exception("request %s crashed", req_id or "?")
            return failed(req_id, "INFERENCE_ERROR", f"{type(exc).__name__}: {exc}")

    async def _describe(self, image_path: str) -> tuple[str, list[str]]:
        # sidecar.md 3.1: image_path is relative to --data-dir, joined here.
        resolved = (self.data_dir / image_path).resolve()
        if not resolved.is_file():
            raise SidecarError("IMAGE_NOT_FOUND", f"no image at {resolved}")
        async with self._gpu:
            return await asyncio.to_thread(self.backend.describe, resolved)

    async def _embed(self, text: str) -> list[float]:
        async with self._gpu:
            vec = await asyncio.to_thread(self.backend.embed, text)
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        if arr.size != self.backend.embed_dim:
            raise SidecarError(
                "EMBED_DIM", f"got {arr.size} dims, expected {self.backend.embed_dim}"
            )
        if not np.isfinite(arr).all():
            # json.dumps would emit a bare `NaN`, which is not valid JSON, and
            # one NaN in the index silently disables the refusal threshold
            # (sidecar.md 3.1). Stop it on this side of the wire.
            raise SidecarError("EMBED_NONFINITE", "embedding contains NaN or inf")
        return [float(x) for x in arr]

    async def _answer(self, question: str, context: list[str]) -> str:
        async with self._gpu:
            return await asyncio.to_thread(self.backend.answer, question, context)


def _field(request: dict[str, Any], name: str) -> str:
    value = request.get(name)
    if not isinstance(value, str) or not value:
        raise SidecarError("BAD_REQUEST", f"{name} must be a non-empty string")
    return value


def _context(request: dict[str, Any]) -> list[str]:
    """`context` may legitimately be absent or empty (sidecar.md 3.2)."""
    value = request.get("context", [])
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise SidecarError("BAD_REQUEST", "context must be a list of strings")
    return value


def _ms(started: float) -> int:
    return max(0, int((time.monotonic() - started) * 1000))


def failed(req_id: str, code: str, message: str) -> dict[str, Any]:
    return {"kind": "failed", "req_id": req_id, "code": code, "message": message}


def build_backend(args: argparse.Namespace) -> Backend:
    if args.backend == "mlx":
        return MLXBackend(
            vlm=args.vlm or DEFAULT_MLX_VLM,
            llm=args.llm or DEFAULT_MLX_LLM,
            embed=args.embed or DEFAULT_MLX_EMBED,
        )
    return CudaBackend(
        vlm=args.vlm or DEFAULT_CUDA_VLM,
        llm=args.llm or DEFAULT_CUDA_LLM,
        embed=args.embed or DEFAULT_CUDA_EMBED,
        quantize=args.quantize,
    )


async def serve(args: argparse.Namespace) -> int:
    backend = build_backend(args)
    if args.embed_dim and backend.embed_dim != args.embed_dim:
        # Better to refuse now than to fill the index with wrong-width rows:
        # the backend would reject every reply with BAD_REPLY anyway.
        log.error(
            "embed dim %d != --embed-dim %d; refusing to start",
            backend.embed_dim,
            args.embed_dim,
        )
        return 1

    socket_path = Path(args.socket)
    if socket_path.exists():
        # A stale socket from a killed process would make bind() fail with
        # EADDRINUSE even though nobody is listening.
        socket_path.unlink()
    socket_path.parent.mkdir(parents=True, exist_ok=True)

    server = Server(backend, args.data_dir.resolve())
    listener = await asyncio.start_unix_server(
        server.handle, path=str(socket_path), limit=READ_LIMIT
    )
    log.info(
        "sidecar listening on %s (vlm=%s llm=%s embed=%s dim=%d)",
        socket_path,
        backend.vlm_model,
        backend.llm_model,
        backend.embed_model,
        backend.embed_dim,
    )
    if args.ready_fd:
        # The verification script waits for this instead of sleeping: model
        # load is tens of seconds and a fixed sleep is either slow or flaky.
        os.write(args.ready_fd, b"ready\n")

    stop = asyncio.get_running_loop().create_future()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            asyncio.get_running_loop().add_signal_handler(
                sig, lambda: stop.done() or stop.set_result(None)
            )
    async with listener:
        with contextlib.suppress(asyncio.CancelledError):
            await stop
    socket_path.unlink(missing_ok=True)
    log.info("sidecar stopped")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python server.py")
    parser.add_argument("--socket", default="/tmp/vlm.sock")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data"),
        help="image_path in a Describe request is relative to this (sidecar.md 3.1)",
    )
    parser.add_argument("--backend", choices=("mlx", "cuda"), default="cuda")
    parser.add_argument("--vlm", default=None)
    parser.add_argument("--llm", default=None)
    parser.add_argument("--embed", default=None)
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="load the cuda backend's models in 4-bit NF4; required on Jetson Orin, "
        "where the fp16 models do not fit in 15.6GB of shared memory",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=1024,
        help="refuse to start unless the model matches; 0 disables the check",
    )
    parser.add_argument(
        "--ready-fd",
        type=int,
        default=0,
        help="write 'ready' to this fd once the socket is listening",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    return asyncio.run(serve(args))


if __name__ == "__main__":
    raise SystemExit(main())
