"""FastAPI app, runtime state, pipeline and background tasks.

`Runtime` is the single object every handler reads through `app.state.runtime`.
It is built in the lifespan so tests and the seed tool can hand in their own
config instead of reading env at import time (backend.md 8.3).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import platform
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.requests import Request

from . import EMBED_TIMEOUT_MS
from .api import build_router, error_response, install_exception_handlers
from .capture import FpsMeter, open_source, require_cv2
from .config import Config
from .db import Database
from .filter import ChangeFilter, downscale
from .search import SearchIndex, load_index
from .sidecar import Sidecar, build_sidecar
from .store import Broadcaster, Frame, LiveStream, Store, encode_live

log = logging.getLogger(__name__)

VLM_QUEUE_SIZE = 4
"""backend.md 3.3: the only place in the pipeline that ever backs up."""

FILTER_INTERVAL_S = 0.5
"""How often the change filter is allowed to look at a frame.

The camera runs at 21fps so the browser gets a live picture (spec.md 2.8), but
diffing all 21 would spend real CPU deciding not to describe frames the
`--cooldown-ms` gate would have refused anyway. Twice a second is far finer
than the 4s cooldown, so nothing the pipeline would have kept is missed."""

OFFLINE_PROBE_HOST = "1.1.1.1"
OFFLINE_PROBE_PORT = 443
OFFLINE_PROBE_INTERVAL_S = 5.0
OFFLINE_PROBE_TIMEOUT_S = 1.0


@dataclass
class Runtime:
    config: Config
    db: Database
    index: SearchIndex
    sidecar: Sidecar
    store: Store
    broadcaster: Broadcaster
    fps: FpsMeter
    live: LiveStream
    started_at: float
    device: str
    offline: bool = True
    vlm_queue: asyncio.Queue[Any] | None = None
    tasks: list[asyncio.Task[None]] = field(default_factory=list)

    def queue_depth(self) -> int:
        return self.vlm_queue.qsize() if self.vlm_queue is not None else 0


def detect_device() -> str:
    """`orin` when we can see Jetson's model node, else the machine arch."""
    model = Path("/proc/device-tree/model")
    with contextlib.suppress(OSError):
        text = model.read_text(errors="ignore").lower()
        if "orin" in text:
            return "orin"
        if "jetson" in text:
            return "jetson"
    return platform.machine()


async def offline_probe(runtime: Runtime) -> None:
    """backend.md 8.4: measure reachability in the background, never in a
    handler. Single writer + GIL-atomic bool assignment, so no lock."""
    while True:
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(OFFLINE_PROBE_HOST, OFFLINE_PROBE_PORT),
                timeout=OFFLINE_PROBE_TIMEOUT_S,
            )
        except (asyncio.TimeoutError, OSError):
            runtime.offline = True
        else:
            runtime.offline = False
            writer.close()
            with contextlib.suppress(OSError, asyncio.TimeoutError):
                await writer.wait_closed()
        await asyncio.sleep(OFFLINE_PROBE_INTERVAL_S)


async def capture_loop(runtime: Runtime, queue: asyncio.Queue[Frame]) -> None:
    """capture -> live view + change_filter -> bounded queue.

    Every frame reaches the live view; only a sampled few are decoded for the
    change filter, and fewer still reach the VLM. Drops the oldest when full:
    never `await queue.put()`, that would let the VLM stall capture and the
    measured fps would collapse (backend.md 3.3).
    """
    config = runtime.config
    change_filter = ChangeFilter(config.diff_threshold, config.cooldown_ms)
    source = open_source(config)
    last_look = 0.0
    try:
        async for frame in source:
            runtime.fps.mark()
            jpeg = frame.jpeg
            if jpeg is None:
                jpeg = await asyncio.to_thread(encode_live, frame.mat)
            runtime.live.publish(jpeg, frame.ts)

            now = time.monotonic()
            if now - last_look < FILTER_INTERVAL_S:
                continue
            last_look = now
            small = await asyncio.to_thread(_downscaled, frame)
            if small is None or not change_filter.should_pass(small):
                continue
            try:
                queue.put_nowait(frame)
            except asyncio.QueueFull:
                with contextlib.suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
                with contextlib.suppress(asyncio.QueueFull):
                    queue.put_nowait(frame)
    except asyncio.CancelledError:
        raise
    except Exception:
        log.exception("capture loop stopped")
    finally:
        with contextlib.suppress(Exception):
            await source.aclose()


def _downscaled(frame: Frame):
    """Decode + downscale in one thread hop. None for a frame cv2 cannot read.

    A truncated JPEG must not end the capture loop -- the camera keeps running
    and the next frame is usually fine.
    """
    try:
        return downscale(frame.decode())
    except Exception as exc:  # ValueError from us, cv2.error from cv2
        log.debug("skipping an undecodable frame: %s", exc)
        return None


async def vlm_loop(runtime: Runtime, queue: asyncio.Queue[Frame]) -> None:
    """Single worker: describe the frame, embed the summary, store + broadcast."""
    while True:
        frame = await queue.get()
        try:
            row = await runtime.store.save_frame(frame)
            summary, objects = await runtime.sidecar.describe(row.path)
            vec = await runtime.sidecar.embed(summary)
            await runtime.store.save_event(
                frame=row,
                summary=summary,
                objects=objects,
                confidence=1.0,
                source="vlm",
                vec=vec,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("failed to turn a frame into an event")
        finally:
            queue.task_done()


def build_runtime(config: Config) -> Runtime:
    """Open the DB, build the schema, load the vector index. Sync on purpose:
    a dimension mismatch must abort startup before we bind the port."""
    config.data_dir.mkdir(parents=True, exist_ok=True)
    config.frames_dir.mkdir(parents=True, exist_ok=True)
    config.thumbs_dir.mkdir(parents=True, exist_ok=True)
    config.db.parent.mkdir(parents=True, exist_ok=True)

    sidecar = build_sidecar(config, EMBED_TIMEOUT_MS)
    db = Database(config.db)
    # Raises EmbedDimMismatch when meta disagrees with --embed-dim; letting it
    # propagate aborts startup before uvicorn binds the port.
    db.init_schema(embed_model=sidecar.embed_model, embed_dim=config.embed_dim)

    index = load_index(db.load_embeddings(), config.embed_dim)
    broadcaster = Broadcaster()
    store = Store(db=db, index=index, data_dir=config.data_dir, broadcaster=broadcaster)
    return Runtime(
        config=config,
        db=db,
        index=index,
        sidecar=sidecar,
        store=store,
        broadcaster=broadcaster,
        fps=FpsMeter(),
        live=LiveStream(),
        started_at=time.monotonic(),
        device=detect_device(),
    )


def create_app(config: Config) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime: Runtime = build_runtime(config)
        app.state.runtime = runtime
        await runtime.sidecar.start()
        runtime.tasks.append(
            asyncio.create_task(offline_probe(runtime), name="offline-probe")
        )
        if not config.no_camera:
            require_cv2()  # fail loudly now, not at the first sample
            queue: asyncio.Queue[Frame] = asyncio.Queue(maxsize=VLM_QUEUE_SIZE)
            runtime.vlm_queue = queue
            runtime.tasks.append(
                asyncio.create_task(capture_loop(runtime, queue), name="capture")
            )
            runtime.tasks.append(asyncio.create_task(vlm_loop(runtime, queue), name="vlm"))
        log.info(
            "mneme up: mode=%s sidecar=%s events_indexed=%d",
            config.mode,
            runtime.sidecar.status,
            runtime.index.count,
        )
        try:
            yield
        finally:
            for task in runtime.tasks:
                task.cancel()
            for task in runtime.tasks:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
            await runtime.sidecar.stop()
            runtime.db.close()

    app = FastAPI(title="Mneme", version="1.4.0", lifespan=lifespan)
    # No auth anywhere, so a wildcard CORS header costs nothing (spec.md 2.7).
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    install_exception_handlers(app)
    app.include_router(build_router())
    _mount_static(app, config.static_dir)
    return app


def _mount_static(app: FastAPI, static_dir: Path) -> None:
    """Serve `--static-dir` at `/` with SPA fallback; /api/* keeps priority
    because the API router is registered first (spec.md 2.7).

    Deliberately no StaticFiles mount. A mount owns its whole prefix, so it
    answers its own 404s and never reaches the fallback below -- which is
    exactly wrong for a SPA, where an unknown path must render index.html.
    The catch-all already serves every file under static_dir, including
    `assets/`, so a mount would only add a prefix that behaves differently
    from the rest of the tree.
    """
    index_html = static_dir / "index.html"

    @app.get("/{path:path}", include_in_schema=False)
    @app.head("/{path:path}", include_in_schema=False)
    async def spa(path: str, request: Request):
        if path == "api" or path.startswith("api/"):
            # Past the API router, so this route does not exist. Segment-exact
            # so a SPA route like /apiary still falls through to index.html.
            return error_response("NOT_FOUND", f"no route /{path}")
        root = static_dir.resolve()
        candidate = (static_dir / path).resolve() if path else index_html
        if path and candidate.is_file() and root in candidate.parents:
            return FileResponse(candidate)
        if index_html.is_file():
            return FileResponse(index_html)
        return error_response(
            "NOT_FOUND",
            f"no static build in {static_dir}; frontend not deployed yet",
        )
