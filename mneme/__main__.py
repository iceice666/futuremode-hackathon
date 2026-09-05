"""`python -m mneme` — pipeline + HTTP server.

Single uvicorn worker, always: the search index, SSE broadcaster and pipeline
queue all live in process memory (backend.md 8.6).
"""

from __future__ import annotations

import logging
import sys

import uvicorn

from .app import create_app
from .config import parse_args


GRACEFUL_SHUTDOWN_S = 5


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    config = parse_args(argv)
    app = create_app(config)
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        workers=1,
        log_level="info",
        # Two of our endpoints are streams that never end on their own: the SSE
        # feed and the live MJPEG view (spec.md 2.5 / 2.8). Without a deadline a
        # graceful shutdown waits for them, and SIGTERM leaves a process that has
        # released the port but still holds the camera -- so the next start looks
        # like it worked while two capture pipelines fight over ./data/incoming.
        timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_S,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
