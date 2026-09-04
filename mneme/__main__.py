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


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    config = parse_args(argv)
    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port, workers=1, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
