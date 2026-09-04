"""CLI + environment configuration. docs/backend.md 8.3 is the only truth.

Precedence: CLI flag > MNEME_* env > default. argparse gives us that for free
via `default=os.environ.get("MNEME_X", <default>)`.

Nothing here runs at import time on module level env: the config object is
built explicitly so seed and tests can hand in their own.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

TRUE_WORDS = frozenset({"1", "true", "yes"})


def env_flag(name: str) -> bool:
    """MNEME_* boolean env: 1/true/yes, case-insensitive. Absent -> False."""
    return os.environ.get(name, "").strip().lower() in TRUE_WORDS


@dataclass(frozen=True, slots=True)
class Config:
    data_dir: Path
    db: Path
    camera: str
    camera_cmd: str | None
    no_camera: bool
    sidecar: Path
    mock_sidecar: bool
    host: str
    port: int
    static_dir: Path
    capture_fps: float
    diff_threshold: float
    cooldown_ms: int
    ask_min_score: float
    embed_dim: int
    sidecar_timeout_ms: int

    @property
    def frames_dir(self) -> Path:
        return self.data_dir / "frames"

    @property
    def thumbs_dir(self) -> Path:
        return self.data_dir / "thumbs"

    @property
    def incoming_dir(self) -> Path:
        """--camera-cmd drops JPEGs here; capture watches and deletes them."""
        return self.data_dir / "incoming"

    @property
    def mode(self) -> str:
        """spec.md 2.1: live | seed-only."""
        return "seed-only" if self.no_camera else "live"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    env = os.environ.get
    parser.add_argument("--data-dir", default=env("MNEME_DATA_DIR", "./data"))
    parser.add_argument("--db", default=env("MNEME_DB"))
    parser.add_argument("--camera", default=env("MNEME_CAMERA", "/dev/video0"))
    parser.add_argument("--camera-cmd", default=env("MNEME_CAMERA_CMD"))
    parser.add_argument("--no-camera", action="store_true", default=env_flag("MNEME_NO_CAMERA"))
    parser.add_argument("--sidecar", default=env("MNEME_SIDECAR", "/tmp/vlm.sock"))
    parser.add_argument(
        "--mock-sidecar", action="store_true", default=env_flag("MNEME_MOCK_SIDECAR")
    )
    parser.add_argument("--bind", default=env("MNEME_BIND", "0.0.0.0:8080"))
    parser.add_argument("--static-dir", default=env("MNEME_STATIC_DIR", "./web"))
    parser.add_argument("--capture-fps", type=float, default=float(env("MNEME_CAPTURE_FPS", "2")))
    parser.add_argument(
        "--diff-threshold", type=float, default=float(env("MNEME_DIFF_THRESHOLD", "12.0"))
    )
    parser.add_argument("--cooldown-ms", type=int, default=int(env("MNEME_COOLDOWN_MS", "4000")))
    parser.add_argument(
        "--ask-min-score", type=float, default=float(env("MNEME_ASK_MIN_SCORE", "0.35"))
    )
    parser.add_argument("--embed-dim", type=int, default=int(env("MNEME_EMBED_DIM", "1024")))
    parser.add_argument(
        "--sidecar-timeout-ms", type=int, default=int(env("MNEME_SIDECAR_TIMEOUT_MS", "20000"))
    )


def split_bind(bind: str) -> tuple[str, int]:
    """`0.0.0.0:8080` -> uvicorn host/port. IPv6 literals use [::]:8080."""
    if bind.startswith("["):
        host, _, port = bind.rpartition("]:")
        return host.lstrip("["), int(port)
    host, sep, port = bind.rpartition(":")
    if not sep:
        raise ValueError(f"--bind must be host:port, got {bind!r}")
    return host, int(port)


def from_namespace(ns: argparse.Namespace) -> Config:
    data_dir = Path(ns.data_dir).expanduser()
    host, port = split_bind(ns.bind)
    return Config(
        data_dir=data_dir,
        db=Path(ns.db).expanduser() if ns.db else data_dir / "memory.db",
        camera=ns.camera,
        camera_cmd=ns.camera_cmd or None,
        no_camera=ns.no_camera,
        sidecar=Path(ns.sidecar),
        mock_sidecar=ns.mock_sidecar,
        host=host,
        port=port,
        static_dir=Path(ns.static_dir).expanduser(),
        capture_fps=ns.capture_fps,
        diff_threshold=ns.diff_threshold,
        cooldown_ms=ns.cooldown_ms,
        ask_min_score=ns.ask_min_score,
        embed_dim=ns.embed_dim,
        sidecar_timeout_ms=ns.sidecar_timeout_ms,
    )


def parse_args(argv: list[str] | None = None) -> Config:
    parser = argparse.ArgumentParser(prog="python -m mneme", description="Mneme backend")
    add_arguments(parser)
    return from_namespace(parser.parse_args(argv))
