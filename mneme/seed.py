"""`python -m mneme.seed` — deterministic fake dataset. docs/backend.md 4.

Same `--seed` must produce byte-identical output: event ids, summaries,
timestamps, thumbnails. That is what lets three people debug one dataset.

To get there, nothing may read the wall clock or a global RNG:

* all randomness goes through an explicit ``random.Random(seed)``,
* ULIDs are built from the event's own millisecond timestamp plus ten
  deterministic random bytes (the default ``ULID()`` uses `time.time()` and
  `os.urandom`, both of which would break reproducibility),
* the time window ends at ``--end``, which defaults to today's UTC midnight —
  a per-day constant, so runs hours apart still match.

Placeholder art is drawn with ``cv2.putText``, which cannot render Chinese:
the images carry an ASCII index only, Chinese lives in ``events.summary``.
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from ulid import ULID

from .db import VEC_DTYPE, create_schema, iso, parse_iso
from .search import l2_normalize
from .sidecar import mock_embed

FULL_SIZE = (1280, 720)
THUMB_WIDTH = 320

# Summary pool. 馬克杯 appears repeatedly on purpose: backend.md 8.8 greps for
# it in both the keyword and the retrieval checks.
SUMMARIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("有人把馬克杯放到桌子右側", ("person", "mug", "desk")),
    ("桌面沒有變化,只有窗外光線變暗", ("desk", "window")),
    ("一個人把黑色後背包放在桌子左側,然後離開", ("person", "backpack", "desk")),
    ("桌上有一條白色充電線,旁邊放著一個馬克杯", ("cable", "mug", "desk")),
    ("有人坐到椅子上開始使用筆記型電腦", ("person", "chair", "laptop")),
    ("馬克杯被拿走了,桌子右側空出一塊", ("mug", "desk")),
    ("門被打開,一個人走進房間", ("person", "door")),
    ("一個人站在窗邊講電話", ("person", "window", "phone")),
    ("桌上的水瓶被拿走,只剩下鍵盤和滑鼠", ("bottle", "keyboard", "mouse")),
    ("房間裡沒有人,燈是關著的", ("room",)),
    ("有人把一疊書放在桌子中間", ("person", "book", "desk")),
    ("椅子被推回桌子下面,桌面收乾淨了", ("chair", "desk")),
    ("窗外開始下雨,玻璃上有水痕", ("window",)),
    ("有人把充電器插到牆上的插座", ("person", "charger", "socket")),
    ("桌上多了一個外送紙袋", ("bag", "desk")),
    ("一個人拿著馬克杯走出房間", ("person", "mug", "door")),
)

PALETTE = (
    (48, 64, 96),
    (72, 96, 72),
    (96, 72, 56),
    (64, 64, 88),
    (40, 88, 96),
    (88, 56, 72),
)


def deterministic_ulid(ts: datetime, rng: random.Random) -> ULID:
    """ULID with a fixed timestamp prefix and seeded randomness."""
    ms = int(ts.timestamp() * 1000)
    return ULID.from_bytes(ms.to_bytes(6, "big") + bytes(rng.getrandbits(8) for _ in range(10)))


def draw_placeholder(index: int, color: tuple[int, int, int], size: tuple[int, int]):
    """Solid colour plus an ASCII index. cv2.putText has no CJK glyphs."""
    import cv2

    width, height = size
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:, :] = color
    label = f"SEED #{index:03d}"
    scale = width / 640.0
    thickness = max(1, round(2 * scale))
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    origin = ((width - text_w) // 2, (height + text_h) // 2)
    cv2.putText(
        canvas,
        label,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (235, 235, 235),
        thickness,
        cv2.LINE_AA,
    )
    return canvas


def write_images(data_dir: Path, frame_id: str, index: int, color) -> tuple[str, str]:
    import cv2

    full = draw_placeholder(index, color, FULL_SIZE)
    thumb_height = round(FULL_SIZE[1] * THUMB_WIDTH / FULL_SIZE[0])
    thumb = draw_placeholder(index, color, (THUMB_WIDTH, thumb_height))
    rel_full = Path("frames") / f"{frame_id}.jpg"
    rel_thumb = Path("thumbs") / f"{frame_id}.jpg"
    for rel, image in ((rel_full, full), (rel_thumb, thumb)):
        target = data_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(target), image, [int(cv2.IMWRITE_JPEG_QUALITY), 85]):
            raise RuntimeError(f"cv2.imwrite failed for {target}")
    return rel_full.as_posix(), rel_thumb.as_posix()


def default_end() -> datetime:
    """Today's UTC midnight: recent, always in the past, constant for a day."""
    now = datetime.now(timezone.utc)
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def seed(
    *,
    out: Path,
    data_dir: Path,
    hours: float,
    count: int,
    seed_value: int,
    embed_dim: int,
    end: datetime,
) -> tuple[int, str, str]:
    rng = random.Random(seed_value)
    data_dir.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Seed output must be a fresh DB: the same --seed regenerates the same
    # ULIDs, so appending to an existing file collides on the primary key.
    for suffix in ("", "-wal", "-shm"):
        Path(f"{out}{suffix}").unlink(missing_ok=True)

    conn = sqlite3.connect(out, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    create_schema(conn, embed_model="mock-embed", embed_dim=embed_dim)

    start = end - timedelta(hours=hours)
    span_s = max((end - start).total_seconds(), 1.0)
    # Even spacing plus a bounded jitter: looks organic, stays ordered.
    step = span_s / max(count, 1)

    conn.execute("BEGIN")
    try:
        for i in range(count):
            offset = i * step + rng.uniform(0, step * 0.6)
            ts = start + timedelta(seconds=round(offset, 3))
            frame_id = f"frm_{deterministic_ulid(ts, rng)}"
            event_id = f"evt_{deterministic_ulid(ts, rng)}"
            summary, objects = SUMMARIES[rng.randrange(len(SUMMARIES))]
            color = PALETTE[i % len(PALETTE)]
            rel_full, rel_thumb = write_images(data_dir, frame_id, i, color)
            conn.execute(
                "INSERT INTO frames(id, ts, path, thumb_path, width, height) "
                "VALUES(?, ?, ?, ?, ?, ?)",
                (frame_id, iso(ts), rel_full, rel_thumb, FULL_SIZE[0], FULL_SIZE[1]),
            )
            conn.execute(
                "INSERT INTO events(id, ts, frame_id, summary, objects, confidence, source) "
                "VALUES(?, ?, ?, ?, ?, ?, 'seed')",
                (
                    event_id,
                    iso(ts),
                    frame_id,
                    summary,
                    json.dumps(list(objects), ensure_ascii=False),
                    round(rng.uniform(0.62, 0.95), 2),
                ),
            )
            # Seed data without embeddings would kill /api/ask on day one.
            vec = l2_normalize(mock_embed(summary, embed_dim))
            conn.execute(
                "INSERT INTO embeddings(event_id, dim, vec) VALUES(?, ?, ?)",
                (event_id, embed_dim, np.ascontiguousarray(vec, dtype=VEC_DTYPE).tobytes()),
            )
    except BaseException:
        conn.execute("ROLLBACK")
        conn.close()
        raise
    conn.execute("COMMIT")

    row = conn.execute(
        "SELECT count(*) AS n, min(ts) AS lo, max(ts) AS hi FROM events"
    ).fetchone()
    result = (int(row["n"]), row["lo"], row["hi"])
    conn.close()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mneme.seed")
    parser.add_argument("--out", default="data/memory.db")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embed-dim", type=int, default=1024)
    parser.add_argument(
        "--end",
        default=None,
        help="ISO 8601 end of the window; default is today's UTC midnight",
    )
    ns = parser.parse_args(argv)
    if ns.count <= 0:
        parser.error("--count must be > 0")

    count, lo, hi = seed(
        out=Path(ns.out).expanduser(),
        data_dir=Path(ns.data_dir).expanduser(),
        hours=ns.hours,
        count=ns.count,
        seed_value=ns.seed,
        embed_dim=ns.embed_dim,
        end=parse_iso(ns.end) if ns.end else default_end(),
    )
    print(f"event_count={count} range={lo} .. {hi} db={ns.out} data_dir={ns.data_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
