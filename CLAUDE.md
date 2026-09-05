# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this project is

**Mneme** — a hackathon project (BUILDMODE GEN-AI HACKATHON 2026, Track 06) that turns a
Jetson Orin + camera into an offline, Chinese-language searchable memory of a room.
A capture pipeline samples frames, a VLM describes changes into one-sentence Chinese
summaries, everything is embedded and indexed, and a web timeline / LINE bot can be asked
natural-language questions ("我的充電器最後放在哪") answered with citations back to the
original frames. Everything — capture, VLM, embedding, LLM — runs on-device; no frame or
question ever leaves the Orin.

`mneme/` (backend), `sidecar/` (inference), `web/` (timeline UI) and `bot/` (a Telegram
bot standing in for the LINE bot — see its module docstring for why) are implemented and
run end to end on the Orin against the live camera; `./start.sh` brings all four up in
dependency order. `docs/` remains the contract: treat it as the spec, not as background
reading.

## Source of truth: the three contract documents

This project is built by three people working in parallel off one interface contract,
split across three files with **continuous, non-overlapping section numbers** (so
`§2.6`, `§8.3`, etc. are unambiguous across files):

| File | Sections | Audience |
|---|---|---|
| [`docs/spec.md`](docs/spec.md) | §0 global conventions, §2 external HTTP API, §5 division of work, §6 timeline, §7 risks | Everyone. Frontend/bot only need §2 |
| [`docs/backend.md`](docs/backend.md) | §1 SQLite schema, §3.3 pipeline shape, §4 fake-data seed, §8 backend implementation contract (except §8.5) | Backend implementer |
| [`docs/sidecar.md`](docs/sidecar.md) | §3.1 wire protocol, §3.2 prompt contract, §8.5 mock sidecar | Whoever writes the VLM/LLM/embedding inference process |

**Read the relevant doc(s) before writing code that touches the HTTP API, the DB schema,
the sidecar wire protocol, or the CLI/env surface.** These are cross-referenced by section
number (e.g. `backend.md §8.3`), not duplicated — do not guess a JSON shape, error code, or
default value; look it up.

If a task requires changing something documented as contract (the API in `spec.md` §2, the
schema in `backend.md` §1, the wire protocol/prompt in `sidecar.md` §3.1/§3.2, dependency
choices in `backend.md` §8.2, CLI/env in `backend.md` §8.3), say so explicitly and bump the
doc's version header — don't change behavior silently. The files version independently:
`spec.md` v1.8, `backend.md` v1.6, `sidecar.md` v1.6.

> **Owed bump:** `sidecar.md` §3.2's prompt text was changed on the Orin (commit
> `7018765`) to stop first-person questions being refused against records that answer
> them. `sidecar.md` still says v1.6 and needs v1.7. Raise it with the sidecar's author
> rather than editing the shared contract unilaterally.

## Non-negotiable constraints (already decided, do not relitigate)

- **Backend language: Python** (asyncio + FastAPI), one process, `uvicorn` with a **single
  worker** — see `backend.md` §8.2/§8.6 for why (in-memory retrieval table, SSE broadcast,
  and the pipeline queue all live in process memory; multiple workers would fragment state).
- **Two separate venvs, two separate processes**: the main app and the inference sidecar.
  The main venv must **never** have `torch` installed. The sidecar has its own
  `requirements.txt`. They talk over a Unix socket (`/tmp/vlm.sock`) with line-delimited
  JSON — see `sidecar.md` §3.1. Do not import the sidecar into the main process even though
  both are Python.
- **Never `pip install opencv-python` from PyPI on Jetson** — it may compile from source
  and burn hours (`backend.md` §8.2). The rule as written assumes JetPack ships Python
  bindings; **this box has only the C++ `libopencv-dev` 4.8.0**, so
  `--system-site-packages` inherits nothing and `import cv2` fails, taking
  `mneme.seed` with it. What works is a prebuilt Jetson wheel, which never builds:
  `pip install --only-binary=:all: --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 opencv-python==4.10.0.84`.
  That wheel has **no GStreamer support**, so it cannot open the CSI camera — see the
  camera note below.
- **No sqlite-vec, no faiss.** Vector search is a single in-memory numpy matmul over all
  embeddings (`backend.md` §8.7) — data volume is a few thousand events, this is fast
  enough. Don't introduce a vector DB.
- **No SQLAlchemy, no aiosqlite.** Stdlib `sqlite3`, one dedicated write connection guarded
  by an `asyncio.Lock`, thread-local read connections (`backend.md` §8.6).
- **No migration framework.** Schema changes during the hackathon just mean deleting and
  re-seeding the DB (`backend.md` §8.6).
- All timestamps are **UTC ISO 8601** everywhere except the single exception documented in
  `spec.md` §0 / `sidecar.md` §3.2: the backend converts to `Asia/Taipei` only when building
  the LLM's `context` string for `/api/ask`'s human-facing `answer`.
- IDs are ULIDs with `evt_` / `frm_` prefixes. DB paths are always relative to
  `--data-dir`, never absolute (so the whole data directory is portable).
- No auth. This is a 48-hour demo on a LAN.
- The reject-threshold behavior in `/api/ask` (cosine `< 0.35` → fixed
  "我沒有看到相關的畫面。", empty citations, no LLM call) is a **hard acceptance
  criterion** — see `backend.md` §8.8's last two curl checks. Don't let the model
  hallucinate an answer when nothing matched.

## This Orin in particular (found the hard way — read before debugging inference)

JetPack 6.2 / L4T 36.4.7 / CUDA 12.6 / Python 3.10, **15.6GB of memory shared between CPU
and GPU**. Almost everything surprising follows from that shared pool.

- **An `NVML_SUCCESS == r INTERNAL ASSERT FAILED` or `NvMapMemAllocInternalTagged: error
  12` is out of memory**, not a PyTorch bug. Tegra's iGPU does not fully support NVML, so
  torch dies inside NVML while trying to report the OOM. Treat it as a memory problem.
- **The fp16 defaults do not fit.** SmolVLM2 + Qwen2.5-7B + bge-m3 is ~21.6GB against
  15.6GB. `sidecar/server.py --quantize` loads the VLM and LLM in 4-bit NF4.
- **bitsandbytes must be built for sm_87.** The PyPI aarch64 wheel targets datacenter ARM
  and dies at run time with `named symbol not found`; take it from
  `https://pypi.jetson-ai-lab.io/jp6/cu126` instead. Same index solves opencv above.
- **transformers must be `<5`.** v5 rejects JetPack's torch 2.5 ("PyTorch was not found")
  and renamed `from_pretrained`'s `torch_dtype` to `dtype`.
- **`typing.Self` needs 3.11**; this is 3.10. `python3-venv` is not installed either — use
  `python3 -m virtualenv --system-site-packages`.
- **vLLM is the fast path, transformers is not.** The same Qwen2.5-VL-3B takes 105s per
  describe through transformers+NF4 (which dequantises every forward pass) and ~3s under
  vLLM's fused `awq_marlin` kernels. `--vlm-url` / `--llm-url` point the sidecar at a
  served model; `--embed-device cpu` then frees the second CUDA context (~1.5GB) that a
  co-resident sidecar would otherwise pay for. See [`vLLM/README.md`](vLLM/README.md).
- **`--gpu-memory-utilization` is a pre-flight budget check, not a cap.** The fraction is
  of *total system memory* and the OS's share counts against it, so setting it *low* makes
  vLLM refuse to start with "No available memory for the cache blocks" — the opposite of
  the x86 intuition. Actual KV allocation is bounded by `--max-model-len`. Page cache also
  counts as used, so `sync && echo 3 | sudo tee /proc/sys/vm/drop_caches` before starting.
- **Two cameras, and `CAM_SRC` in `start.sh` picks one** (default `usb`):
  - **USB** (`CAM_SRC=usb`) — a Generic USB Camera on `/dev/video1`, wide angle, and it
    emits **MJPG natively** at 1280x720. Its pipeline therefore neither decodes nor
    encodes: `v4l2src ! image/jpeg ! jpegparse ! videorate ! multifilesink`. `videorate`
    does negotiate `image/jpeg`, but only with `jpegparse` in front of it. Address it as
    `/dev/v4l/by-id/usb-Generic_USB_Camera_*-video-index0`, never `/dev/video1` — the
    number moves when USB devices are replugged.
  - **CSI** (`CAM_SRC=csi`) — the IMX219 on the ribbon cable, emitting 10-bit Bayer
    (`RG10`) that `cv2.VideoCapture` cannot decode. Its pipeline has to convert
    (`nvvidconv`) and encode (`jpegenc`). The sensor floor is 21fps, so throttle with
    `videorate`. A killed pipeline leaves the Argus session wedged (`Failed to create
    CaptureSession`); `sudo systemctl restart nvargus-daemon` clears it. **Worth knowing
    on demo day.**

  Either way frames arrive through `--camera-cmd` (`spec.md` §7) and `multifilesink` —
  `num-buffers=1` with `filesink` hangs without flushing. Both run at **21fps**, not 1:
  the web UI plays those frames as a live MJPEG stream (`spec.md` §2.8) and the backend
  samples them down to ~2/s before the change filter, so the VLM's workload is unchanged.
  Frames reach the stream as the camera's own JPEG bytes — never decoded, never
  re-encoded — which is what makes 21fps affordable next to vLLM.
- **The live streams make SIGTERM a real question.** `/api/stream` and
  `/api/frames/live.mjpg` never end on their own, so uvicorn's *graceful* shutdown waits
  for them forever: the process releases the port (so the next start binds and looks
  healthy) while still holding the camera and writing into the same `data/incoming`, and
  two capture pipelines then interleave two rooms into one timeline. `uvicorn.run` is
  pinned to `timeout_graceful_shutdown=5` for this, and `./start.sh stop` escalates to
  `kill -9`. If capture ever looks like it is running at double speed, look for a second
  `python -m mneme` before looking at the code.
- **Hugging Face's Xet backend corrupts resumed downloads.** On a flaky connection it
  stalls, then resumes into a file that is byte-for-byte the right size but has a garbage
  header. `HF_HUB_DISABLE_XET=1` is ignored by huggingface_hub 0.30; `pip uninstall hf_xet`
  is what actually forces the classic resumable HTTP path. Verify weights by parsing the
  safetensors header, never by comparing file size.

## Repo layout (per `backend.md` §8.1 — build it if it doesn't exist yet)

```
mneme/            # main package: __main__.py, app.py, config.py, db.py, capture.py,
                  # filter.py, sidecar.py, store.py, search.py, seed.py, api/*.py
sidecar/          # server.py, prompts.py, requirements.txt — separate venv, has CUDA/torch
web/              # static frontend (React + htm, vendored), served at /
bot/              # telegram_bot.py — reads TELEGRAM_API_KEY from .env; its own process
data/             # runtime output: memory.db, frames/, thumbs/ (default --data-dir)
scripts/          # verify_sidecar.py — the wire contract against real inference
vLLM/             # the served-inference track: jetson-containers clone (gitignored, ~30GB
                  # with weights) plus launch/test scripts. Not part of backend.md §8.1.
```

Single package, two entrypoints, no monorepo / src-layout / migration tooling — this is
intentionally minimal for a 48-hour build (`backend.md` §8.1).

## Running / testing

```bash
# no Orin / no camera: seed fake data + the deterministic mock sidecar
.venv/bin/python -m mneme.seed --out data/memory.db --data-dir ./data --hours 8 --count 60 --seed 42
.venv/bin/python -m mneme --no-camera --mock-sidecar
```

On the Orin, all three inference roles are served by one vLLM (see the section above for
why). Start it first, then the sidecar, then the app:

```bash
# 1. vLLM serving Qwen2.5-VL-3B-Instruct-AWQ on :8000
cd vLLM && ./serve_qwen2_5_vl.sh

# 2. sidecar: VLM and LLM over HTTP, only the embedder resident, on the CPU
export HF_HUB_CACHE=$PWD/vLLM/jetson-containers/data/models/huggingface
M=Qwen/Qwen2.5-VL-3B-Instruct-AWQ
sidecar/.venv/bin/python sidecar/server.py --backend cuda \
    --vlm-url http://127.0.0.1:8000/v1 --vlm $M \
    --llm-url http://127.0.0.1:8000/v1 --llm $M \
    --embed-device cpu --socket /tmp/vlm.sock --data-dir ./data

# 3. the app, with the USB camera through GStreamer (cv2 cannot open the CSI one,
#    and going through GStreamer keeps the JPEG passthrough for both)
.venv/bin/python -m mneme --data-dir ./data --sidecar /tmp/vlm.sock --bind 0.0.0.0:8080 \
    --sidecar-timeout-ms 60000 --capture-fps 21 \
    --camera-cmd 'gst-launch-1.0 v4l2src device=/dev/v4l/by-id/usb-Generic_USB_Camera_200901010001-video-index0 ! image/jpeg,width=1280,height=720,framerate=30/1 ! jpegparse ! videorate ! image/jpeg,framerate=21/1 ! multifilesink location=frame_%05d.jpg'

# 4. the telegram bot, which finds its token in .env by itself
.venv/bin/python bot/telegram_bot.py --api http://127.0.0.1:8080
```

`./start.sh` does all four (and `./start.sh stop` / `status`). It is the demo-day path;
the commands above are what it runs. `CAM_SRC=csi ./start.sh` swaps in the ribbon-cable
camera; both pipelines live at the top of the script.

`--sidecar-timeout-ms` has to be raised: describe costs ~14s (two VLM turns) against a 20s
default, and a client timeout mid-describe drops the connection and wedges the pipeline.

`--mock-sidecar` swaps in an in-process deterministic fake model (`sidecar.md` §8.5) so the
whole API can be developed and tested without a GPU. `/api/health` must honestly report
`"sidecar": "mock"` and `"mode": "seed-only"` — never fake these fields to look better.

The acceptance checklist in `backend.md` §8.8 is the closest thing to a test suite right
now; run it after any change touching the API, schema, or retrieval. The two checks at the
end are hard requirements: the reject-answer path, and byte-identical reproducibility of
`python -m mneme.seed` for a fixed `--seed`.

`scripts/verify_sidecar.py` drives the whole wire contract against real inference (33
checks). To exercise the deployed shape rather than a local-load one that will not fit
beside vLLM, pass the same URLs:

```bash
.venv/bin/python scripts/verify_sidecar.py --backend cuda --no-camera \
    --vlm-url http://127.0.0.1:8000/v1 --vlm $M \
    --llm-url http://127.0.0.1:8000/v1 --llm $M \
    --embed-device cpu --data-dir /tmp/verify --db /tmp/verify/memory.db
```

**Give it a freshly seeded database of its own.** It takes the newest 16 events as its
corpus but asserts on a specific seeded row (the mug), so after any live capture the real
events crowd the seed data out and it reports three failures that are the fixture's fault,
not the code's. That cost real debugging time — the script is at its most misleading
exactly where it is most needed.

## Full CLI/env reference

`backend.md` §8.3 is the single source of truth for every flag / `MNEME_*` env var and its
default. Don't hardcode a default anywhere without checking that table first.
