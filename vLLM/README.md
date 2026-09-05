# vLLM on Orin — Qwen2.5-VL-3B

Runs Qwen2.5-VL-3B behind vLLM's OpenAI-compatible server on a Jetson Orin (JetPack 6.2 /
L4T 36.4.7 / CUDA 12.6, 15GB unified memory), using
[dusty-nv/jetson-containers](https://github.com/dusty-nv/jetson-containers) (cloned into
`jetson-containers/`) for the Jetson-patched CUDA/PyTorch/vLLM build.

**Status: working.** ~4.3s for a one-sentence Chinese description of a single image;
~10.5GB of 15.6GB RAM in use while serving.

Why Docker rather than plain pip: vLLM publishes no wheel for Jetson's iGPU (sm_87).

> 繁體中文版說明見 [`README.zh-TW.md`](README.zh-TW.md)。

---

## Quick start

Once set up, day-to-day use is two commands:

```bash
cd /home/jetson/futuremode/vLLM
./serve_qwen2_5_vl.sh                                   # start (detached)
python3 test_client.py <image> "Describe this image."   # test
```

Other useful ones:

```bash
docker logs -f mneme-vllm              # follow logs
docker stop mneme-vllm                 # stop
curl http://127.0.0.1:8000/v1/models   # is it up?
```

The server binds `0.0.0.0:8000`, so other machines on the LAN can reach it.

---

## First-time setup

### 1. Install Docker and register the nvidia runtime

```bash
sudo apt update && sudo apt install -y docker.io
sudo usermod -aG docker $USER
sudo systemctl enable --now docker

# Essential - without this, docker run --runtime nvidia fails
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Then log out and back in (or `newgrp docker`), otherwise the current shell still lacks the
`docker` group and you get `permission denied ... /var/run/docker.sock`.

`nvidia-container-toolkit` ships with JetPack, but it still has to be registered with
Docker via `nvidia-ctk` — otherwise you get `unknown or invalid runtime name: nvidia`.

### 2. Download the weights

```bash
./download_model.sh
```

~3.5GB into `jetson-containers/data/models/huggingface` (mounted as `/data` in the
container). Download is deliberately separate from launch — see "Flaky networks" below.

### 3. Launch

```bash
./serve_qwen2_5_vl.sh
```

The first run pulls the 5.7GB container image (`dustynv/vllm:0.8.6-r36.4-cu128-24.04`).

---

## Jetson-specific gotchas (all hit for real, in this order)

This box behaves very differently from an x86 server with a discrete GPU.

### `unknown or invalid runtime name: nvidia`

Having `nvidia-container-toolkit` installed is not enough; run
`sudo nvidia-ctk runtime configure --runtime=docker` and restart Docker.

### `NVML_SUCCESS == r INTERNAL ASSERT FAILED` + `NvMapMemAllocInternalTagged: error 12`

Looks like an NVML bug; it is actually **out of memory** (`error 12` = ENOMEM). Tegra's
iGPU doesn't fully support NVML, so PyTorch dies inside NVML while trying to report the
OOM, hiding the real cause. **Treat an NVML assert here as a memory problem first.**

### bf16 doesn't fit — use AWQ

`Qwen2.5-VL-3B-Instruct` in bf16 is ~7.5GB of weights, and the OS already holds ~6GB, so
allocating the weights ENOMEMs. The 4-bit **`Qwen2.5-VL-3B-Instruct-AWQ`** build is 3.3GB;
vLLM picks the `awq_marlin` kernel automatically, which is well supported on sm_87.

### `--gpu-memory-utilization` means something different here

With unified memory there is no separate VRAM: the fraction applies to **total system
memory**, and **memory the OS already uses counts against that budget**. So:

```
KV cache available ≈ total × util − (OS in use + weights + activations)
                   ≈ 15.3GiB × util − (5.7 + 3.3 + activations)
```

Setting it *too low* starves the KV cache and startup fails outright — the opposite of the
x86 intuition. Currently `0.75`.

### `No available memory for the cache blocks` — the real culprit is **video**

Raising `util` alone still failed. This log line is the tell:

```
Encoder cache will be initialized with a budget of 4096 tokens,
and profiled with 2 video items of the maximum feature size.
```

`--limit-mm-per-prompt image=1` limits images but **not video**. Qwen2.5-VL accepts video,
so vLLM sized its activation budget off two maximum-size video clips and consumed the whole
allowance. This project only handles still frames, so disable it:
`--limit-mm-per-prompt image=1,video=0`. **This was the fix that made it start.**

Relatedly, `--mm-processor-kwargs '{"max_pixels": 401408}'` caps a single image at ~512
visual tokens (the default allows up to 16384), keeping the vision encoder's activation
memory bounded.

### Flaky networks break the download and resume badly

HF's newer Xet/CAS backend fails with `CAS service error ... IncompleteBody` on a dropped
connection, taking the whole engine startup down with it. `download_model.sh` sets
`HF_HUB_DISABLE_XET=1` to use the classic HTTP path (which resumes partial files) and wraps
it in up to 100 retries. Keeping download separate from launch means a failed download
doesn't force a full startup retry.

---

## Current configuration

Everything in `serve_qwen2_5_vl.sh` is overridable via same-named env vars:

| Setting | Value | Why |
|---|---|---|
| `MODEL` | `Qwen/Qwen2.5-VL-3B-Instruct-AWQ` | 4-bit, 3.3GB; bf16 doesn't fit |
| `GPU_MEM_UTIL` | `0.75` | Must exceed existing OS usage to leave KV cache |
| `MAX_MODEL_LEN` | `4096` | Yields 7,248 tokens of KV cache |
| `MAX_NUM_SEQS` | `8` | Default 256 blows up KV cache demand |
| `MAX_PIXELS` | `401408` | ~512 visual tokens per image |
| `DTYPE` | `float16` | |
| `--limit-mm-per-prompt` | `image=1,video=0` | **Disables video profiling** |
| `--enforce-eager` | on | Skips CUDA graph capture memory |

For example, to try a longer context:

```bash
MAX_MODEL_LEN=8192 ./serve_qwen2_5_vl.sh
```

---

## Notes

- This is a standalone OpenAI-compatible HTTP service on port 8000. It is **not** wired
  into the `/tmp/vlm.sock` sidecar protocol from `orin-memory-spec.md` §3 — that wasn't part
  of this task. To integrate, the sidecar should call this server's `/v1/chat/completions`
  rather than loading the model in-process.
- Memory headroom is thin (~4.7GB). Check the server is still alive after running anything
  else memory-hungry.
