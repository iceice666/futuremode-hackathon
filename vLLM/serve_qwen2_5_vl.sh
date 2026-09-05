#!/usr/bin/env bash
# Launches Qwen2.5-VL-3B-Instruct behind vLLM's OpenAI-compatible API server,
# inside the jetson-containers vLLM image (pulled/built on first run if not cached).
#
# Runs detached (docker run -d) instead of jetson-containers' default -it wrapper,
# so this works from non-interactive shells/scripts and survives disconnects.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/jetson-containers"

# AWQ 4-bit build: ~3.5GB of weights instead of ~7.5GB for bf16, which is what
# actually fits alongside the OS on this 15GB unified-memory Orin.
MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct-AWQ}"
PORT="${PORT:-8000}"
# vLLM's budget = GPU_MEM_UTIL * total memory. On Jetson that total is system RAM,
# so this has to stay under what's actually free, not just under 1.0.
# 0.75 worked from a clean boot but failed once page cache had grown to ~5GB: vLLM reads
# "free" memory, which excludes reclaimable cache, so it under-counts what it can actually
# get. 0.85 lets it claim memory the kernel will reclaim from cache on demand.
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
DTYPE="${DTYPE:-float16}"
# Qwen2.5-VL defaults to accepting huge images (16384 visual tokens), and vLLM sizes
# the vision encoder's activation budget off that maximum - which leaves nothing for
# the KV cache here. 401408 = 512*28*28, i.e. cap images at ~512 visual tokens.
MAX_PIXELS="${MAX_PIXELS:-401408}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
# Default is 256 concurrent sequences, which sizes the KV cache demand way past what
# this box has. 8 is plenty for a single-camera demo.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
CONTAINER_NAME="${CONTAINER_NAME:-mneme-vllm}"
DATA_DIR="$(pwd)/data"

IMAGE="$(./autotag -q vllm)"
if [ -z "$IMAGE" ]; then
    echo "autotag failed to resolve an image; falling back to known-good tag" >&2
    IMAGE="dustynv/vllm:0.8.6-r36.4-cu128-24.04"
fi

if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    echo "Removing existing container '$CONTAINER_NAME'..."
    docker rm -f "$CONTAINER_NAME" >/dev/null
fi

mkdir -p "$DATA_DIR"

docker run -d \
    --runtime nvidia \
    --env NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    --env HF_HUB_DISABLE_XET=1 \
    --network host \
    --shm-size=8g \
    --name "$CONTAINER_NAME" \
    --volume "$DATA_DIR:/data" \
    "$IMAGE" \
    vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --limit-mm-per-prompt image=1,video=0 \
    --mm-processor-kwargs "{\"max_pixels\": $MAX_PIXELS}" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --enforce-eager \
    --dtype "$DTYPE"

echo "Started container '$CONTAINER_NAME'. Follow logs with:"
echo "  docker logs -f $CONTAINER_NAME"
echo "Stop it with:"
echo "  docker stop $CONTAINER_NAME"
