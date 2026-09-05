#!/usr/bin/env bash
# Pre-downloads the model weights into the shared /data HF cache, with retries.
#
# Split out from serve_qwen2_5_vl.sh because on a flaky connection the download is
# the fragile part: HF's Xet/CAS backend aborts the whole engine startup on a
# dropped connection, so we disable it (HF_HUB_DISABLE_XET=1) and use the classic
# HTTP path, which resumes partial files instead of restarting them.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/jetson-containers"

MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct-AWQ}"
IMAGE="${IMAGE:-dustynv/vllm:0.8.6-r36.4-cu128-24.04}"
MAX_TRIES="${MAX_TRIES:-100}"
DATA_DIR="$(pwd)/data"

docker run --rm \
    --network host \
    -e HF_HUB_DISABLE_XET=1 \
    -e MODEL="$MODEL" \
    -e MAX_TRIES="$MAX_TRIES" \
    --volume "$DATA_DIR:/data" \
    "$IMAGE" \
    bash -c '
        for i in $(seq 1 "$MAX_TRIES"); do
            echo "=== download attempt $i/$MAX_TRIES ==="
            if python3 -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(os.environ[\"MODEL\"], max_workers=2)
print(\"DOWNLOAD_COMPLETE\")
"; then
                exit 0
            fi
            echo "attempt $i failed, retrying in 5s..."
            sleep 5
        done
        echo "gave up after $MAX_TRIES attempts"
        exit 1
    '
