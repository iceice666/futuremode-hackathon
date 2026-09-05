#!/usr/bin/env bash
# Runs the vLLM server against a set of images and reports per-image latency.
#
#   ./test.sh                          # all sample images in jetson-containers/data/images
#   ./test.sh dogs.jpg flowers.jpg     # sample images by name
#   ./test.sh /path/to/my.jpg          # any path
#   PROMPT="這是什麼顏色?" ./test.sh dogs.jpg
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

PROMPT="${PROMPT:-用一句繁體中文描述這張圖片裡有什麼。}"
SAMPLE_DIR="${SAMPLE_DIR:-jetson-containers/data/images}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"

if ! curl -sf -m 5 "$BASE_URL/models" >/dev/null 2>&1; then
    echo "Server is not responding at $BASE_URL" >&2
    echo "Start it with: ./serve_qwen2_5_vl.sh" >&2
    exit 1
fi

echo "Model:  $(curl -sf "$BASE_URL/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')"
echo "Prompt: $PROMPT"
echo

if [ $# -gt 0 ]; then
    images=("$@")
else
    mapfile -t images < <(find "$SAMPLE_DIR" -maxdepth 1 -type f -name '*.jpg' | sort)
fi

total=0
count=0
failed=0

for img in "${images[@]}"; do
    # bare filenames are looked up in the sample directory
    if [ ! -f "$img" ] && [ -f "$SAMPLE_DIR/$img" ]; then
        img="$SAMPLE_DIR/$img"
    fi
    if [ ! -f "$img" ]; then
        echo "=== $img ==="
        echo "  (not found, skipped)"
        echo
        failed=$((failed + 1))
        continue
    fi

    echo "=== $(basename "$img") ==="
    start=$(date +%s.%N)
    if output=$(python3 test_client.py "$img" "$PROMPT" 2>&1); then
        end=$(date +%s.%N)
        elapsed=$(echo "$end - $start" | bc)
        echo "$output"
        printf "耗時: %.1f 秒\n\n" "$elapsed"
        total=$(echo "$total + $elapsed" | bc)
        count=$((count + 1))
    else
        echo "$output" | tail -3
        echo "  (request failed)"
        echo
        failed=$((failed + 1))
    fi
done

if [ "$count" -gt 0 ]; then
    printf "%d 張成功,平均 %.1f 秒/張" "$count" "$(echo "$total / $count" | bc -l)"
    [ "$failed" -gt 0 ] && printf ",%d 張失敗" "$failed"
    echo
else
    echo "沒有成功的請求"
    exit 1
fi
