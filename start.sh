#!/usr/bin/env bash
# Bring up the whole demo on the Orin, in dependency order.
#
#   ./start.sh          vLLM, sidecar, backend (+ bot if a token is exported)
#   ./start.sh stop     stop them again
#   ./start.sh status   what is up
#
# Order matters: the sidecar's VLM and LLM are served by vLLM, and the backend
# waits on the sidecar's socket. Logs land in ./run/.
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN="$REPO/run"
MODEL="Qwen/Qwen2.5-VL-3B-Instruct-AWQ"
export HF_HUB_CACHE="$REPO/vLLM/jetson-containers/data/models/huggingface"

# Two cameras on this box; CAM_SRC picks one.
#
#   ./start.sh              the USB camera (default)
#   CAM_SRC=csi ./start.sh  the IMX219 on the ribbon cable
#
# Either way frames arrive as JPEGs written by GStreamer, not through cv2
# (spec.md 7): this build of cv2 has no GStreamer support and cannot decode the
# CSI sensor's 10-bit Bayer at all.
#
# 21fps, not 1: the web UI plays these frames as a live MJPEG stream
# (spec.md 2.8), and the backend samples them down to ~2/s before the change
# filter, so the VLM's workload does not change.
CAM_FPS=21

# by-id, not /dev/video1 -- the number moves when USB devices are replugged, and
# a demo that depends on enumeration order is a demo that breaks.
USB_CAM=/dev/v4l/by-id/usb-Generic_USB_Camera_200901010001-video-index0

# The USB camera emits MJPG natively, so this pipeline neither decodes nor
# encodes anything: the camera's own JPEG lands on disk and is forwarded to the
# browser byte for byte. ~94KB a frame, so ~2MB/s per viewer.
CAM_USB="gst-launch-1.0 v4l2src device=$USB_CAM ! image/jpeg,width=1280,height=720,framerate=30/1 ! jpegparse ! videorate ! image/jpeg,framerate=21/1 ! multifilesink location=frame_%05d.jpg"

# The CSI sensor has to be converted and encoded on the way past. 720p at
# quality 60 measures ~85KB a frame; drop the quality or the width if the
# venue's wifi cannot carry it.
CAM_CSI='gst-launch-1.0 nvarguscamerasrc ! video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1 ! nvvidconv ! video/x-raw,width=1280,height=720,format=I420 ! videorate ! video/x-raw,framerate=21/1 ! jpegenc quality=60 ! multifilesink location=frame_%05d.jpg'

case "${CAM_SRC:-usb}" in
    usb) CAM="$CAM_USB" ;;
    csi) CAM="$CAM_CSI" ;;
    *)   printf '!! CAM_SRC must be usb or csi, not %s\n' "$CAM_SRC" >&2; exit 1 ;;
esac

mkdir -p "$RUN"

say() { printf '\033[36m==\033[0m %s\n' "$*"; }
die() { printf '\033[31m!!\033[0m %s\n' "$*" >&2; exit 1; }

# Match on a path fragment that a `pkill -f` of this script would not also
# match -- a pattern that hits your own command line kills your own shell.
pids_for() { ps -eo pid,args | grep "$1" | grep -v grep | awk '{print $1}'; }

# A backend that ignores SIGTERM is worse than one that never stopped: it lets
# go of the port, so the next start binds and looks fine, while the old process
# still holds the camera and writes into the same ./data/incoming. Always check,
# and escalate.
stop_all() {
    # Backend first, then any camera pipeline it left behind.
    local pats=("[t]elegram_bot.py" "[b]in/python -m mneme" "[s]idecar/server.py" "[g]st-launch-1.0")
    local pat p
    for pat in "${pats[@]}"; do
        for p in $(pids_for "$pat"); do say "stopping $p"; kill "$p" 2>/dev/null; done
    done
    sleep 4
    for pat in "${pats[@]}"; do
        for p in $(pids_for "$pat"); do
            say "$p ignored SIGTERM; killing"; kill -9 "$p" 2>/dev/null
        done
    done
    docker stop mneme-vllm >/dev/null 2>&1 && say "stopped vLLM"
    say "stopped"
}

status_all() {
    docker ps --filter name=mneme-vllm --format 'vLLM     {{.Status}}' 2>/dev/null | grep . || echo "vLLM     down"
    [ -S /tmp/vlm.sock ] && echo "sidecar  socket up" || echo "sidecar  down"
    curl -sf -m 3 http://127.0.0.1:8080/api/health >/dev/null 2>&1 \
        && echo "backend  http://$(hostname -I | awk '{print $1}'):8080" || echo "backend  down"
    [ -n "$(pids_for '[t]elegram_bot.py')" ] && echo "bot      polling" || echo "bot      down"
}

case "${1:-start}" in
  stop)   stop_all; exit 0 ;;
  status) status_all; exit 0 ;;
  start)  ;;
  *)      die "usage: $0 [start|stop|status]" ;;
esac

# 1. vLLM ---------------------------------------------------------------
if curl -sf -m 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
    say "vLLM already up"
else
    say "starting vLLM (loads ~40s)"
    # Page cache counts against vLLM's memory check, so a long-running box can
    # fail to start what a freshly booted one starts fine.
    sync && echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1 \
        || say "could not drop page cache (needs sudo); continuing"
    ( cd "$REPO/vLLM" && GPU_MEM_UTIL=0.78 MAX_MODEL_LEN=2048 ./serve_qwen2_5_vl.sh ) \
        > "$RUN/vllm.log" 2>&1 || die "vLLM failed to launch, see $RUN/vllm.log"
    for _ in $(seq 1 60); do
        curl -sf -m 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && break
        sleep 5
    done
    curl -sf -m 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 \
        || die "vLLM did not come up; docker logs mneme-vllm"
fi

# 2. sidecar ------------------------------------------------------------
if [ -S /tmp/vlm.sock ] && [ -n "$(pids_for '[s]idecar/server.py')" ]; then
    say "sidecar already up"
else
    say "starting sidecar (embeds on CPU; VLM and LLM served by vLLM)"
    rm -f /tmp/vlm.sock
    ( cd "$REPO/sidecar" && setsid "$REPO/sidecar/.venv/bin/python" \
        "$REPO/sidecar/server.py" --backend cuda \
        --vlm-url http://127.0.0.1:8000/v1 --vlm "$MODEL" \
        --llm-url http://127.0.0.1:8000/v1 --llm "$MODEL" \
        --embed-device cpu --socket /tmp/vlm.sock --data-dir "$REPO/data" \
        > "$RUN/sidecar.log" 2>&1 < /dev/null & )
    for _ in $(seq 1 60); do [ -S /tmp/vlm.sock ] && break; sleep 5; done
    [ -S /tmp/vlm.sock ] || die "sidecar did not come up, see $RUN/sidecar.log"
fi

# 3. backend ------------------------------------------------------------
if curl -sf -m 3 http://127.0.0.1:8080/api/health >/dev/null 2>&1; then
    say "backend already up"
else
    say "starting backend + camera (CAM_SRC=${CAM_SRC:-usb})"
    rm -f "$REPO/data/incoming"/*.jpg 2>/dev/null
    # 60s, not the 20s default: one describe is two VLM turns and costs ~14s,
    # and a client timeout mid-describe drops the connection and stalls capture.
    ( cd "$REPO" && setsid "$REPO/.venv/bin/python" -m mneme \
        --data-dir ./data --sidecar /tmp/vlm.sock --bind 0.0.0.0:8080 \
        --sidecar-timeout-ms 60000 --capture-fps "$CAM_FPS" --camera-cmd "$CAM" \
        > "$RUN/mneme.log" 2>&1 < /dev/null & )
    for _ in $(seq 1 40); do
        curl -sf -m 3 http://127.0.0.1:8080/api/health >/dev/null 2>&1 && break
        sleep 3
    done
    curl -sf -m 3 http://127.0.0.1:8080/api/health >/dev/null 2>&1 \
        || die "backend did not come up, see $RUN/mneme.log"
fi

# 4. telegram bot (optional) --------------------------------------------
# The bot reads TELEGRAM_API_KEY from .env itself; this only decides whether
# there is any point in starting it.
if [ -z "${MNEME_TELEGRAM_TOKEN:-}" ] && ! grep -qs '^TELEGRAM_API_KEY=.' "$REPO/.env"; then
    say "no TELEGRAM_API_KEY in .env; skipping the telegram bot"
elif [ -n "$(pids_for '[t]elegram_bot.py')" ]; then
    say "bot already polling"
else
    say "starting telegram bot"
    ( cd "$REPO" && setsid "$REPO/.venv/bin/python" bot/telegram_bot.py \
        > "$RUN/bot.log" 2>&1 < /dev/null & )
fi

echo
status_all
