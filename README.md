# Mneme

**一台 Jetson Orin,把一個空間裡發生的事變成可以用中文搜尋的記憶。完全離線。**

BUILDMODE GEN-AI HACKATHON 2026 ｜ Track 06 BUILDMODE Open

---

## 這解決什麼問題

監視器只會錄影。當你要找「我的充電器最後放在哪」,錄影幫不上忙——你得先知道要倒帶到哪一段,才找得到那一段。

Mneme 讓裝置持續觀察一個空間,把每一件發生的事即時寫成一句話並建立索引。你直接用中文問,它回答,並附上那一刻的畫面。

**所有影像處理與模型推論都在 Orin 上完成。** 畫面不上傳、不進雲端 API。拔掉網路線,系統照常運作——這是把攝影機放進宿舍、實驗室或家裡的前提,也是雲端方案結構上做不到的事。

---

## 系統架構

```
  USB / CSI camera
      │  GStreamer → JPEG,21 fps(cv2 開不了這兩顆鏡頭,見下方「需求」)
      ▼
  ┌─────────────────────────────────────────┐
  │  Python (asyncio + FastAPI),單一 worker  │
  │                                         │
  │  capture ─┬─────────────────────────────┼──► /api/frames/live.mjpg
  │           │  原始 JPEG,不解碼不轉檔       │    (21 fps 直送瀏覽器)
  │           │                             │
  │           └─► 降到 2/s ─► change_filter │
  │                  │              │       │
  │                  │              ▼       │
  │                  │            store ────┼──► SQLite (WAL)
  │                  ▼              ▲       │      events / frames
  │            [unix socket]        │       │      embeddings (f32 blob)
  │                  │              │       │
  │  FastAPI HTTP ◄──┴──────────────┘       │
  └──────┬──────────────────┬───────────────┘
         │                  │ /tmp/vlm.sock
         │                  ▼
         │        ┌──────────────────────────┐
         │        │  Python sidecar          │
         │        │   embed — 句子 → 向量     │  ← bge-m3,常駐 CPU
         │        │   VLM  — 影像 → 一句話    │ ─┐
         │        │   LLM  — 檢索結果 → 回答  │ ─┼─► HTTP :8000/v1
         │        └──────────────────────────┘  │
         │                                      ▼
         │                       ┌────────────────────────────┐
         │                       │  vLLM (docker)             │
         │                       │  Qwen2.5-VL-3B-Instruct    │
         │                       │  -AWQ,一顆兼兩角           │
         │                       └────────────────────────────┘
         │
         ├─► Web timeline (SSE 即時更新 + MJPEG 即時畫面)
         └─► Telegram bot (POST /api/ask)
```

### 四個關鍵設計決定

**1. change filter 擋在 VLM 前面。** VLM 是整條 pipeline 唯一昂貴的一段。畫面先降到 64×64 灰階算 mean absolute diff,超過閾值才放行,並加冷卻時間避免同一件事被重複描述。實測擋掉約九成畫面,這是 Orin 能即時運作的關鍵。

**2. 推論跟 orchestration 分成兩個 process。** VLM 載入要吃掉數 GB GPU 記憶體、啟動要數十秒,而且模型崩了不該把 HTTP server 一起帶走。所以推論獨立成 sidecar,主程式負責取樣、排程、儲存、檢索與 HTTP,兩邊用 unix socket 上的 line-delimited JSON 溝通。兩邊都是 Python,但**不共用 process、不共用 venv** —— 主程式的環境裡沒有 torch,想重啟模型不用重啟 API。

**3. 向量不用 extension。** 事件量級在數千筆,啟動時把 embeddings 全讀進一塊 `float32` 陣列,一次 numpy matmul 算完 cosine,延遲遠低於一次 VLM 呼叫。省下的是部署複雜度。

**4. 推論交給 vLLM,sidecar 只留 embedding。** 同一顆 Qwen2.5-VL-3B,走 transformers + NF4 每張要 105 秒(bitsandbytes 每次 forward 都重新解量化),走 vLLM 的 fused `awq_marlin` kernel 約 3 秒。所以 sidecar 的 `--vlm-url` / `--llm-url` 指向本機一個 OpenAI 相容的 vLLM,describe 與 answer 共用同一個 engine;sidecar 自己只留 bge-m3,而且 `--embed-device cpu` 跑在 CPU 上——省下的第二個 CUDA context 約 1.5GB,在只有 15.6GB 統一記憶體的 Orin 上是能不能同時開機的差別。

### 為什麼需要本地硬體

這個系統每秒都在看畫面。用雲端 VLM API 意味著把一個空間的連續影像持續傳給第三方——沒有人會把這種東西裝在自己房間。本地推論不是最佳化選擇,是這個產品能不能存在的前提。

### 為什麼是 Telegram bot 而不是 LINE bot

`spec.md` §7 把「現場網路爛,LINE webhook 進不來」列為風險,而那是真的:webhook 需要會場把 inbound 流量路由到他們 LAN 上的一台裝置。Telegram 的 `getUpdates` 是 long polling——只有 outbound,不用公開 URL、不用打 NAT、不用 TLS 憑證,Orin 主動撥出去掛在 socket 上就好。記憶本體(相機、VLM、embedding、回答)全部在本地,拔網路線只會帶走 bot,web UI 照常撐完 demo。

---

## 執行方式

### 需求

- NVIDIA Jetson Orin ｜ JetPack 6.x ｜ CUDA 12.x ｜ 15.6GB 統一記憶體
- Python 3.10(JetPack 6 內建)
- `docker` 與 nvidia runtime —— vLLM 跑在 [jetson-containers](https://github.com/dusty-nv/jetson-containers) 的映像裡
- 攝影機:USB(預設)或 ribbon cable 上的 IMX219 CSI。**兩顆都不經過 `cv2.VideoCapture`** —— 這台的 cv2 沒有 GStreamer 支援,而 CSI 吐的是 cv2 解不了的 10-bit Bayer,所以畫面一律由 `--camera-cmd` 的 GStreamer pipeline 寫成 JPEG(`spec.md` §7)。cv2 仍然要能 import,change filter 用它縮圖。

### 啟動(這是 demo day 實際跑的路徑)

```bash
./start.sh            # vLLM → sidecar → backend →(有 token 才啟)telegram bot
./start.sh status     # 四個各自活著沒
./start.sh stop       # 反序停掉,SIGTERM 不理的升級成 kill -9

CAM_SRC=csi ./start.sh   # 改用 ribbon cable 上的 IMX219
```

順序不能換:sidecar 的 VLM/LLM 由 vLLM 提供,backend 又等 sidecar 的 socket。log 落在 `./run/`。

開 `http://<orin-ip>:8080` 看 timeline。畫面面板預設播 `/api/frames/live.mjpg` 的即時串流,點任一事件則釘在那一刻的截圖;串流開不起來(seed-only、backend 重啟、相機打嗝)會自動退回輪詢最新保留的畫格並持續重試。

<details>
<summary><code>start.sh</code> 實際下的四道指令</summary>

```bash
M=Qwen/Qwen2.5-VL-3B-Instruct-AWQ
export HF_HUB_CACHE=$PWD/vLLM/jetson-containers/data/models/huggingface

# 1. vLLM:一顆 VL 模型同時服務 describe 與 answer,約 40s 載入
#    先 drop page cache —— vLLM 的記憶體預檢把 page cache 算成已用,
#    長時間開機的機器會因此拒絕啟動。
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
( cd vLLM && GPU_MEM_UTIL=0.78 MAX_MODEL_LEN=2048 ./serve_qwen2_5_vl.sh )

# 2. sidecar:VLM/LLM 走 HTTP,只有 embedder 常駐,而且在 CPU 上
sidecar/.venv/bin/python sidecar/server.py --backend cuda \
    --vlm-url http://127.0.0.1:8000/v1 --vlm $M \
    --llm-url http://127.0.0.1:8000/v1 --llm $M \
    --embed-device cpu --socket /tmp/vlm.sock --data-dir ./data

# 3. backend:相機透過 GStreamer,不透過 cv2。--sidecar-timeout-ms 必須調高,
#    一次 describe 是兩輪 VLM 約 14s,撞到 20s 預設會斷線並卡住 pipeline。
.venv/bin/python -m mneme --data-dir ./data --sidecar /tmp/vlm.sock \
    --bind 0.0.0.0:8080 --sidecar-timeout-ms 60000 --capture-fps 21 \
    --camera-cmd 'gst-launch-1.0 v4l2src device=/dev/v4l/by-id/usb-Generic_USB_Camera_200901010001-video-index0 ! image/jpeg,width=1280,height=720,framerate=30/1 ! jpegparse ! videorate ! image/jpeg,framerate=21/1 ! multifilesink location=frame_%05d.jpg'

# 4. telegram bot(選用):自己從 .env 讀 TELEGRAM_API_KEY
.venv/bin/python bot/telegram_bot.py --api http://127.0.0.1:8080
```

相機以 by-id 定址而非 `/dev/video1` —— 重插 USB 後編號會變。`--capture-fps 21` 是給即時串流的,backend 在送進 change filter 前會自己降到 2/s,VLM 的負載不變。

</details>

首次安裝:

```bash
# 主 venv —— 絕對不裝 torch(backend.md §8.2)。python3-venv 這台沒裝,用 virtualenv。
python3 -m virtualenv --system-site-packages .venv && .venv/bin/pip install -e .

# sidecar 自己的 venv,torch 只住在這裡
python3 -m virtualenv sidecar/.venv && sidecar/.venv/bin/pip install -r sidecar/requirements.txt

# vLLM:clone jetson-containers、註冊 nvidia runtime、抓 ~3.5GB 權重
git clone https://github.com/dusty-nv/jetson-containers vLLM/jetson-containers
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
cd vLLM && ./download_model.sh
```

完整步驟與踩過的坑見 [`vLLM/README.md`](./vLLM/README.md)。`vLLM/jetson-containers/` 連權重約 30GB,已 gitignore。

### 沒有 Orin 也想試

```bash
.venv/bin/python -m mneme.seed --out data/memory.db --data-dir ./data --hours 8 --count 60 --seed 42
.venv/bin/python -m mneme --no-camera --mock-sidecar
```

`--mock-sidecar` 用 in-process 的確定性假模型取代 CUDA 推論,整條 API 都能跑(規則見 [`docs/sidecar.md` §8.5](./docs/sidecar.md#85-mock-sidecar))。`/api/health` 會誠實回 `sidecar: "mock"`、`mode: "seed-only"`。

### 驗證真推論的 wire contract

`--mock-sidecar` 驗得到 API 層,但驗不到 sidecar 的 wire protocol(mock 根本不開 socket)。`scripts/verify_sidecar.py` 驗的就是這一段,共 33 條檢查。Apple silicon 上可以離開 Orin 跑真的:

```bash
python3 -m venv sidecar/.venv
sidecar/.venv/bin/pip install -r sidecar/requirements-mlx.txt
.venv/bin/python scripts/verify_sidecar.py
```

起一個 `sidecar/server.py --backend mlx`(MLX 版的同樣三個模型),把 [`docs/sidecar.md` §3.1 / §3.2](./docs/sidecar.md#31-wire-protocol) 的每一條與 §8.8 的硬性拒答測完,**並用真的攝影機畫面驗 `describe`**(抓兩張要求 summary 不同,擋掉「無視像素、每次吐同一句」的假通過),全過才 exit 0。沒鏡頭就 `--no-camera` 跑 33 條協議檢查。細節與三則實測結論見 [`docs/sidecar.md` §8.9](./docs/sidecar.md#89-在-macos-上驗證真-sidecar)。

在 Orin 上驗的是**部署形狀**,所以要帶跟 `start.sh` 一樣的 URL,不要讓它自己 load 一份塞不下 vLLM 旁邊的權重:

```bash
.venv/bin/python scripts/verify_sidecar.py --backend cuda --no-camera \
    --vlm-url http://127.0.0.1:8000/v1 --vlm $M \
    --llm-url http://127.0.0.1:8000/v1 --llm $M \
    --embed-device cpu --data-dir /tmp/verify --db /tmp/verify/memory.db
```

**要給它一份全新 seed 的資料庫。** 它拿最新 16 筆事件當語料,卻對其中一筆 seed 資料(馬克杯那筆)下斷言;真實 capture 跑過之後 seed 資料會被擠出去,於是報三個失敗——那是 fixture 的錯,不是程式的錯。

### 環境變數

全部 CLI flag 與 `MNEME_*` 環境變數見 [`docs/backend.md` §8.3](./docs/backend.md#83-cli-與環境變數) —— 那裡是唯一真相,這裡不重複列表。現場最常要動的兩個:`MNEME_DIFF_THRESHOLD`(change filter 閾值,預設 `12.0`,換場地的光線就要重調)與 `MNEME_SIDECAR_TIMEOUT_MS`(預設 `20000`,但一次 describe 要約 14s,`start.sh` 已改成 `60000`)。

### demo day 出事時先看這裡

| 症狀 | 真正的原因 |
|---|---|
| vLLM 說 `No available memory for the cache blocks` | `--gpu-memory-utilization` 是**開機前的預算檢查**,不是上限,而且分母是系統總記憶體、page cache 也算已用。設**低**反而起不來——這跟 x86 的直覺相反。`sync && echo 3 \| sudo tee /proc/sys/vm/drop_caches` 之後再試(`start.sh` 已經會做)。 |
| torch 噴 `NVML_SUCCESS == r INTERNAL ASSERT FAILED` 或 `NvMapMemAllocInternalTagged: error 12` | 這是 OOM,不是 bug。Tegra 的 iGPU 不完整支援 NVML,所以 torch 在回報 OOM 的路上死在 NVML 裡。 |
| CSI 相機 `Failed to create CaptureSession` | 上一個 pipeline 被砍時把 Argus session 卡住了。`sudo systemctl restart nvargus-daemon`。 |
| capture 看起來跑兩倍快、timeline 交錯兩個房間 | 有第二個 `python -m mneme` 還活著。live stream 永不結束,uvicorn 的 graceful shutdown 會無限等它,期間它已經放掉 port(下一個 start 綁得上、看起來很健康)卻還握著相機、寫進同一個 `data/incoming`。先 `ps` 找,再看程式碼。 |
| 模型權重載入時報奇怪的格式錯誤 | HuggingFace 的 Xet backend 會把中斷後續傳的檔案寫成「大小正確、header 是垃圾」。`HF_HUB_DISABLE_XET=1` 對 huggingface_hub 0.30 無效,要 `pip uninstall hf_xet`。驗權重請解析 safetensors header,不要比對檔案大小。 |

---

## API

對外 API 契約見 [`docs/spec.md`](./docs/spec.md);後端實作契約(schema / seed / 選型 / CLI)見 [`docs/backend.md`](./docs/backend.md);VLM / LLM / embedding sidecar 的 wire protocol 與 prompt 見 [`docs/sidecar.md`](./docs/sidecar.md)。核心三支:

- `GET /api/events` — 事件列表,支援時間範圍與游標分頁
- `POST /api/ask` — 自然語言問答,回答附事件引用與縮圖;cosine 低於 `0.35` 直接拒答,不叫 LLM
- `GET /api/stream` — SSE,新事件即時推播
- `GET /api/frames/live.mjpg` — 攝影機即時畫面,直接轉發相機自己的 JPEG
- `GET /api/health` — 含 `offline` 欄位,實際偵測外網可達性

---

## 素材與既有資源揭露

依大會規則,以下為非本次賽期間產出的內容:

| 項目 | 來源 | 授權 |
|---|---|---|
| VLM 權重(Orin 實際使用) | [`Qwen/Qwen2.5-VL-3B-Instruct-AWQ`](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct-AWQ) | Qwen RESEARCH LICENSE |
| LLM 權重(Orin 實際使用) | 同上 —— 一顆 VL 模型同時擔任 describe 與 answer(見 [`vLLM/README.md`](./vLLM/README.md)) | Qwen RESEARCH LICENSE |
| Embedding 模型 | [`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3)(1024 維) | MIT |
| VLM / LLM 預設值(未指定 `--vlm-url` 時) | [`HuggingFaceTB/SmolVLM2-2.2B-Instruct`](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct)、[`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | 皆 Apache-2.0 |
| macOS 驗證用權重(`--backend mlx`) | `mlx-community` 的同三顆模型量化版 | 同上游(Apache-2.0 / MIT) |
| JetPack / CUDA | NVIDIA 官方映像 | NVIDIA 授權條款 |
| vLLM 推論引擎 | [vllm-project/vllm](https://github.com/vllm-project/vllm),透過 [dusty-nv/jetson-containers](https://github.com/dusty-nv/jetson-containers) 的 Jetson 映像執行 | Apache-2.0 / MIT |
| 前端 vendored JS | React + ReactDOM 18.3.1、[htm](https://github.com/developit/htm) | MIT / Apache-2.0 |
| Python 相依套件 | 見 `pyproject.toml`、`sidecar/requirements.txt`、`sidecar/requirements-mlx.txt` | 各自授權 |

本專案程式碼於 2026/09/04–09/06 賽期內撰寫,未使用團隊既有專案程式。示範用的 seed 資料由賽期內撰寫的 `python -m mneme.seed` 以固定亂數種子程式產生(色塊佔位圖 + 內建中文句庫),不含任何既有素材;實際展示的畫面為現場攝影機即時拍攝。

---

## 授權

MIT
