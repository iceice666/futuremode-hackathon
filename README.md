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
      │  GStreamer → JPEG,21 fps(cv2 開不了這兩顆鏡頭 — operations.md)
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
         │                       │  -AWQ,一顆兼兩角            │
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

## 快速開始

### 沒有硬體也能跑

假資料 + in-process 的確定性假模型,整條 API 與 web UI 都能跑,不需要 GPU:

```bash
python3 -m venv .venv && .venv/bin/pip install -e .
# cv2 不在 pyproject 的相依裡 —— Jetson 上它來自系統 site-packages(見 operations.md)。
# 離開 Jetson 就自己補一顆:
.venv/bin/pip install opencv-python-headless

.venv/bin/python -m mneme.seed --out data/memory.db --data-dir ./data --hours 8 --count 60 --seed 42
.venv/bin/python -m mneme --no-camera --mock-sidecar
```

開 `http://127.0.0.1:8080`。`/api/health` 會誠實回 `sidecar: "mock"`、`mode: "seed-only"`。

### 在 Jetson Orin 上跑真的

需求:

- NVIDIA Jetson Orin ｜ JetPack 6.x ｜ CUDA 12.x ｜ 15.6GB 統一記憶體
- Python 3.10(JetPack 6 內建)
- `docker` 與 nvidia runtime —— vLLM 跑在 [jetson-containers](https://github.com/dusty-nv/jetson-containers) 的映像裡
- 攝影機:USB(預設)或 ribbon cable 上的 IMX219 CSI

安裝步驟見 [`docs/operations.md`](./docs/operations.md#首次安裝) —— 三個 venv 的邊界與
Jetson 上幾個必須指定來源的 wheel 都寫在那裡。裝完之後:

```bash
./start.sh            # vLLM → sidecar → backend →(有 token 才啟)telegram bot
./start.sh status     # 四個各自活著沒
./start.sh stop       # 反序停掉

CAM_SRC=csi ./start.sh   # 改用 ribbon cable 上的 IMX219
```

順序不能換:sidecar 的 VLM/LLM 由 vLLM 提供,backend 又等 sidecar 的 socket。
log 落在 `./run/`。

開 `http://<orin-ip>:8080` 看 timeline。畫面面板預設播 `/api/frames/live.mjpg` 的即時
串流,點任一事件則釘在那一刻的截圖;串流開不起來(seed-only、backend 重啟、相機打嗝)
會自動退回輪詢最新保留的畫格並持續重試。

---

## 文件

| 文件 | 內容 |
|---|---|
| [`docs/spec.md`](./docs/spec.md) | 對外介面契約:全域約定、HTTP API、分工邊界、風險與退路 |
| [`docs/backend.md`](./docs/backend.md) | 後端實作契約:SQLite schema、pipeline 形狀、seed、CLI 與環境變數、驗收清單 |
| [`docs/sidecar.md`](./docs/sidecar.md) | 推論契約:unix socket wire protocol、prompt 契約、mock sidecar |
| [`docs/operations.md`](./docs/operations.md) | 部署與現場運維:安裝、`start.sh` 實際指令、驗證方式、Jetson 上踩過的坑 |
| [`vLLM/README.md`](./vLLM/README.md) | 在 Jetson 上把 Qwen2.5-VL 服起來的完整步驟 |

前三份是契約,有獨立版本號,改動要先講。`docs/operations.md` 是運維筆記。

---

## API

完整契約見 [`docs/spec.md`](./docs/spec.md) §2。核心幾支:

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
