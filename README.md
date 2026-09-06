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
  USB camera
      │  1–2 fps
      ▼
  ┌─────────────────────────────────────────┐
  │  Python (asyncio + FastAPI)             │
  │                                         │
  │  capture ─► change_filter ─► store ─────┼──► SQLite (WAL)
  │                  │              ▲       │      events / frames
  │                  │              │       │      embeddings (f32 blob)
  │                  ▼              │       │
  │            [unix socket]        │       │
  │                  │              │       │
  │  FastAPI HTTP ◄──┴──────────────┘       │
  └──────┬──────────────────┬───────────────┘
         │                  │ /tmp/vlm.sock
         │                  ▼
         │        ┌──────────────────────────┐
         │        │  Python sidecar (CUDA)   │
         │        │   VLM  — 影像 → 一句話    │
         │        │   embed — 句子 → 向量     │
         │        │   LLM  — 檢索結果 → 回答  │
         │        └──────────────────────────┘
         │
         ├─► Web timeline (SSE 即時更新)
         └─► LINE bot (POST /api/ask)
```

### 三個關鍵設計決定

**1. change filter 擋在 VLM 前面。** VLM 是整條 pipeline 唯一昂貴的一段。畫面先降到 64×64 灰階算 mean absolute diff,超過閾值才放行,並加冷卻時間避免同一件事被重複描述。實測擋掉約九成畫面,這是 Orin 能即時運作的關鍵。

**2. 推論跟 orchestration 分成兩個 process。** VLM 載入要吃掉數 GB GPU 記憶體、啟動要數十秒,而且模型崩了不該把 HTTP server 一起帶走。所以推論獨立成 sidecar,主程式負責取樣、排程、儲存、檢索與 HTTP,兩邊用 unix socket 上的 line-delimited JSON 溝通。兩邊都是 Python,但**不共用 process、不共用 venv** —— 主程式的環境裡沒有 torch,想重啟模型不用重啟 API。

**3. 向量不用 extension。** 事件量級在數千筆,啟動時把 embeddings 全讀進一塊 `float32` 陣列,一次 numpy matmul 算完 cosine,延遲遠低於一次 VLM 呼叫。省下的是部署複雜度。

### 為什麼需要本地硬體

這個系統每秒都在看畫面。用雲端 VLM API 意味著把一個空間的連續影像持續傳給第三方——沒有人會把這種東西裝在自己房間。本地推論不是最佳化選擇,是這個產品能不能存在的前提。

---

## 執行方式

### 需求

- NVIDIA Jetson Orin ｜ JetPack 6.x ｜ CUDA 12.x
- Python 3.10+(JetPack 6 內建)、系統 `cv2`(JetPack 內建,不要 pip 裝)
- USB 攝影機

### 啟動

```bash
# 1. 推論 sidecar(先跑,主程式會等 socket)
cd sidecar
pip install -r requirements.txt
python server.py --socket /tmp/vlm.sock

# 2. 主程式
python -m venv --system-site-packages .venv && .venv/bin/pip install -e .
.venv/bin/python -m mneme \
    --data-dir ./data \
    --camera /dev/video0 \
    --sidecar /tmp/vlm.sock \
    --bind 0.0.0.0:8080

# 3. LINE bot(選用)
cd bot && python main.py --api http://localhost:8080
```

開 `http://<orin-ip>:8080` 看 timeline。

### 沒有 Orin 也想試

```bash
.venv/bin/python -m mneme.seed --out data/memory.db --data-dir ./data --hours 8 --count 60 --seed 42
.venv/bin/python -m mneme --no-camera --mock-sidecar
```

`--mock-sidecar` 用 in-process 的確定性假模型取代 CUDA 推論,整條 API 都能跑(規則見 [`docs/sidecar.md` §8.5](./docs/sidecar.md#85-mock-sidecar))。`/api/health` 會誠實回 `sidecar: "mock"`、`mode: "seed-only"`。

### 在 macOS 上驗證真推論

`--mock-sidecar` 驗得到 API 層,但驗不到 sidecar 的 wire protocol(mock 根本不開 socket)。Apple silicon 上可以跑真的:

```bash
python3 -m venv sidecar/.venv
sidecar/.venv/bin/pip install -r sidecar/requirements-mlx.txt
.venv/bin/python scripts/verify_sidecar.py
```

起一個 `sidecar/server.py --backend mlx`(MLX 版的同樣三個模型),把 [`docs/sidecar.md` §3.1 / §3.2](./docs/sidecar.md#31-wire-protocol) 的每一條與 §8.8 的硬性拒答測完,**並用真的攝影機畫面驗 `describe`**(抓兩張要求 summary 不同,擋掉「無視像素、每次吐同一句」的假通過),全過才 exit 0。沒鏡頭就 `--no-camera` 跑 33 條協議檢查。細節與三則實測結論見 [`docs/sidecar.md` §8.9](./docs/sidecar.md#89-在-macos-上驗證真-sidecar)。

### 環境變數

全部 CLI flag 與 `MNEME_*` 環境變數見 [`docs/backend.md` §8.3](./docs/backend.md#83-cli-與環境變數) —— 那裡是唯一真相,這裡不重複列表。最常需要現場調的是 `MNEME_DIFF_THRESHOLD`(change filter 閾值,預設 `12.0`)。

---

## API

對外 API 契約見 [`docs/spec.md`](./docs/spec.md);後端實作契約(schema / seed / 選型 / CLI)見 [`docs/backend.md`](./docs/backend.md);VLM / LLM / embedding sidecar 的 wire protocol 與 prompt 見 [`docs/sidecar.md`](./docs/sidecar.md)。核心三支:

- `GET /api/events` — 事件列表,支援時間範圍與游標分頁
- `POST /api/ask` — 自然語言問答,回答附事件引用與縮圖
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
