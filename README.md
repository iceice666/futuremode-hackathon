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
  │  Rust (tokio)                           │
  │                                         │
  │  capture ─► change_filter ─► store ─────┼──► SQLite (WAL)
  │                  │              ▲       │      events / frames
  │                  │              │       │      embeddings (f32 blob)
  │                  ▼              │       │
  │            [unix socket]        │       │
  │                  │              │       │
  │  axum HTTP API ◄─┴──────────────┘       │
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

**2. 推論留在 Python,orchestration 留在 Rust。** Jetson 上的 CUDA 生態綁在 Python,硬要用 Rust 綁定只會把時間花在編譯而不是產品。Rust 負責取樣、排程、儲存、檢索與 HTTP,透過 unix socket 上的 line-delimited JSON 跟 sidecar 溝通,兩邊各自發揮。

**3. 向量不用 extension。** 事件量級在數千筆,Rust 端直接暴力算 cosine 相似度,延遲遠低於一次 VLM 呼叫。省下的是編譯與部署的複雜度。

### 為什麼需要本地硬體

這個系統每秒都在看畫面。用雲端 VLM API 意味著把一個空間的連續影像持續傳給第三方——沒有人會把這種東西裝在自己房間。本地推論不是最佳化選擇,是這個產品能不能存在的前提。

---

## 執行方式

### 需求

- NVIDIA Jetson Orin ｜ JetPack 6.x ｜ CUDA 12.x
- Rust 1.7x+、Python 3.10+
- USB 攝影機

### 啟動

```bash
# 1. 推論 sidecar(先跑,Rust 端會等 socket)
cd sidecar
pip install -r requirements.txt
python server.py --socket /tmp/vlm.sock

# 2. 主程式
cargo run --release -- \
    --db memory.db \
    --camera /dev/video0 \
    --sidecar /tmp/vlm.sock \
    --bind 0.0.0.0:8080

# 3. LINE bot(選用)
cd bot && python main.py --api http://localhost:8080
```

開 `http://<orin-ip>:8080` 看 timeline。

### 沒有 Orin 也想試

```bash
cargo run --bin seed -- --out memory.db --hours 8 --count 60   # 產生示範資料
cargo run --release -- --db memory.db --no-camera --mock-sidecar
```

### 環境變數

| 變數 | 預設 | 說明 |
|---|---|---|
| `MNEME_DIFF_THRESHOLD` | `12.0` | change filter 閾值,現場光線不同務必調整 |
| `MNEME_COOLDOWN_MS` | `4000` | 觸發後的冷卻時間 |
| `MNEME_CAPTURE_FPS` | `2` | 取樣率 |

---

## API

完整契約見 [`SPEC.md`](./docs/SPEC.md)。核心三支:

- `GET /api/events` — 事件列表,支援時間範圍與游標分頁
- `POST /api/ask` — 自然語言問答,回答附事件引用與縮圖
- `GET /api/health` — 含 `offline` 欄位,實際偵測外網可達性

---

## 素材與既有資源揭露

依大會規則,以下為非本次賽期間產出的內容:

| 項目 | 來源 | 授權 |
|---|---|---|
| VLM 模型權重 |  |  |
| Embedding 模型 |  |  |
| LLM 權重 |  |  |
| JetPack / CUDA | NVIDIA 官方映像 | NVIDIA 授權條款 |
| Rust / Python 相依套件 | 見 `Cargo.toml`、`requirements.txt` | 各自授權 |

本專案程式碼於 2026/09/04–09/06 賽期內撰寫,未使用團隊既有專案程式。示範用的 seed 資料為賽期內於現場錄製。

> 交件前把上表的括號填掉。空著比寫錯還糟。

---


## 授權

MIT
