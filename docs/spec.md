# 離線視覺記憶 — 介面契約 v1

三人平行開工用。**這份文件是唯一真相來源**,改動要先在群組講一聲並更新版本號。

目標:改任何一層都不需要另外兩層在場。Bryan 和 coffeecat 從第一小時就接假資料開工,iceice666 那邊通了才換真的。

---

## 0. 全域約定

| 項目 | 規則 |
|---|---|
| 時間 | 一律 ISO 8601 + UTC,例 `2026-09-05T14:03:21.482Z`。前端自己轉 `Asia/Taipei`,後端絕不存本地時間 |
| ID | `evt_` / `frm_` 前綴 + ULID,例 `evt_01JBQ...`。可排序,不用另外排 |
| 錯誤 | HTTP 4xx/5xx + `{"error": {"code": "...", "message": "..."}}`,code 用 SCREAMING_SNAKE |
| Base URL | `http://<orin-ip>:8080`,demo 時走區網。不做 auth,48 小時別浪費在這 |
| 縮圖 | 一律 JPEG,寬 320px,路徑由 API 給,前端不要自己拼 |

---

## 1. SQLite Schema

單一檔案 `memory.db`,開 WAL。向量不用 extension,幾千筆事件在 Rust 端暴力算 cosine 完全夠快,別為了 sqlite-vec 卡編譯。

```sql
PRAGMA journal_mode = WAL;

-- 每一張被保留下來的畫面
CREATE TABLE frames (
    id          TEXT PRIMARY KEY,           -- frm_<ulid>
    ts          TEXT NOT NULL,              -- ISO 8601 UTC
    path        TEXT NOT NULL,              -- 原圖絕對路徑
    thumb_path  TEXT NOT NULL,
    width       INTEGER NOT NULL,
    height      INTEGER NOT NULL
);
CREATE INDEX idx_frames_ts ON frames(ts);

-- VLM 描述出來的一件事
CREATE TABLE events (
    id          TEXT PRIMARY KEY,           -- evt_<ulid>
    ts          TEXT NOT NULL,
    frame_id    TEXT NOT NULL REFERENCES frames(id),
    summary     TEXT NOT NULL,              -- 一句話,中文,VLM 產出
    objects     TEXT NOT NULL DEFAULT '[]', -- JSON array of string,例 ["mug","backpack"]
    confidence  REAL NOT NULL DEFAULT 1.0,
    source      TEXT NOT NULL DEFAULT 'vlm' -- 'vlm' | 'seed' | 'manual'
);
CREATE INDEX idx_events_ts ON events(ts);

-- 向量另放一張表,避免掃 events 時把 blob 一起拉進記憶體
CREATE TABLE embeddings (
    event_id    TEXT PRIMARY KEY REFERENCES events(id) ON DELETE CASCADE,
    dim         INTEGER NOT NULL,
    vec         BLOB NOT NULL               -- f32 little-endian,長度 = dim * 4
);

-- demo 時要秀「問過什麼」很好用,也方便事後 debug prompt
CREATE TABLE queries (
    id          TEXT PRIMARY KEY,
    ts          TEXT NOT NULL,
    question    TEXT NOT NULL,
    answer      TEXT NOT NULL,
    cited       TEXT NOT NULL DEFAULT '[]', -- JSON array of event_id
    latency_ms  INTEGER NOT NULL
);
```

`source` 這欄很重要:demo 前預錄的資料標 `seed`,現場即時產生的標 `vlm`,前端可以用不同顏色標示,證明是真的即時在跑。

---

## 2. 對外 HTTP API(Bryan / coffeecat 只需要看這節)

### `GET /api/health`

Demo 第一個要秀的東西。

```json
{
  "status": "ok",
  "device": "orin",
  "vlm_model": "SmolVLM2-2.2B-Instruct",
  "llm_model": "qwen2.5-7b-instruct-q4",
  "offline": true,
  "uptime_s": 4821,
  "capture_fps": 1.8,
  "queue_depth": 0,
  "event_count": 342
}
```

`offline` 是實際去 ping 外網失敗才回 true,不要寫死 —— 拔網路線那一刻畫面上要看得到它變 true。

### `GET /api/events`

Query params:`from`(ISO,選填)、`to`(選填)、`limit`(預設 50,上限 200)、`cursor`(上一頁最後一筆的 id)、`q`(關鍵字,對 summary 做 LIKE,不是語意搜尋)

```json
{
  "events": [
    {
      "id": "evt_01JBQ3M8...",
      "ts": "2026-09-05T14:03:21.482Z",
      "summary": "一個人把黑色後背包放在桌子左側,然後離開",
      "objects": ["person", "backpack", "desk"],
      "confidence": 0.87,
      "source": "vlm",
      "thumb_url": "/api/frames/frm_01JBQ3M8.../thumb"
    }
  ],
  "next_cursor": "evt_01JBQ2X1..."
}
```

沒有更多資料時 `next_cursor` 給 `null`。

### `GET /api/frames/{id}/thumb`

回 `image/jpeg`。加 `?full=1` 回原圖。

### `POST /api/ask`

LINE bot 和 web 都打這支,行為完全一致。

```json
// Request
{ "question": "我的充電器最後一次出現是什麼時候", "top_k": 5 }

// Response
{
  "answer": "最後一次是今天下午 2:03,在桌子左側,之後就沒有再出現過。",
  "citations": [
    {
      "event_id": "evt_01JBQ3M8...",
      "ts": "2026-09-05T14:03:21.482Z",
      "summary": "桌上有一條白色充電線,旁邊放著一個馬克杯",
      "thumb_url": "/api/frames/frm_01JBQ3M8.../thumb",
      "score": 0.81
    }
  ],
  "latency_ms": 1840
}
```

約定:`citations` 最多 3 筆,依 score 由高到低。`answer` 用中文,一到兩句。找不到相關事件時 `answer` 要明講「沒有看到」,`citations` 給空陣列 —— **不准硬掰**,評審一定會問一個沒發生過的事情。

### `GET /api/stream`(SSE)

即時事件推播。web timeline 靠這個不用 polling。

```
event: observed
data: {"id":"evt_...","ts":"...","summary":"...","thumb_url":"..."}

event: heartbeat
data: {"ts":"..."}
```

Heartbeat 每 15 秒一次,前端拿來判斷連線還活著。

---

## 3. Rust ↔ Python sidecar 契約(iceice666 內部用)

推論留在 Python,Rust 不碰 CUDA。Unix socket `/tmp/vlm.sock`,line-delimited JSON,一行一個請求、一行一個回應。

```rust
#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum SidecarRequest {
    Describe { req_id: String, image_path: PathBuf },
    Embed    { req_id: String, text: String },
    Answer   { req_id: String, question: String, context: Vec<String> },
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum SidecarResponse {
    Described { req_id: String, summary: String, objects: Vec<String>, ms: u64 },
    Embedded  { req_id: String, vec: Vec<f32>, ms: u64 },
    Answered  { req_id: String, answer: String, ms: u64 },
    Failed    { req_id: String, code: String, message: String },
}
```

Pipeline 形狀,用 tokio channel 串,每段各自有 backpressure:

```
capture (1-2 fps)
   │  Frame { ts, mat }
   ▼
change_filter        ← 這關擋掉 9 成畫面,是 Orin 撐得住的關鍵
   │  Frame
   ▼
vlm_sidecar (單一 worker,有界 channel 容量 4,滿了就丟舊的)
   │  Event
   ▼
store (SQLite + embed) ──► SSE broadcast
```

`change_filter` 先用最笨的做法:灰階 downscale 到 64x64,算 mean absolute diff,超過閾值才放行,連續放行時加冷卻時間避免同一件事被描述十次。閾值開成環境變數,現場光線不同一定要調。

---

## 4. 假資料 seed(第一小時就要有)

iceice666 先產這個,另外兩人整場都可以離線開發。

```bash
# 產生過去 8 小時、約 60 筆假事件 + 對應縮圖
cargo run --bin seed -- --out memory.db --hours 8 --count 60
```

縮圖用純色 + 文字的佔位圖就好,不要為了假資料去找素材。summary 寫得像真的:「有人把馬克杯放到桌子右側」「桌面沒有變化,只有窗外光線變暗」這種。

---

## 5. 分工邊界

| 人 | 負責 | 交付的東西 |
|---|---|---|
| iceice666 | capture / filter / sidecar / store / 所有 HTTP endpoint | 一個跑得起來的 binary + seed 工具 |
| Bryan | timeline web UI、health 面板 | 靜態檔案,由 Rust server 直接 serve |
| coffeecat | LINE bot(webhook → `/api/ask` → 回文字 + 圖) | 獨立 process,只依賴第 2 節 |

**唯一的跨人依賴是第 2 節的 JSON 格式。** 其他都各做各的。

---

## 6. 時間表對照

- **T+5h** — 模型在 Orin 上跑通、`seed` 可用、health endpoint 回真的東西
- **T+16h** — 三邊各自接假資料能動
- **T+30h** — 真 pipeline 串通,`/api/ask` 回得出合理答案
- **T+40h** — 錄好 demo 用的 seed 資料,場景演過三次
- **T+40h 之後** — 功能凍結,只修 bug 和做簡報

---

## 7. 已知風險與退路

| 風險 | 退路 |
|---|---|
| VLM 在 Orin 上塞不下或太慢 | 換更小的模型;再不行改成 YOLO 物件偵測 + 模板句子生成,demo 照樣成立 |
| LLM 問答品質差 | `/api/ask` 退化成純向量檢索,只回事件卡片不回自然語言 |
| 現場網路爛,LINE webhook 進不來 | Web UI 要有一個功能完全相同的問答輸入框,當主 demo |
| 相機在現場光線下 filter 亂觸發 | 閾值走環境變數,現場調;真的不行就固定間隔取樣 |

每一條退路都要在 T+30h 前實際測過一次,不能只寫在紙上。
