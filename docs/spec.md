# 離線視覺記憶 — 介面契約 v1.1

三人平行開工用。**這份文件是唯一真相來源**,改動要先在群組講一聲並更新版本號。

目標:改任何一層都不需要另外兩層在場。Bryan 和 coffeecat 從第一小時就接假資料開工,iceice666 那邊通了才換真的。

**v1.1 改了什麼**(v1 → v1.1,對外 JSON 沒有 breaking change,只是把 v1 沒定義的東西釘死):

- §0 加了路徑、時區責任歸屬、向量維度三條全域約定
- §1 加 `meta` 表;`frames.path` 從絕對路徑改成相對 `--data-dir`
- §2 分頁語意釘死(排序方向 / cursor exclusive / 區間開閉);修掉 v1 範例裡 `next_cursor` 跟陣列最後一筆不一致的錯
- §2 `/api/health` 多 `embed_model` `embed_dim` `sidecar` `mode`;SSE `observed` payload 補成跟 `/api/events` 同 shape
- §2 `/api/ask` 拒答門檻數值化(`score < 0.35`);新增 §2.6 錯誤 code 字彙表、§2.7 靜態檔案與 CORS
- §3 加連線併發約定與 §3.2 prompt 契約(含 `context` 的字串格式)
- §4 seed 明訂必須一起產 embeddings
- 新增 §8 後端實作契約:專案佈局、依賴選型、CLI/env(從 README 搬過來,README 只留連結)、offline 偵測、mock sidecar、SQLite 併發、檢索實作、驗收清單

---

## 0. 全域約定

| 項目 | 規則 |
|---|---|
| 時間 | 一律 ISO 8601 + UTC,例 `2026-09-05T14:03:21.482Z`。前端自己轉 `Asia/Taipei`,後端絕不存本地時間 |
| ID | `evt_` / `frm_` 前綴 + ULID,例 `evt_01JBQ...`。可排序,不用另外排 |
| 錯誤 | HTTP 4xx/5xx + `{"error": {"code": "...", "message": "..."}}`,code 用 SCREAMING_SNAKE。完整字彙見 §2.6 |
| Base URL | `http://<orin-ip>:8080`,demo 時走區網。不做 auth,48 小時別浪費在這 |
| 縮圖 | 一律 JPEG,寬 320px,路徑由 API 給,前端不要自己拼 |
| 路徑 | DB 裡一律存相對 `--data-dir` 的路徑,絕不存絕對路徑。整包資料夾搬到別台機器還能跑 |
| 時區責任 | 唯一例外是 `/api/ask` 的 `answer`:那是給人看的中文句子,由後端組 prompt 時就轉成 `Asia/Taipei`(見 §3.2)。其他所有欄位、所有 API,一律 UTC |
| 向量維度 | 全庫單一維度,由 `/api/health.embed_dim` 公告。混維度視為 bug,不做相容 |

---

## 1. SQLite Schema

單一檔案 `memory.db`,開 WAL。向量不用 extension,幾千筆事件在 Rust 端暴力算 cosine 完全夠快,別為了 sqlite-vec 卡編譯。

```sql
PRAGMA journal_mode = WAL;

-- 每一張被保留下來的畫面
CREATE TABLE frames (
    id          TEXT PRIMARY KEY,           -- frm_<ulid>
    ts          TEXT NOT NULL,              -- ISO 8601 UTC
    path        TEXT NOT NULL,              -- 原圖路徑,相對 --data-dir
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

-- key/value,記 schema 版本與當前 embedding 模型/維度
-- 必備 key:schema_version / embed_model / embed_dim
CREATE TABLE meta (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL
);
```

`source` 這欄很重要:demo 前預錄的資料標 `seed`,現場即時產生的標 `vlm`,前端可以用不同顏色標示,證明是真的即時在跑。

---

## 2. 對外 HTTP API(Bryan / coffeecat 只需要看這節)

### 2.1 `GET /api/health`

Demo 第一個要秀的東西。

```json
{
  "status": "ok",
  "device": "orin",
  "vlm_model": "SmolVLM2-2.2B-Instruct",
  "llm_model": "qwen2.5-7b-instruct-q4",
  "embed_model": "bge-m3",
  "embed_dim": 1024,
  "offline": true,
  "uptime_s": 4821,
  "capture_fps": 1.8,
  "queue_depth": 0,
  "event_count": 342,
  "sidecar": "up",
  "mode": "live"
}
```

每個欄位的定義釘死,不要各自解讀:

- `status` — `ok` | `degraded`(sidecar 掛了但還能讀舊資料)。**永遠回 HTTP 200**,前端看這欄,不要看 status code
- `offline` — 實際偵測外網可達性才回 true,不准寫死。做法見 §8.4。拔網路線那一刻畫面上要看得到它變 true
- `capture_fps` — 最近 10 秒滑動窗的實測值,不是設定值。`--no-camera` 時回 `0.0`
- `queue_depth` — `change_filter → vlm_sidecar` 那條有界 channel 當下的長度(容量 4,見 §3.3)。全 pipeline 只有這裡會塞
- `event_count` — `SELECT count(*) FROM events`
- `sidecar` — `up` | `down` | `mock`
- `mode` — `live` | `seed-only`(`--no-camera` 時)
- `embed_dim` — 前端不用管,但 demo 時秀出來很有說服力

### 2.2 `GET /api/events`

Query params:`from`(ISO,選填)、`to`(選填)、`limit`(預設 50,上限 200)、`cursor`(上一頁回的 `next_cursor`)、`q`(關鍵字,對 summary 做 LIKE,不是語意搜尋)

```json
{
  "events": [
    {
      "id": "evt_01JBQ3M8ZK",
      "ts": "2026-09-05T14:03:21.482Z",
      "summary": "一個人把黑色後背包放在桌子左側,然後離開",
      "objects": ["person", "backpack", "desk"],
      "confidence": 0.87,
      "source": "vlm",
      "thumb_url": "/api/frames/frm_01JBQ3M8ZK/thumb"
    },
    {
      "id": "evt_01JBQ2X1PT",
      "ts": "2026-09-05T13:58:04.117Z",
      "summary": "桌面沒有變化,只有窗外光線變暗",
      "objects": ["desk", "window"],
      "confidence": 0.71,
      "source": "vlm",
      "thumb_url": "/api/frames/frm_01JBQ2X1PT/thumb"
    }
  ],
  "next_cursor": "evt_01JBQ2X1PT"
}
```

排序與分頁契約,前端照這個寫翻頁:

- 固定排序 `ts DESC, id DESC`,新的在前。前端不用自己排
- `cursor` 語意是 **exclusive**:回傳嚴格排在該 id 之後的資料,不會重複上一頁最後一筆
- `next_cursor` **一定等於本頁陣列最後一筆的 `id`**。本頁筆數少於 `limit` 時給 `null`,代表沒有更多了
- `from` / `to` 都是**閉區間** `[from, to]`,比對 `events.ts`
- `cursor` 可以跟 `from` / `to` / `q` 併用,條件取交集
- `q` 對 `summary` 做 `LIKE '%q%'`,大小寫不敏感,不碰 `objects`。語意搜尋只存在於 `/api/ask`
- `limit` 超過 200 直接夾到 200,不報錯;`limit <= 0` 或非數字回 `INVALID_PARAM`
- `cursor` 格式不合法回 `INVALID_CURSOR`;格式合法但查不到那筆,當作沒有這個游標,回空陣列 + `next_cursor: null`

### 2.3 `GET /api/frames/{id}/thumb`

回 `image/jpeg`。加 `?full=1` 回原圖(同樣是 JPEG)。id 查不到回 404 `FRAME_NOT_FOUND`。

frame id 一旦產生內容就不會變,所以帶 `Cache-Control: public, max-age=31536000, immutable`,前端可以放心重複請求。

### 2.4 `POST /api/ask`

LINE bot 和 web 都打這支,行為完全一致。

```json
// Request
{ "question": "我的充電器最後一次出現是什麼時候", "top_k": 5 }

// Response
{
  "answer": "最後一次是今天下午 2:03,在桌子左側,之後就沒有再出現過。",
  "citations": [
    {
      "event_id": "evt_01JBQ3M8ZK",
      "ts": "2026-09-05T14:03:21.482Z",
      "summary": "桌上有一條白色充電線,旁邊放著一個馬克杯",
      "thumb_url": "/api/frames/frm_01JBQ3M8ZK/thumb",
      "score": 0.81
    }
  ],
  "latency_ms": 1840
}
```

`question` 必填,1–500 字,超出回 `INVALID_PARAM`。`top_k` 選填,預設 5,夾在 1–20。

檢索與拒答規則,這節是 `/api/ask` 的全部行為:

- `question` 送 sidecar `Embed` 拿向量,對全庫 embeddings 算 cosine,取前 `top_k`(實作見 §8.7)
- `score` 就是**原始 cosine**,範圍 `[-1, 1]`,不正規化、不 rescale、不轉百分比。前端要顯示百分比自己換
- **拒答門檻:最高分 `< 0.35` 就判定沒看到** —— 不呼叫 LLM,`answer` 固定回 `"我沒有看到相關的畫面。"`,`citations` 給 `[]`。門檻走 `--ask-min-score`,現場光線/資料不同要調
- 過門檻:只把 `score >= 門檻` 的送進 prompt,最多 3 筆。`top_k` 只影響檢索寬度,**不影響回傳筆數**
- `citations` 依 `score` 由高到低,最多 3 筆
- LLM 自己也可能判斷 context 不夠而回「沒有看到」(prompt 有明確指示,見 §3.2)。這種情況 `citations` **仍照實附上檢索到的事件** —— 讓評審看得到我們檢索到什麼,只是不硬掰
- 每次呼叫都寫一筆 `queries`,成功失敗都寫,`latency_ms` 一起存
- `latency_ms` = 收到 request 到組完 response,含 embed + LLM
- sidecar 沒連上回 503 `SIDECAR_UNAVAILABLE`;超時回 504 `SIDECAR_TIMEOUT`

**不准硬掰**,評審一定會問一個沒發生過的事情。§8.8 的最後一條 curl 就是在測這個,過不了不算做完。

### 2.5 `GET /api/stream`(SSE)

即時事件推播。web timeline 靠這個不用 polling。`Content-Type: text/event-stream`,不做 gzip。

```
retry: 3000

event: observed
data: {"id":"evt_01JBQ3M8ZK","ts":"2026-09-05T14:03:21.482Z","summary":"一個人把黑色後背包放在桌子左側,然後離開","objects":["person","backpack","desk"],"confidence":0.87,"source":"vlm","thumb_url":"/api/frames/frm_01JBQ3M8ZK/thumb"}

event: heartbeat
data: {"ts":"2026-09-05T14:03:36.000Z"}
```

- `observed` 的 data **跟 `/api/events` 陣列元素是同一個 shape**,前端一個 parser 兩邊共用,收到推播不需要再打一次 API
- Heartbeat 每 15 秒一次,前端拿來判斷連線還活著
- `retry: 3000` 讓 `EventSource` 斷線後三秒自動重連
- broadcast channel 容量 64。慢的 client 掉事件就是掉了,**不重送**;前端重連後自己打 `/api/events` 補齊

### 2.6 錯誤 code 字彙表

| code | HTTP | 什麼時候 |
|---|---|---|
| `INVALID_PARAM` | 400 | query 或 body 欄位型別、範圍不對 |
| `INVALID_CURSOR` | 400 | `cursor` 不是合法 id 格式 |
| `FRAME_NOT_FOUND` | 404 | `/api/frames/{id}` 查不到 |
| `SIDECAR_UNAVAILABLE` | 503 | sidecar 沒連上或連線中斷 |
| `SIDECAR_TIMEOUT` | 504 | sidecar 超過 `--sidecar-timeout-ms` 沒回 |
| `SIDECAR_FAILED` | 502 | sidecar 回了 `Failed` |
| `INTERNAL` | 500 | 其他 |

前端只需要分三類處理:400 系列是自己參數錯、502/503/504 顯示「本地模型忙碌中,稍後再試」、500 顯示通用錯誤。

### 2.7 靜態檔案與 CORS

- Rust server 把 `--static-dir`(預設 `./web`)掛在 `/`,`/api/*` 的路由優先
- 路徑不以 `/api` 開頭又找不到檔案 → 回 `index.html`(SPA fallback),讓 Bryan 用前端 router 沒問題
- Bryan 開發時跑自己的 dev server:Rust 端一律回 `Access-Control-Allow-Origin: *`。反正不做 auth,沒有風險

---

## 3. Rust ↔ Python sidecar 契約(iceice666 內部用)

### 3.1 Wire protocol

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

連線與併發約定:

- Rust 只開**一條** socket 連線,而且**同時只有一個 in-flight 請求**。`req_id` 純粹用來對帳和 log 追蹤,不是為了多工 —— sidecar 端不需要做併發
- 每行一個 JSON,`\n` 結尾。字串裡不得有裸換行(`serde_json` 預設就會轉義,別自己拼字串)
- 收到不認識的 `kind` 回 `Failed { code: "UNKNOWN_KIND" }`,不要直接崩
- 斷線後 Rust 每 2 秒重連。期間 `/api/health.sidecar` = `down`、`status` = `degraded`,`/api/ask` 回 `SIDECAR_UNAVAILABLE`,但 `/api/events` 照常能讀

### 3.2 Prompt 契約

`Answer.context` 的每個元素是**後端組好的一行字串**,格式固定:

```
[1] 2026-09-05 22:03(台北時間) 一個人把黑色後背包放在桌子左側,然後離開
```

- 序號 `[1]`–`[3]` 由後端配,順序跟 `citations` 一致
- **時間在這裡就轉好 `Asia/Taipei` 並寫成中文可讀格式**。LLM 不做時區換算,也不需要知道 UTC 存在 —— 這是 §0「時區責任」那條的唯一例外
- system prompt 固定寫在 `sidecar/prompts.py`,**改它算改契約**,要更新版本號:

```
你是一個離線影像記憶助理。只能根據提供的觀察記錄回答,用一到兩句中文。
記錄裡沒有的事情,直接回「我沒有看到相關的畫面。」——不要推測、不要補完、不要說「可能」。
提到時間就用記錄裡給的時間,不要自己算。
```

`Answer` 回來的 `answer` 後端不做二次加工,原樣放進 response。

### 3.3 Pipeline 形狀

用 tokio channel 串,每段各自有 backpressure:

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
# 產生過去 8 小時、約 60 筆假事件 + 對應縮圖 + embeddings
cargo run --bin seed -- --out data/memory.db --data-dir ./data --hours 8 --count 60 --seed 42
```

- **seed 一定要順便寫 `embeddings`**,不然 `/api/ask` 在假資料上直接死,T+16h 那一格就過不了。沒有真模型時用 §8.5 的 deterministic mock embedding,維度跟 `--embed-dim` 一致
- 所有事件 `source` 標 `seed`
- 縮圖 320px 寬純色 + 事件序號文字;原圖 1280×720 同款。**不要為了假資料去找素材**
- 同一組 `--seed <n>` 產出完全一樣的資料,三個人可以對同一份資料 debug
- 跑完印出 `event_count` 和時間範圍,方便確認有沒有產成功

summary 寫得像真的:「有人把馬克杯放到桌子右側」「桌面沒有變化,只有窗外光線變暗」這種。

---

## 5. 分工邊界

| 人 | 負責 | 交付的東西 |
|---|---|---|
| iceice666 | capture / filter / sidecar / store / 所有 HTTP endpoint | 一個跑得起來的 binary + seed 工具 |
| Bryan | timeline web UI、health 面板 | 靜態檔案放 `web/`,由 Rust server 直接 serve,規則見 §2.7 |
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
| `nokhwa` 在 Jetson 上抓不到 camera | 用 `--camera-cmd`(§8.3)跑 `ffmpeg -f v4l2 … -f image2` 寫檔到 `<data-dir>/incoming`,capture 改成 watch 那個目錄 |

每一條退路都要在 T+30h 前實際測過一次,不能只寫在紙上。

---

## 8. 後端實作契約(iceice666 / 後端 agent 照這節做)

第 2 節是對外承諾,這節是怎麼把它做出來。**這節的選型不要自己換** —— 換了會撞到 Orin 上的編譯地獄,那是 48 小時裡最貴的失誤。

### 8.1 專案佈局

單一 crate、兩個 bin、共用 `lib.rs`。48 小時不要開 workspace。

```
.
├── Cargo.toml
├── src/
│   ├── lib.rs            # 對外導出 config / db / store / search / sidecar
│   ├── main.rs           # bin mneme:pipeline + HTTP server
│   ├── bin/seed.rs       # bin seed
│   ├── config.rs         # clap CLI + env,見 8.3
│   ├── db.rs             # schema 建立 + 所有 SQL
│   ├── capture.rs        # camera → Frame
│   ├── filter.rs         # change_filter
│   ├── sidecar.rs        # unix socket client + mock 實作
│   ├── store.rs          # frames/events/embeddings 寫入 + SSE broadcast
│   ├── search.rs         # in-memory cosine 檢索
│   └── api/
│       ├── mod.rs        # router 組裝 + 錯誤型別 → §2.6 的 code
│       ├── health.rs
│       ├── events.rs
│       ├── frames.rs
│       ├── ask.rs
│       └── stream.rs
├── sidecar/
│   ├── server.py
│   ├── prompts.py        # §3.2 的 system prompt
│   └── requirements.txt
├── web/                  # Bryan,Rust 直接 serve
└── data/                 # 執行時產生,--data-dir 預設值
    ├── memory.db
    ├── frames/
    └── thumbs/
```

### 8.2 依賴選型

| 用途 | crate | 版本 | 為什麼是它 |
|---|---|---|---|
| async runtime | `tokio` | 1(`full`) | |
| HTTP | `axum` | 0.7 | SSE 內建 `axum::response::sse`,不用自己寫 |
| middleware | `tower-http` | 0.5(`cors`,`fs`,`trace`) | 靜態檔案 + CORS 都靠它,§2.7 |
| SQLite | `rusqlite` | 0.31(`bundled`) | bundled 免裝系統 lib,Orin 上省事。**不要用 sqlx**:不需要 async DB,見 8.6 |
| 連線池 | `r2d2` + `r2d2_sqlite` | 0.8 / 0.24 | |
| ID | `ulid` | 1 | |
| 時間 | `chrono`(`serde`) + `chrono-tz` | 0.4 / 0.9 | `chrono-tz` 只為 §3.2 的台北時間 |
| JSON | `serde` / `serde_json` | 1 | |
| CLI | `clap` | 4(`derive`,`env`) | `env` feature 讓 flag 直接吃 `MNEME_*`,不用自己讀 |
| log | `tracing` + `tracing-subscriber` | 0.1 / 0.3 | |
| 影像 | `image` | 0.25 | 縮圖、灰階、diff、seed 佔位圖全用它。**不要引 opencv** |
| 攝影機 | `nokhwa` | 0.10(`input-native`) | 純 Rust 走 v4l2,不用編 opencv |
| 錯誤 | `anyhow` + `thiserror` | 1 | 內部用 anyhow,API 邊界用 thiserror 對應 §2.6 |

`nokhwa` 是唯一的高風險依賴,退路寫在 §7,T+30h 前要實測。

### 8.3 CLI 與環境變數

**這張表是唯一真相**,README 只放連結。CLI flag 優先於 env,env 優先於預設值。

| flag | env | 預設 | 說明 |
|---|---|---|---|
| `--data-dir` | `MNEME_DATA_DIR` | `./data` | 原圖 / 縮圖 / DB 的根。DB 裡存相對此處的路徑(§0) |
| `--db` | `MNEME_DB` | `<data-dir>/memory.db` | |
| `--camera` | `MNEME_CAMERA` | `/dev/video0` | |
| `--camera-cmd` | `MNEME_CAMERA_CMD` | 無 | 給定時**不用 nokhwa**:跑這個外部命令寫 JPEG 到 `<data-dir>/incoming`,capture 改成 watch 該目錄並在讀完後刪檔。`nokhwa` 掛掉時的退路(§7) |
| `--no-camera` | `MNEME_NO_CAMERA` | `false` | 不開 capture,純 serve 現有資料。`health.mode` = `seed-only` |
| `--sidecar` | `MNEME_SIDECAR` | `/tmp/vlm.sock` | |
| `--mock-sidecar` | `MNEME_MOCK_SIDECAR` | `false` | 見 8.5 |
| `--bind` | `MNEME_BIND` | `0.0.0.0:8080` | |
| `--static-dir` | `MNEME_STATIC_DIR` | `./web` | §2.7 |
| `--capture-fps` | `MNEME_CAPTURE_FPS` | `2` | |
| `--diff-threshold` | `MNEME_DIFF_THRESHOLD` | `12.0` | 64×64 灰階 mean absolute diff,0–255 尺度。現場光線不同務必調 |
| `--cooldown-ms` | `MNEME_COOLDOWN_MS` | `4000` | 觸發後的冷卻時間 |
| `--ask-min-score` | `MNEME_ASK_MIN_SCORE` | `0.35` | 低於此分視為「沒看到」,§2.4 |
| `--embed-dim` | `MNEME_EMBED_DIM` | `1024` | 必須跟 sidecar 的模型一致,啟動時對 `meta.embed_dim` 檢查 |
| `--sidecar-timeout-ms` | `MNEME_SIDECAR_TIMEOUT_MS` | `20000` | `Describe` / `Answer` 用;`Embed` 用 `5000` |

### 8.4 offline 偵測

- 背景 task,每 5 秒跑一次 `tokio::net::TcpStream::connect("1.1.1.1:443")`,套 1 秒 timeout
- 成功 → `offline = false`,失敗 → `true`。結果放 `AtomicBool`,handler 只讀
- **絕不在 request handler 裡等網路**。拔線時 health 一定要秒回,不然畫面會卡住
- 不用 ICMP ping(要 root),不用 DNS 查詢(系統快取會騙人)
- 拔線後最慢 6 秒畫面翻成 `true`。demo 前先演一次確認節奏,講稿要配合這個延遲

### 8.5 mock sidecar

`--mock-sidecar` 時不開 socket,in-process 實作跟真 sidecar 同一個 trait:

- `Describe` → 從固定句庫挑一句(拿圖片灰階 mean 當索引),`objects` 用那句對應的固定清單,`ms` 回 `300`
- `Embed` → **deterministic**:用 `DefaultHasher` 吃 text 當 PRNG seed,產 `--embed-dim` 個 f32 再 L2 normalize。同一句話永遠同一個向量。語意相似度當然不像真模型,但**檢索、排序、拒答門檻、citations 組裝全都能測**
- `Answer` → 把 context 第一筆改寫成「最後一次是 <時間>,<summary>」;context 空的時候回 §3.2 的固定拒答句
- `health.sidecar` = `mock`

mock 的存在意義是讓 API 層在沒有 Orin 的機器上完整開發與測試。**它不是 demo 路徑**,`mode` / `sidecar` 兩個欄位一定要誠實反映,不准為了畫面好看寫成 `up`。

### 8.6 SQLite 併發

- 一個 `r2d2::Pool<SqliteConnectionManager>`,size 4
- 寫入路徑只有 store task 一條,不會有寫寫衝突。`busy_timeout` 設 5000ms
- 啟動時跑 `PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA foreign_keys=ON;`
- schema 用 `CREATE TABLE IF NOT EXISTS` 在啟動時建好,寫 `meta.schema_version = 1`。**48 小時不做 migration framework**,schema 改了就砍 db 重 seed
- 掃全庫的查詢包在 `tokio::task::spawn_blocking`,不要卡 runtime

### 8.7 檢索實作

- 啟動時把所有 embeddings 一次讀進記憶體:`RwLock<Vec<(String /* event_id */, Vec<f32>)>>`
- store 寫入新 event 後直接 push 進去,不重讀 DB
- 3000 筆 × 1024 維 f32 ≈ 12MB,暴力 cosine 一次 < 5ms,遠低於一次 VLM 呼叫。**不要引 sqlite-vec**
- 所有向量**入庫前就 L2 normalize**,cosine 退化成 dot product,少一次開根號
- 啟動時偵測到維度跟 `--embed-dim` 不符的向量:log error 並**拒絕啟動**,不要靜靜跳過 —— 混維度會讓分數變成垃圾而且很難查

### 8.8 驗收(交付前自己跑過,全部要過)

```bash
cargo build --release
cargo run --bin seed -- --out data/memory.db --data-dir ./data --hours 8 --count 60 --seed 42
cargo run --release -- --no-camera --mock-sidecar &

# mode=seed-only, sidecar=mock, event_count=60, status=ok
curl -s localhost:8080/api/health | jq

# 2 筆,next_cursor 等於第 2 筆的 id
curl -s 'localhost:8080/api/events?limit=2' | jq

# 帶上一頁的 next_cursor,不得重複第一頁任何一筆
curl -s 'localhost:8080/api/events?limit=2&cursor=<next_cursor>' | jq

# 關鍵字命中,且每筆 summary 都含「馬克杯」
curl -s 'localhost:8080/api/events?q=馬克杯' | jq

# content-type: image/jpeg
curl -sI localhost:8080/api/frames/<frame_id>/thumb

# citations 非空,score 由高到低
curl -s -XPOST localhost:8080/api/ask -H 'content-type: application/json' \
     -d '{"question":"馬克杯放在哪"}' | jq

# answer 明講沒看到,citations 為 []  ← 硬性驗收
curl -s -XPOST localhost:8080/api/ask -H 'content-type: application/json' \
     -d '{"question":"有沒有人在跳舞"}' | jq

# 15 秒內看得到 heartbeat
curl -N localhost:8080/api/stream
```

最後一條拒答測試是硬性驗收 —— 過不了不算做完(§2.4)。
