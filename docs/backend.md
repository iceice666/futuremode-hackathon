# 離線視覺記憶 — 後端契約 v1.3

Rust 後端(iceice666 / 後端 agent)的實作真相來源。對外承諾在 [`docs/spec.md`](./spec.md),推論契約在 [`docs/sidecar.md`](./sidecar.md)。**三份文件的節號連續、不重複**:

- `spec.md` §0 全域約定、§2 對外 HTTP API、§5 分工、§6 時間表、§7 風險
- 本文件 §1 SQLite schema、§3.3 pipeline 形狀、§4 假資料 seed、§8 後端實作契約(§8.5 除外)
- `sidecar.md` §3.1 wire protocol、§3.2 prompt 契約、§8.5 mock sidecar

改動這裡的任何一節要先在群組講一聲並更新版本號。§1 的 schema、§8.2 的選型都算契約。

**v1.3 改了什麼**:把 §3.1 / §3.2 / §8.5 搬到 `docs/sidecar.md`,內容與節號原樣保留。Rust 端行為、CLI 預設值、schema 全都沒動。

**v1.2 改了什麼**:從 `spec.md` v1.1 把 §1 / §3 / §4 / §8 原封不動搬到本文件,節號與內容都沒動,只把跨節引用改成跨檔引用。沒有任何行為或 JSON 變更。

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

## 3. Rust ↔ Python sidecar 契約(iceice666 內部用)

### 3.1 Wire protocol

搬到 [`docs/sidecar.md` §3.1](./sidecar.md#31-wire-protocol)。Rust 端只要記住:一條連線、同時一個 in-flight 請求、斷線每 2 秒重連。

### 3.2 Prompt 契約

搬到 [`docs/sidecar.md` §3.2](./sidecar.md#32-prompt-契約)。Rust 端的責任是組 `Answer.context`:序號 `[1]`–`[3]` 跟 `citations` 同序,**時間在這裡就轉成 `Asia/Taipei` 中文格式**(`spec.md` §0「時區責任」的唯一例外),最多 3 筆。回來的 `answer` 不做二次加工。

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

- **seed 一定要順便寫 `embeddings`**,不然 `/api/ask` 在假資料上直接死,T+16h 那一格就過不了。沒有真模型時用 `sidecar.md` §8.5 的 deterministic mock embedding,維度跟 `--embed-dim` 一致
- 所有事件 `source` 標 `seed`
- 縮圖 320px 寬純色 + 事件序號文字;原圖 1280×720 同款。**不要為了假資料去找素材**
- 同一組 `--seed <n>` 產出完全一樣的資料,三個人可以對同一份資料 debug
- 跑完印出 `event_count` 和時間範圍,方便確認有沒有產成功

summary 寫得像真的:「有人把馬克杯放到桌子右側」「桌面沒有變化,只有窗外光線變暗」這種。

---

## 8. 後端實作契約(iceice666 / 後端 agent 照這節做)

`spec.md` §2 是對外承諾,這節是怎麼把它做出來。**這節的選型不要自己換** —— 換了會撞到 Orin 上的編譯地獄,那是 48 小時裡最貴的失誤。

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
│       ├── mod.rs        # router 組裝 + 錯誤型別 → spec.md §2.6 的 code
│       ├── health.rs
│       ├── events.rs
│       ├── frames.rs
│       ├── ask.rs
│       └── stream.rs
├── sidecar/
│   ├── server.py
│   ├── prompts.py        # sidecar.md §3.2 的 system prompt
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
| middleware | `tower-http` | 0.5(`cors`,`fs`,`trace`) | 靜態檔案 + CORS 都靠它,spec.md §2.7 |
| SQLite | `rusqlite` | 0.31(`bundled`) | bundled 免裝系統 lib,Orin 上省事。**不要用 sqlx**:不需要 async DB,見 8.6 |
| 連線池 | `r2d2` + `r2d2_sqlite` | 0.8 / 0.24 | |
| ID | `ulid` | 1 | |
| 時間 | `chrono`(`serde`) + `chrono-tz` | 0.4 / 0.9 | `chrono-tz` 只為 sidecar.md §3.2 的台北時間 |
| JSON | `serde` / `serde_json` | 1 | |
| CLI | `clap` | 4(`derive`,`env`) | `env` feature 讓 flag 直接吃 `MNEME_*`,不用自己讀 |
| log | `tracing` + `tracing-subscriber` | 0.1 / 0.3 | |
| 影像 | `image` | 0.25 | 縮圖、灰階、diff、seed 佔位圖全用它。**不要引 opencv** |
| 攝影機 | `nokhwa` | 0.10(`input-native`) | 純 Rust 走 v4l2,不用編 opencv |
| 錯誤 | `anyhow` + `thiserror` | 1 | 內部用 anyhow,API 邊界用 thiserror 對應 spec.md §2.6 |

`nokhwa` 是唯一的高風險依賴,退路寫在 `spec.md` §7,T+30h 前要實測。

### 8.3 CLI 與環境變數

**這張表是唯一真相**,README 只放連結。CLI flag 優先於 env,env 優先於預設值。

| flag | env | 預設 | 說明 |
|---|---|---|---|
| `--data-dir` | `MNEME_DATA_DIR` | `./data` | 原圖 / 縮圖 / DB 的根。DB 裡存相對此處的路徑(spec.md §0) |
| `--db` | `MNEME_DB` | `<data-dir>/memory.db` | |
| `--camera` | `MNEME_CAMERA` | `/dev/video0` | |
| `--camera-cmd` | `MNEME_CAMERA_CMD` | 無 | 給定時**不用 nokhwa**:跑這個外部命令寫 JPEG 到 `<data-dir>/incoming`,capture 改成 watch 該目錄並在讀完後刪檔。`nokhwa` 掛掉時的退路(spec.md §7) |
| `--no-camera` | `MNEME_NO_CAMERA` | `false` | 不開 capture,純 serve 現有資料。`health.mode` = `seed-only` |
| `--sidecar` | `MNEME_SIDECAR` | `/tmp/vlm.sock` | |
| `--mock-sidecar` | `MNEME_MOCK_SIDECAR` | `false` | 見 `sidecar.md` §8.5 |
| `--bind` | `MNEME_BIND` | `0.0.0.0:8080` | |
| `--static-dir` | `MNEME_STATIC_DIR` | `./web` | spec.md §2.7 |
| `--capture-fps` | `MNEME_CAPTURE_FPS` | `2` | |
| `--diff-threshold` | `MNEME_DIFF_THRESHOLD` | `12.0` | 64×64 灰階 mean absolute diff,0–255 尺度。現場光線不同務必調 |
| `--cooldown-ms` | `MNEME_COOLDOWN_MS` | `4000` | 觸發後的冷卻時間 |
| `--ask-min-score` | `MNEME_ASK_MIN_SCORE` | `0.35` | 低於此分視為「沒看到」,spec.md §2.4 |
| `--embed-dim` | `MNEME_EMBED_DIM` | `1024` | 必須跟 sidecar 的模型一致,啟動時對 `meta.embed_dim` 檢查 |
| `--sidecar-timeout-ms` | `MNEME_SIDECAR_TIMEOUT_MS` | `20000` | `Describe` / `Answer` 用;`Embed` 用 `5000` |

### 8.4 offline 偵測

- 背景 task,每 5 秒跑一次 `tokio::net::TcpStream::connect("1.1.1.1:443")`,套 1 秒 timeout
- 成功 → `offline = false`,失敗 → `true`。結果放 `AtomicBool`,handler 只讀
- **絕不在 request handler 裡等網路**。拔線時 health 一定要秒回,不然畫面會卡住
- 不用 ICMP ping(要 root),不用 DNS 查詢(系統快取會騙人)
- 拔線後最慢 6 秒畫面翻成 `true`。demo 前先演一次確認節奏,講稿要配合這個延遲

### 8.5 mock sidecar

搬到 [`docs/sidecar.md` §8.5](./sidecar.md#85-mock-sidecar)。實作在 `src/sidecar.rs`,跟真 sidecar 共用同一個 trait;`health.sidecar` 一律誠實回 `mock`。

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

最後一條拒答測試是硬性驗收 —— 過不了不算做完(spec.md §2.4)。
