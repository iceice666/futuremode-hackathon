# 離線視覺記憶 — 後端契約 v1.4

Python 後端(iceice666 / 後端 agent)的實作真相來源。對外承諾在 [`docs/spec.md`](./spec.md),推論契約在 [`docs/sidecar.md`](./sidecar.md)。**三份文件的節號連續、不重複**:

- `spec.md` §0 全域約定、§2 對外 HTTP API、§5 分工、§6 時間表、§7 風險
- 本文件 §1 SQLite schema、§3.3 pipeline 形狀、§4 假資料 seed、§8 後端實作契約(§8.5 除外)
- `sidecar.md` §3.1 wire protocol、§3.2 prompt 契約、§8.5 mock sidecar

改動這裡的任何一節要先在群組講一聲並更新版本號。§1 的 schema、§8.2 的選型都算契約。

**v1.4 改了什麼**:後端實作語言從 Rust 換成 Python(asyncio + FastAPI),§3.3 / §8.1 / §8.2 / §8.4 / §8.6 / §8.7 / §8.8 照新選型重寫,§4 的 seed 換成 `python -m mneme.seed`。**對外沒有 breaking change**:`spec.md` §2 的 JSON、§1 的 schema、§8.3 的 flag / env 名稱與預設值全部原封不動。sidecar 那邊也沒動 —— unix socket 上的 wire protocol 跟 v1.3 完全一樣,只是另一端從 Rust 換成 Python。

**v1.3 改了什麼**:把 §3.1 / §3.2 / §8.5 搬到 `docs/sidecar.md`,內容與節號原樣保留。

**v1.2 改了什麼**:從 `spec.md` v1.1 把 §1 / §3 / §4 / §8 原封不動搬到本文件,節號與內容都沒動,只把跨節引用改成跨檔引用。

---

## 1. SQLite Schema

單一檔案 `memory.db`,開 WAL。向量不用 extension,幾千筆事件用 numpy 一次 matmul 算 cosine 完全夠快,別為了 sqlite-vec 卡編譯。

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

blob 讀寫一律走 `numpy`:寫 `vec.astype("<f4").tobytes()`,讀 `np.frombuffer(blob, dtype="<f4")`。**明寫 `<f4` 不要用 `float32` 別名** —— dtype 字串裡的 `<` 就是 schema 註解那句「little-endian」,不要讓它依賴機器 endianness。

---

## 3. 主程式 ↔ Python sidecar 契約(iceice666 內部用)

後端跟 sidecar 都是 Python,但**還是兩個 process、兩個 venv**,協議一個字都不改。理由是 sidecar 要吃 CUDA / torch,主程式不能被它的載入時間跟 GPU 記憶體綁死;而且模型崩了只該讓 `/api/health.sidecar` 變 `down`,不該把 HTTP server 一起帶走。**不要為了「都是 Python」就 import 進來直接呼叫。**

### 3.1 Wire protocol

搬到 [`docs/sidecar.md` §3.1](./sidecar.md#31-wire-protocol)。主程式端只要記住:一條連線、同時一個 in-flight 請求、斷線每 2 秒重連。

### 3.2 Prompt 契約

搬到 [`docs/sidecar.md` §3.2](./sidecar.md#32-prompt-契約)。主程式的責任是組 `Answer.context`:序號 `[1]`–`[3]` 跟 `citations` 同序,**時間在這裡就轉成 `Asia/Taipei` 中文格式**(`spec.md` §0「時區責任」的唯一例外),最多 3 筆。回來的 `answer` 不做二次加工。

### 3.3 Pipeline 形狀

用 `asyncio.Queue` 串,每段各自有 backpressure:

```
capture (1-2 fps)
   │  Frame { ts, mat }        ← cv2 讀取在 asyncio.to_thread 裡跑
   ▼
change_filter        ← 這關擋掉 9 成畫面,是 Orin 撐得住的關鍵
   │  Frame
   ▼
vlm_sidecar (單一 worker,asyncio.Queue(maxsize=4),滿了就丟舊的)
   │  Event
   ▼
store (SQLite + embed) ──► SSE broadcast
```

- `change_filter` 先用最笨的做法:灰階 downscale 到 64x64,算 mean absolute diff,超過閾值才放行,連續放行時加冷卻時間避免同一件事被描述十次。閾值開成環境變數,現場光線不同一定要調
- 丟舊的寫法固定是 `try: q.put_nowait(f) except asyncio.QueueFull: q.get_nowait(); q.put_nowait(f)`。**絕不用 `await q.put()`** —— 那會讓 capture 被 VLM 拖住,fps 直接崩
- `capture` / `change_filter` 兩段的 cv2 呼叫(`read`、`cvtColor`、`resize`、`imencode`)一律包 `await asyncio.to_thread(...)`。cv2 會放掉 GIL,但不包的話 event loop 會被同步呼叫卡住,SSE heartbeat 和 `/api/health` 就開始抖
- `queue_depth`(`spec.md` §2.1)直接回這條 queue 的 `qsize()`,全 pipeline 只有這裡會塞

---

## 4. 假資料 seed(第一小時就要有)

iceice666 先產這個,另外兩人整場都可以離線開發。

```bash
# 產生過去 8 小時、約 60 筆假事件 + 對應縮圖 + embeddings
python -m mneme.seed --out data/memory.db --data-dir ./data --hours 8 --count 60 --seed 42
```

- **seed 一定要順便寫 `embeddings`**,不然 `/api/ask` 在假資料上直接死,T+16h 那一格就過不了。沒有真模型時用 `sidecar.md` §8.5 的 deterministic mock embedding,維度跟 `--embed-dim` 一致
- 所有事件 `source` 標 `seed`
- 縮圖 320px 寬純色 + 事件序號文字;原圖 1280×720 同款。**不要為了假資料去找素材**。佔位圖用 `cv2.putText` 畫,它**不支援中文** —— 圖上只寫 ASCII 序號,中文只出現在 DB 的 `summary`
- 同一組 `--seed <n>` 產出完全一樣的資料,三個人可以對同一份資料 debug。亂數一律走 `random.Random(seed)` 與 `np.random.default_rng(seed)` 這種**顯式帶 seed 的實例**,不要用 module-level 的 `random.*` / `np.random.*`
- 跑完印出 `event_count` 和時間範圍,方便確認有沒有產成功

summary 寫得像真的:「有人把馬克杯放到桌子右側」「桌面沒有變化,只有窗外光線變暗」這種。

---

## 8. 後端實作契約(iceice666 / 後端 agent 照這節做)

`spec.md` §2 是對外承諾,這節是怎麼把它做出來。**這節的選型不要自己換** —— 換了會撞到 Orin 上的安裝地獄,那是 48 小時裡最貴的失誤。

### 8.1 專案佈局

單一 package、兩個 entrypoint。48 小時不要開 monorepo、不要 src layout。

```
.
├── pyproject.toml
├── mneme/
│   ├── __init__.py
│   ├── __main__.py       # python -m mneme:pipeline + HTTP server
│   ├── app.py            # FastAPI app + lifespan(pipeline task 在這裡起停)
│   ├── config.py         # argparse + env,見 8.3
│   ├── db.py             # schema 建立 + 所有 SQL
│   ├── capture.py        # camera → Frame
│   ├── filter.py         # change_filter
│   ├── sidecar.py        # unix socket client + mock 實作
│   ├── store.py          # frames/events/embeddings 寫入 + SSE broadcast
│   ├── search.py         # in-memory cosine 檢索
│   ├── seed.py           # python -m mneme.seed
│   └── api/
│       ├── __init__.py   # router 組裝 + exception handler → spec.md §2.6 的 code
│       ├── health.py
│       ├── events.py
│       ├── frames.py
│       ├── ask.py
│       └── stream.py
├── sidecar/
│   ├── server.py
│   ├── prompts.py        # sidecar.md §3.2 的 system prompt
│   └── requirements.txt
├── web/                  # Bryan,主程式直接 serve
└── data/                 # 執行時產生,--data-dir 預設值
    ├── memory.db
    ├── frames/
    └── thumbs/
```

**兩個 venv**:主程式 venv 只裝 §8.2 那張表,**不准裝 torch**;sidecar 有自己的 `requirements.txt`。主程式的 venv 必須開 `--system-site-packages`(見 §8.2 的 cv2 那條):

```bash
python -m venv --system-site-packages .venv && .venv/bin/pip install -e .
```

### 8.2 依賴選型

| 用途 | 套件 | 版本 | 為什麼是它 |
|---|---|---|---|
| async runtime | stdlib `asyncio` | 3.10+ | JetPack 6 內建 3.10,不要自己編新版 Python |
| HTTP | `fastapi` | 0.115 | pydantic v2 做 §2 的 request 驗證,錯誤直接映到 `INVALID_PARAM` |
| ASGI server | `uvicorn[standard]` | 0.30 | `[standard]` 帶 uvloop + httptools。**固定單 worker**,見 8.6 |
| SSE | `sse-starlette` | 2.1 | `retry:`、heartbeat、client 斷線清理都有。**不要自己拼 StreamingResponse**,漏清理會累積 task |
| SQLite | stdlib `sqlite3` | — | **不要 sqlalchemy、不要 aiosqlite**:不需要 async DB,見 8.6 |
| 向量 | `numpy` | 1.26 | 檢索、L2 normalize、blob 編解碼全用它,見 8.7 |
| ID | `python-ulid` | 2 | `from ulid import ULID`。**不是 `ulid-py`**,兩個套件 import 名撞在一起,裝錯會炸 |
| 時間 | stdlib `datetime` + `zoneinfo` | — | `zoneinfo` 只為 sidecar.md §3.2 的台北時間;`ZoneInfo("Asia/Taipei")` 要系統有 tzdata,Jetson 有 |
| JSON | stdlib `json` | — | 想換 orjson 等 demo 過了再說 |
| CLI | stdlib `argparse` | — | env 讀取靠 `default=os.environ.get(...)`,precedence 見 8.3 |
| log | stdlib `logging` | — | uvicorn 的 log config 直接沿用,不要另外接框架 |
| 影像 | 系統 `cv2`(JetPack 內建) | 4.x | capture、灰階、diff、縮圖、seed 佔位圖全用它。**絕對不要 `pip install opencv-python`** |

`cv2` 是唯一的高風險依賴,而風險點是**安裝不是編譯**:JetPack 把 `cv2` 裝在系統 site-packages,而 arm64 上 `pip install opencv-python` 有機會現場編譯 —— 一顆 4 小時的地雷。所以 venv 開 `--system-site-packages`,並在啟動時 `import cv2` 失敗就直接 log error 退出,不要留到第一次取樣才炸。抓不到 camera 的退路寫在 `spec.md` §7,T+30h 前要實測。

### 8.3 CLI 與環境變數

**這張表是唯一真相**,README 只放連結。CLI flag 優先於 env,env 優先於預設值(`argparse` 的 `default=os.environ.get("MNEME_X", <預設>)` 天然就是這個順序)。布林 flag 用 `action="store_true"`,env 端接受 `1/true/yes`(大小寫不敏感)。

| flag | env | 預設 | 說明 |
|---|---|---|---|
| `--data-dir` | `MNEME_DATA_DIR` | `./data` | 原圖 / 縮圖 / DB 的根。DB 裡存相對此處的路徑(spec.md §0) |
| `--db` | `MNEME_DB` | `<data-dir>/memory.db` | |
| `--camera` | `MNEME_CAMERA` | `/dev/video0` | |
| `--camera-cmd` | `MNEME_CAMERA_CMD` | 無 | 給定時**不開 `cv2.VideoCapture`**:跑這個外部命令寫 JPEG 到 `<data-dir>/incoming`,capture 改成 watch 該目錄並在讀完後刪檔。cv2 抓不到 camera 時的退路(spec.md §7) |
| `--no-camera` | `MNEME_NO_CAMERA` | `false` | 不開 capture,純 serve 現有資料。`health.mode` = `seed-only` |
| `--sidecar` | `MNEME_SIDECAR` | `/tmp/vlm.sock` | |
| `--mock-sidecar` | `MNEME_MOCK_SIDECAR` | `false` | 見 `sidecar.md` §8.5 |
| `--bind` | `MNEME_BIND` | `0.0.0.0:8080` | 拆成 uvicorn 的 host / port |
| `--static-dir` | `MNEME_STATIC_DIR` | `./web` | spec.md §2.7 |
| `--capture-fps` | `MNEME_CAPTURE_FPS` | `2` | |
| `--diff-threshold` | `MNEME_DIFF_THRESHOLD` | `12.0` | 64×64 灰階 mean absolute diff,0–255 尺度。現場光線不同務必調 |
| `--cooldown-ms` | `MNEME_COOLDOWN_MS` | `4000` | 觸發後的冷卻時間 |
| `--ask-min-score` | `MNEME_ASK_MIN_SCORE` | `0.35` | 低於此分視為「沒看到」,spec.md §2.4 |
| `--embed-dim` | `MNEME_EMBED_DIM` | `1024` | 必須跟 sidecar 的模型一致,啟動時對 `meta.embed_dim` 檢查 |
| `--sidecar-timeout-ms` | `MNEME_SIDECAR_TIMEOUT_MS` | `20000` | `Describe` / `Answer` 用;`Embed` 用 `5000` |

設定物件用一個 frozen `dataclass`,在 lifespan 起頭建好塞進 `app.state`。**不要在模組層讀 env**,那樣 seed 工具和測試都沒辦法換設定。

### 8.4 offline 偵測

- 背景 `asyncio.Task`,每 5 秒跑一次 `await asyncio.wait_for(asyncio.open_connection("1.1.1.1", 443), timeout=1.0)`,拿到 writer 就 `writer.close()`
- 成功 → `offline = False`,失敗(`TimeoutError` / `OSError`)→ `True`。結果寫進 `app.state` 的一個 bool,handler 只讀。單一 asyncio task 寫、GIL 保證 bool 賦值原子,**不需要 Lock**
- **絕不在 request handler 裡等網路**,也不要用 `socket.create_connection`(同步呼叫會卡住整個 event loop)。拔線時 health 一定要秒回,不然畫面會卡住
- 不用 ICMP ping(要 root),不用 DNS 查詢(系統快取會騙人)
- 拔線後最慢 6 秒畫面翻成 `true`。demo 前先演一次確認節奏,講稿要配合這個延遲

### 8.5 mock sidecar

搬到 [`docs/sidecar.md` §8.5](./sidecar.md#85-mock-sidecar)。實作在 `mneme/sidecar.py`,跟真 sidecar client 共用同一個 `typing.Protocol`;`health.sidecar` 一律誠實回 `mock`。

### 8.6 SQLite 併發與單一 process

- **uvicorn 固定 1 worker**。in-memory 檢索表(§8.7)、SSE broadcast、pipeline queue 全都在 process 記憶體裡,開多 worker 會變成幾份互不相通的狀態 —— 事件推播漏一半、`queue_depth` 亂跳。要靠多核就靠 sidecar 的 GPU,不是靠 web worker
- 寫入路徑只有 store 一條:一個專用的寫連線,包在 `asyncio.Lock` 裡序列化,所有寫都 `await asyncio.to_thread(...)`。不會有寫寫衝突
- 讀連線用 `threading.local()` 給每個 executor thread 一條,`sqlite3.connect(..., check_same_thread=False)` 只用在寫連線上。連線數自然收在 `to_thread` 的 executor 上限,不用另外接 pool 套件
- `busy_timeout` 設 5000ms(`PRAGMA busy_timeout = 5000`)
- 啟動時跑 `PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA foreign_keys=ON;`。**`foreign_keys` 是 per-connection**,每條新連線都要重下,不是開一次就算
- schema 用 `CREATE TABLE IF NOT EXISTS` 在啟動時建好,寫 `meta.schema_version = 1`。**48 小時不做 migration framework**,schema 改了就砍 db 重 seed
- 掃全庫的查詢包在 `asyncio.to_thread`,不要卡 event loop
- `sqlite3` 的 `isolation_level=None` + 自己下 `BEGIN`/`COMMIT`,不要靠 module 的隱式交易 —— 隱式模式下 `SELECT` 不會被交易包住,行為會讓人誤判

### 8.7 檢索實作

- 啟動時把所有 embeddings 一次讀進記憶體:`ids: list[str]` 加一個 `(buf: np.ndarray, count: int)` 的 tuple,`buf` shape `(capacity, dim)`、dtype `float32`
- store 寫入新 event 後直接寫進 `buf[count]` 再把 tuple 換成 `(buf, count + 1)`,不重讀 DB。容量滿了就 double 後複製。**讀者只讀 `buf[:count]`**,所以「先寫 row、後換 tuple」這個順序在 GIL 下天然安全,不需要 Lock
- cosine 就是一次 matmul:`scores = buf[:count] @ q`,再 `np.argpartition(-scores, min(top_k, count) - 1)[:top_k]` 取前 k、最後 `np.argsort` 把這 k 筆排好。**`kth` 一定要夾成 `min(top_k, count) - 1`** —— 直接寫 `argpartition(-scores, top_k)` 在 `count <= top_k` 時會丟 `ValueError: kth out of bounds`,而剛 seed 完或空庫正好是這個情況,`/api/ask` 會變 500。`count == 0` 直接走拒答路徑,不要進 numpy
- 3000 筆 × 1024 維 f32 = 12.3MB,實測一次檢索(matmul + 取前 5 + 排序)約 0.06ms,遠低於一次 VLM 呼叫。**不要引 sqlite-vec、不要引 faiss**
- 所有向量**入庫前就 L2 normalize**(`v / np.linalg.norm(v)`),cosine 退化成 dot product。normalize 前先擋 `norm == 0`,不然會產出一整排 `nan` 分數,拒答門檻就失效了
- `score` 用 `float(scores[i])` 轉回 Python float 再進 JSON。**不要把 `np.float32` 丟給 `json`**,它不是 JSON serializable,會在 `/api/ask` 炸 500
- 啟動時偵測到維度跟 `--embed-dim` 不符的向量:log error 並**拒絕啟動**,不要靜靜跳過 —— 混維度會讓分數變成垃圾而且很難查

### 8.8 驗收(交付前自己跑過,全部要過)

```bash
python -m venv --system-site-packages .venv && .venv/bin/pip install -e .
.venv/bin/python -c 'import cv2, numpy; print(cv2.__version__, numpy.__version__)'

.venv/bin/python -m mneme.seed --out data/memory.db --data-dir ./data --hours 8 --count 60 --seed 42
.venv/bin/python -m mneme --no-camera --mock-sidecar &

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

# 同一組 --seed 產出必須完全一致(換 --out 再跑一次,event id 與 summary 全同)
.venv/bin/python -m mneme.seed --out /tmp/again.db --data-dir /tmp/again --hours 8 --count 60 --seed 42
```

最後兩條是硬性驗收:拒答測試過不了不算做完(spec.md §2.4);seed 不可重現的話三個人會對不同資料 debug,`--seed` 就白寫了。
