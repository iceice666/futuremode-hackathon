# 離線視覺記憶 — 介面契約 v1.5

三人平行開工用。**對外介面的唯一真相來源是這一份**,改動要先在群組講一聲並更新版本號。

目標:改任何一層都不需要另外兩層在場。Bryan 和 coffeecat 從第一小時就接假資料開工,iceice666 那邊通了才換真的。

## 文件分工

| 文件 | 內容 | 誰要看 |
|---|---|---|
| 本文件 | §0 全域約定、§2 對外 HTTP API、§5 分工邊界、§6 時間表、§7 風險與退路 | 全員。Bryan / coffeecat 只需要 §2 |
| [`docs/backend.md`](./backend.md) | §1 SQLite schema、§3.3 pipeline 形狀、§4 假資料 seed、§8 後端實作契約(§8.5 除外) | iceice666 / 後端 agent |
| [`docs/sidecar.md`](./sidecar.md) | §3.1 wire protocol、§3.2 prompt 契約、§8.5 mock sidecar | 寫 VLM / LLM / embedding 推論的人 |

節號在三份文件之間連續、不重複,所以既有的 `§2.6` `§8.3` 這類引用不會歧義。跨檔引用一律寫成 `backend.md §8.3` / `sidecar.md §3.2`。

**v1.5 改了什麼**(v1.4 → v1.5):**§2.4 的拒答規則改寫** —— 實測真 bge-m3 的中文 cosine 有地板,單一分數門檻分不開「有發生」與「沒發生」(`sidecar.md` §8.9 第一則),所以語意拒答的責任正式歸給 `sidecar.md` §3.2 的 system prompt,`--ask-min-score` 降級成下限護欄。**JSON shape 一個欄位都沒動**,變的是可達組合:`answer` 是拒答句時 `citations` 可能非空。§2.6 錯誤 code、分頁語意、其他 endpoint 零變更

**v1.4 改了什麼**(v1.3 → v1.4):後端實作語言從 Rust 換成 Python(asyncio + FastAPI),細節全在 `backend.md` v1.4。**這份文件的對外契約零變更** —— §0、§2 的 JSON、欄位、分頁語意、錯誤 code、拒答門檻全部原樣;只把 §2.7 / §3 / §5 / §7 / §8 裡「Rust」這個字換成「後端」,並把 §7 的 `nokhwa` 那條退路改成 cv2 的版本

**v1.3 改了什麼**(v1.2 → v1.3):把 §3.1 / §3.2 / §8.5 從 `backend.md` 再拆到 `docs/sidecar.md`。對外行為沒動。

**v1.2 改了什麼**(v1.1 → v1.2):把 §1 / §3 / §4 / §8 搬到 `docs/backend.md`,內容與節號原樣保留。對外 JSON、行為、預設值全都沒動。

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
| 時區責任 | 唯一例外是 `/api/ask` 的 `answer`:那是給人看的中文句子,由後端組 prompt 時就轉成 `Asia/Taipei`(見 sidecar.md §3.2)。其他所有欄位、所有 API,一律 UTC |
| 向量維度 | 全庫單一維度,由 `/api/health.embed_dim` 公告。混維度視為 bug,不做相容 |

---

## 1. SQLite Schema

搬到 [`docs/backend.md` §1](./backend.md#1-sqlite-schema)。前端與 bot 不直接讀 DB,只看 §2 的 JSON。

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
- `offline` — 實際偵測外網可達性才回 true,不准寫死。做法見 backend.md §8.4。拔網路線那一刻畫面上要看得到它變 true
- `capture_fps` — 最近 10 秒滑動窗的實測值,不是設定值。`--no-camera` 時回 `0.0`
- `queue_depth` — `change_filter → vlm_sidecar` 那條有界 channel 當下的長度(容量 4,見 backend.md §3.3)。全 pipeline 只有這裡會塞
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

- `question` 送 sidecar `Embed` 拿向量,對全庫 embeddings 算 cosine,取前 `top_k`(實作見 backend.md §8.7)
- `score` 就是**原始 cosine**,範圍 `[-1, 1]`,不正規化、不 rescale、不轉百分比。前端要顯示百分比自己換
- **拒答有兩條路,prompt 那條才是主要的。** 真 bge-m3 的中文 cosine 有地板 —— 實測 grounded 問句 `馬克杯放在哪` 拿 0.813,**沒發生過**的 `有沒有人在跳舞` 拿 0.716,分離度只有 0.097(`sidecar.md` §8.9)。**單一 cosine 門檻分不開這兩者**,所以語意拒答由 `sidecar.md` §3.2 的 system prompt 負責:LLM 判斷 context 不足就回固定句 `"我沒有看到相關的畫面。"`,而 `citations` **仍照實附上檢索到的事件** —— 讓評審看得到我們檢索到什麼,只是不硬掰
- **`--ask-min-score` 是下限護欄,不是語意判定。** 最高分 `< 門檻`(預設 `0.35`)才走這條:不呼叫 LLM,`answer` 固定回同一句拒答句,`citations` 給 `[]`。它擋的是空庫、壞向量、以及 mock sidecar 那種無關句 cosine 落在 0 附近的情形。**真模型上幾乎不會觸發,不要靠它拒答**;要改門檻必須拿真資料量過(`sidecar.md` §8.9),照 mock 的直覺設值只會讓它變裝飾
- 過門檻:只把 `score >= 門檻` 的送進 prompt,最多 3 筆。`top_k` 只影響檢索寬度,**不影響回傳筆數**
- `citations` 依 `score` 由高到低,最多 3 筆
- 所以 `answer` 是拒答句時,`citations` **可能非空**(prompt 路徑,常態)也**可能是 `[]`**(護欄路徑)。前端兩種都要能顯示,不要假設拒答就沒有引用
- 每次呼叫都寫一筆 `queries`,成功失敗都寫,`latency_ms` 一起存
- `latency_ms` = 收到 request 到組完 response,含 embed + LLM
- sidecar 沒連上回 503 `SIDECAR_UNAVAILABLE`;超時回 504 `SIDECAR_TIMEOUT`

**不准硬掰**,評審一定會問一個沒發生過的事情。backend.md §8.8 的最後一條 curl 就是在測這個:驗收條件是 `answer` 明講沒看到,**不是 `citations` 為 `[]`** —— 真模型上拒答會帶著引用回來。過不了不算做完。

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

- 後端把 `--static-dir`(預設 `./web`)掛在 `/`,`/api/*` 的路由優先
- 路徑不以 `/api` 開頭又找不到檔案 → 回 `index.html`(SPA fallback),讓 Bryan 用前端 router 沒問題
- Bryan 開發時跑自己的 dev server:後端一律回 `Access-Control-Allow-Origin: *`。反正不做 auth,沒有風險

---

## 3. 主程式 ↔ 推論 sidecar 契約

wire protocol 與 prompt 契約搬到 [`docs/sidecar.md` §3.1 / §3.2](./sidecar.md#31-wire-protocol);pipeline 形狀(§3.3)在 [`docs/backend.md`](./backend.md#33-pipeline-形狀)。純內部協議,對 §2 的外部行為沒有影響。

---

## 4. 假資料 seed(第一小時就要有)

搬到 [`docs/backend.md` §4](./backend.md#4-假資料-seed第一小時就要有)。要記的只有一句:seed 出來的事件 `source` 一律是 `seed`,而且第一小時就會有,另外兩人整場都能離線開發。

---

## 5. 分工邊界

| 人 | 負責 | 交付的東西 |
|---|---|---|
| iceice666 | capture / filter / sidecar / store / 所有 HTTP endpoint | 一個跑得起來的服務(`python -m mneme`)+ seed 工具 |
| Bryan | timeline web UI、health 面板 | 靜態檔案放 `web/`,由後端直接 serve,規則見 §2.7 |
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
| `cv2.VideoCapture` 在 Jetson 上抓不到 camera | 用 `--camera-cmd`(backend.md §8.3)跑 `ffmpeg -f v4l2 … -f image2` 寫檔到 `<data-dir>/incoming`,capture 改成 watch 那個目錄 |

每一條退路都要在 T+30h 前實際測過一次,不能只寫在紙上。

---

## 8. 後端實作契約

搬到 [`docs/backend.md` §8](./backend.md#8-後端實作契約iceice666--後端-agent-照這節做):專案佈局、依賴選型、CLI/env 表、offline 偵測、mock sidecar、SQLite 併發、檢索實作、驗收清單。

CLI flag 與 `MNEME_*` 環境變數的唯一真相是 [`docs/backend.md` §8.3](./backend.md#83-cli-與環境變數)。
