# 離線視覺記憶 — 推論 sidecar 契約 v1.4

VLM / LLM / embedding 三種推論的唯一真相來源。**sidecar 作者只需要看這一份**,不需要讀後端怎麼排程。

三份文件的節號連續、不重複:

| 文件 | 內容 |
|---|---|
| [`docs/spec.md`](./spec.md) | §0 全域約定、§2 對外 HTTP API、§5 分工、§6 時間表、§7 風險 |
| [`docs/backend.md`](./backend.md) | §1 SQLite schema、§3.3 pipeline 形狀、§4 seed、§8 後端實作契約(§8.5 除外) |
| 本文件 | §3.1 wire protocol、§3.2 prompt 契約、§8.5 mock sidecar |

**§3.1 的 `kind` 值與欄位名、§3.2 的 system prompt、§8.5 的 mock 行為都算契約**,改動要先在群組講一聲並更新版本號。

**v1.4 改了什麼**:後端從 Rust 換成 Python(`backend.md` v1.4),所以本文件把「Rust 端」一律改寫成「後端」,§3.1 的型別宣告從 serde enum 改寫成等價的 Python 定義。**wire 上的 JSON 一個位元都沒變**:`kind` 值、欄位名、型別、`Failed` 語意、逾時、重連間隔全部同 v1.3。sidecar 端不用改任何程式。

**v1.3 改了什麼**:從 `backend.md` v1.2 把 §3.1 / §3.2 / §8.5 原封不動搬到本文件。內容、節號、欄位名、prompt 文字都沒動,只把跨節引用改成跨檔引用。

---

## 邊界:sidecar 負責什麼

| 職責 | 誰做 |
|---|---|
| CUDA、模型載入、推論 | **sidecar(Python)** |
| 取樣、change filter、SQLite、檢索、HTTP、SSE | 後端主程式,見 `backend.md` |
| 把 `Answer.context` 的時間轉成 `Asia/Taipei` 並排序號 | **後端**,見 §3.2 |
| 拒答門檻(cosine `< 0.35`)判定 | **後端**,sidecar 不做;見 `spec.md` §2.4 |
| 向量 L2 normalize | 後端入庫前做(`backend.md` §8.7);sidecar 回原始向量即可 |

三個模型與版本以 `/api/health` 公告的為準(`spec.md` §2.1 範例:VLM `SmolVLM2-2.2B-Instruct`、LLM `qwen2.5-7b-instruct-q4`、embed `bge-m3`、`embed_dim` 1024)。**`Embedded.vec` 的長度必須等於 `--embed-dim`**(預設 1024,真相表在 `backend.md` §8.3),不一致後端會拒絕啟動。

程式放 `sidecar/server.py`,system prompt 放 `sidecar/prompts.py`,依賴寫 `sidecar/requirements.txt`(完整 repo 佈局見 `backend.md` §8.1)。

**sidecar 有自己的 venv**,主程式那個 venv 不裝 torch(`backend.md` §8.2)。後端跟 sidecar 都是 Python,但仍然是兩個 process:模型載入時間、GPU 記憶體、崩潰範圍都要跟 HTTP server 隔開。所以 sidecar 不會被 import,只被連線。

逾時由後端切:`Describe` / `Answer` 用 `--sidecar-timeout-ms`(預設 20000),`Embed` 固定 `5000`。超過後端直接回 `spec.md` §2.6 的 `SIDECAR_TIMEOUT`,所以慢就是失敗,寧可回 `Failed` 也不要吊著。

---

## 3.1 Wire protocol

推論留在專屬 process,主程式不碰 CUDA。Unix socket `/tmp/vlm.sock`,line-delimited JSON,一行一個請求、一行一個回應。

```python
# 請求(後端 → sidecar):tag 欄位是 "kind",值為 snake_case
class Describe(TypedDict):   # kind: "describe"
    kind: Literal["describe"]; req_id: str; image_path: str
class Embed(TypedDict):      # kind: "embed"
    kind: Literal["embed"];    req_id: str; text: str
class Answer(TypedDict):     # kind: "answer"
    kind: Literal["answer"];   req_id: str; question: str; context: list[str]

# 回應(sidecar → 後端)
class Described(TypedDict):  # kind: "described"
    kind: Literal["described"]; req_id: str; summary: str; objects: list[str]; ms: int
class Embedded(TypedDict):   # kind: "embedded"
    kind: Literal["embedded"];  req_id: str; vec: list[float]; ms: int
class Answered(TypedDict):   # kind: "answered"
    kind: Literal["answered"];  req_id: str; answer: str; ms: int
class Failed(TypedDict):     # kind: "failed"
    kind: Literal["failed"];    req_id: str; code: str; message: str
```

型別對應:`ms` 是非負整數(毫秒),`vec` 是 JSON number 陣列、長度等於 `--embed-dim`、精度視為 f32(後端讀進來就轉 `float32`,多餘位數會被截掉)。wire 上長這樣,兩端照這個 key 名寫:

```json
{"kind":"describe","req_id":"...","image_path":"frames/frm_01JBQ3M8ZK.jpg"}
{"kind":"described","req_id":"...","summary":"一個人把黑色後背包放在桌子左側,然後離開","objects":["person","backpack","desk"],"ms":812}
```

`image_path` 是相對 `--data-dir` 的路徑(`spec.md` §0「路徑」那條),sidecar 自己跟 `--data-dir` 拼。`ms` 是 sidecar 自己量的推論耗時,純粹給 log 和 demo 用。

連線與併發約定:

- 後端只開**一條** socket 連線,而且**同時只有一個 in-flight 請求**。`req_id` 純粹用來對帳和 log 追蹤,不是為了多工 —— sidecar 端不需要做併發
- 一個 `/api/ask` 會**連續佔用** sidecar 跑完 `embed` + `answer` 才放手,capture pipeline 的 `describe` 插不進中間。**協議沒變** —— 還是一條連線、同時一個 in-flight 請求,sidecar 端看不出差別。理由是量出來的:兩個 RPC 各搶一次鎖時,pipeline 會在縫隙塞一個約 800ms 的 `describe`,實測讓 ask 的 `embed` 等了 +708ms;改成一次佔用後,滿載 pipeline 下的 ask 從 2.36s 降到 1.64s(純推論地板是 1.56s),而 pipeline 吞吐不變(1.15 events/s)。**互動優先權沒用、preempt 更糟**:pipeline 同時只有一個 RPC 在飛,ask 等的是「正在跑」那個,GPU 跑到一半沒辦法倒帶 —— 實測插隊只買到 1ms,砍掉進行中的 `describe` 反而讓 ask 慢到 2.37s 又白丟 7 張畫面
- 每行一個 JSON,`\n` 結尾。字串裡不得有裸換行:一律 `json.dumps(obj, ensure_ascii=False) + "\n"`,別自己拼字串。**`json.dumps` 預設就會轉義控制字元**,中文靠 `ensure_ascii=False` 保持可讀(要用 ASCII escape 也合法,兩邊都解得開)
- 讀取用 `await reader.readline()`,不要 `read(n)` 自己切。`Embedded.vec` 1024 維實測一行約 22KB(實測值,見下方 json 範例的同款 dumps),還在 asyncio 預設 64KB limit 內,但換更大的 embedding 模型就會撞到 —— 那時要調 `open_unix_connection(limit=...)`,不是改協議
- 收到不認識的 `kind` 回 `{"kind":"failed","code":"UNKNOWN_KIND",...}`,不要直接崩
- 斷線後後端每 2 秒重連。期間 `/api/health.sidecar` = `down`、`status` = `degraded`,`/api/ask` 回 `SIDECAR_UNAVAILABLE`,但 `/api/events` 照常能讀

`Failed` 回來時後端對外翻成 502 `SIDECAR_FAILED`(`spec.md` §2.6);`code` 字串自己取,會進 log,`message` 給人看。

後端**會驗回覆**,不合約定的一律翻成 502 `SIDECAR_FAILED`,不會變成 500 —— `spec.md` §2.6 的錯誤分類沒有「後端自己爛掉」這一格,而且每個 `/api/ask` 都還是欠一列 `queries`(§2.4)。會被擋掉的情況與對應 `code`:

| 情況 | `code` |
|---|---|
| 一行超過 1MiB(`READ_LIMIT`)還沒看到 `\n` | `OVERSIZED_REPLY`,**連線會被丟掉重連**(framing 已經壞了,後面每次讀都會從半行開始) |
| 合法 JSON 但不是 object(array / number / `null` / string) | `BAD_REPLY` |
| `kind` 對了但缺 payload 欄位(`summary` / `vec` / `answer`) | `BAD_REPLY` |
| `vec` 不是數字陣列、長短不齊、含 `null` / `NaN` / `inf` | `BAD_REPLY` |
| `vec` 長度不等於 `--embed-dim` | `BAD_REPLY` |
| `objects` 不是 list | `BAD_REPLY` |
| `kind` 不是預期的那個 | `UNEXPECTED_KIND` |
| 整行不是合法 JSON | `BAD_JSON`,連線丟掉重連 |
| `req_id` 跟請求不符 | `REQ_ID_MISMATCH` |

`NaN` / `inf` 擋在這裡是必要的:一個 `NaN` 進了檢索表就會讓每個 cosine 分數變 `NaN`,拒答門檻(`--ask-min-score`)靜靜失效,`/api/ask` 從此不再拒答。**`json.dumps` 會把 `float('nan')` 寫成裸 `NaN`,那不是合法 JSON**,Python 這端解得開、別的 client 不一定 —— sidecar 端出手前自己先擋掉。

## 3.2 Prompt 契約

`Answer.context` 的每個元素是**後端組好的一行字串**,格式固定:

```
[1] 2026-09-05 22:03(台北時間) 一個人把黑色後背包放在桌子左側,然後離開
```

- 序號 `[1]`–`[3]` 由後端配,順序跟 `citations` 一致
- **時間在這裡就轉好 `Asia/Taipei` 並寫成中文可讀格式**。LLM 不做時區換算,也不需要知道 UTC 存在 —— 這是 `spec.md` §0「時區責任」那條的唯一例外
- system prompt 固定寫在 `sidecar/prompts.py`,**改它算改契約**,要更新版本號:

```
你是一個離線影像記憶助理。只能根據提供的觀察記錄回答,用一到兩句中文。
記錄裡沒有的事情,直接回「我沒有看到相關的畫面。」——不要推測、不要補完、不要說「可能」。
提到時間就用記錄裡給的時間,不要自己算。
```

`Answer` 回來的 `answer` 後端不做二次加工,原樣放進 response。

`context` 最多 3 筆,也可能是空陣列(後端判定過門檻才呼叫,但空的情況要能處理:照 system prompt 回固定拒答句)。`Describe` 產出的 `summary` 同樣是**一句中文**,直接進 DB 和 `/api/events`,不會再被加工。

## 8.5 mock sidecar

`--mock-sidecar` 時不開 socket,in-process 實作跟真 sidecar client 同一個 `typing.Protocol`:

- `Describe` → 從固定句庫挑一句(拿圖片灰階 mean 當索引),`objects` 用那句對應的固定清單,`ms` 回 `300`
- `Embed` → **deterministic**:`seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")`,再 `np.random.default_rng(seed).standard_normal(embed_dim, dtype=np.float32)` 後 L2 normalize。同一句話永遠同一個向量,而且**跨機器、跨 Python 版本都一樣**(所以不准用內建 `hash()`:它有 per-process seed)。語意相似度當然不像真模型,但**檢索、排序、拒答門檻、citations 組裝全都能測**
- `Answer` → 把 context 第一筆改寫成「最後一次是 <時間>,<summary>」;context 空的時候回 §3.2 的固定拒答句
- `health.sidecar` = `mock`

mock 的存在意義是讓 API 層在沒有 Orin 的機器上完整開發與測試。**它不是 demo 路徑**,`mode` / `sidecar` 兩個欄位一定要誠實反映,不准為了畫面好看寫成 `up`。

mock 是後端的 in-process 實作(`mneme/sidecar.py`),真 sidecar 作者不需要實作它 —— 但**行為要對得上**:真 sidecar 換上去之後,`backend.md` §8.8 的驗收清單必須照樣全過,包含最後那條硬性拒答測試。
