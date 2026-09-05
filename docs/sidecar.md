# 離線視覺記憶 — 推論 sidecar 契約 v1.6

VLM / LLM / embedding 三種推論的唯一真相來源。**sidecar 作者只需要看這一份**,不需要讀後端怎麼排程。

三份文件的節號連續、不重複:

| 文件 | 內容 |
|---|---|
| [`docs/spec.md`](./spec.md) | §0 全域約定、§2 對外 HTTP API、§5 分工、§6 時間表、§7 風險 |
| [`docs/backend.md`](./backend.md) | §1 SQLite schema、§3.3 pipeline 形狀、§4 seed、§8 後端實作契約(§8.5 除外) |
| 本文件 | §3.1 wire protocol、§3.2 prompt 契約、§8.5 mock sidecar、§8.9 macOS 驗證 |

**§3.1 的 `kind` 值與欄位名、§3.2 的 system prompt、§8.5 的 mock 行為都算契約**,改動要先在群組講一聲並更新版本號。

**v1.6 改了什麼**:§8.9 第一則從「實測發現」升級成**已決議** —— 語意拒答的責任正式歸給 §3.2 的 system prompt,`--ask-min-score` 降級成下限護欄(對應 `spec.md` v1.5 §2.4、`backend.md` v1.5 §8.3)。**§3.1 wire、§3.2 prompt 文字、§8.5 mock 行為一個字都沒動。**

**v1.5 改了什麼**:sidecar 從契約變成實作 —— `sidecar/server.py`(`--backend cuda` 給 Orin、`--backend mlx` 給 macOS 驗證)、`sidecar/prompts.py`、`scripts/verify_sidecar.py`(43 條檢查,含真攝影機畫面)。**§3.1 的 wire 與 §3.2 的 prompt 文字一個字都沒動**,新增的只有 §8.9。§8.9 記了三件實測結論,其中「`--ask-min-score 0.35` 對真 bge-m3 不會觸發」會影響現場調參、「鏡頭別對著螢幕」會影響 demo 佈置,務必看一眼。

**v1.4 改了什麼**:後端從 Rust 換成 Python(`backend.md` v1.4),所以本文件把「Rust 端」一律改寫成「後端」,§3.1 的型別宣告從 serde enum 改寫成等價的 Python 定義。**wire 上的 JSON 一個位元都沒變**:`kind` 值、欄位名、型別、`Failed` 語意、逾時、重連間隔全部同 v1.3。sidecar 端不用改任何程式。

**v1.3 改了什麼**:從 `backend.md` v1.2 把 §3.1 / §3.2 / §8.5 原封不動搬到本文件。內容、節號、欄位名、prompt 文字都沒動,只把跨節引用改成跨檔引用。

---

## 邊界:sidecar 負責什麼

| 職責 | 誰做 |
|---|---|
| CUDA、模型載入、推論 | **sidecar(Python)** |
| 取樣、change filter、SQLite、檢索、HTTP、SSE | 後端主程式,見 `backend.md` |
| 把 `Answer.context` 的時間轉成 `Asia/Taipei` 並排序號 | **後端**,見 §3.2 |
| 語意拒答判定 | **§3.2 的 system prompt**(sidecar 端的 LLM)。後端的 `--ask-min-score` 只是下限護欄,見 `spec.md` §2.4 與 §8.9 第一則 |
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

## 8.9 在 macOS 上驗證真 sidecar

§8.5 最後一句是「真 sidecar 換上去之後 §8.8 必須照樣全過」——但 mock 不開 socket,所以 `--mock-sidecar` 對 §3.1 的 wire protocol **一個字都沒驗到**。以前只有 Orin 能驗,現在 Apple silicon 也可以:`sidecar/server.py --backend mlx` 用 MLX 載同樣那三個模型,協議與 prompt 完全共用同一份實作。

```bash
python3 -m venv sidecar/.venv
sidecar/.venv/bin/pip install -r sidecar/requirements-mlx.txt

# 需要 seed 過的資料(§8.8 第一段)
.venv/bin/python -m mneme.seed --out data/memory.db --data-dir ./data --hours 8 --count 60 --seed 42
.venv/bin/python scripts/verify_sidecar.py
```

腳本自己起 sidecar、跑完 **43 條**檢查、全過才 exit 0。涵蓋:§3.1 的 framing / `req_id` 回echo / `vec` 長度與有限性 / `UNKNOWN_KIND` 不崩且連線存活、§3.2 的空 context 拒答句與 grounded 回答、§8.8 的檢索排序與硬性拒答、**攝影機真畫面的 `describe`**,外加 `mneme.sidecar.SocketSidecar` **未修改**就能對接。硬性拒答那條**無條件走 §3.2 的 prompt 路徑**(不是看下限護欄有沒有剛好觸發),所以條數固定,不隨門檻值浮動。

### 攝影機那段在驗什麼

seed 圖是純色底加上 `SEED #059` 字樣,所以**一個完全無視像素、每次都吐同一句話的 VLM 也能通過其他所有檢查**。攝影機那段擋掉這種假通過:抓兩張中間隔 2 秒的真畫面(腳本會提示你動一下),要求兩句 summary **必須不同**,並檢查是中文、`objects` 是短標籤、summary 裡沒有 `SEED`。

```bash
.venv/bin/python scripts/verify_sidecar.py                # 預設就會用攝影機
.venv/bin/python scripts/verify_sidecar.py --no-camera     # 只跑 33 條協議檢查
.venv/bin/python scripts/verify_sidecar.py --keep-frames    # 留下畫面自己看
```

抓完的畫面會刪掉:那兩張是驗證用的暫存,沒有對應的 `events` 列,留在 `data/frames/` 只會變成 `--no-camera` 之後還被 HTTP serve 出去的垃圾。

**沒有攝影機或被 OS 拒絕存取時是 SKIP,不是 FAIL**,並且那 10 條檢查**不會被算成通過**(`--no-camera` 就是剛好 33 條)—— 機器沒鏡頭是環境問題,不該偽裝成契約壞掉,也不該偽裝成契約沒問題。macOS 第一次開鏡頭會跳權限視窗,拒絕後 `isOpened()` 回 `False` 並在 stderr 印 `not authorized to capture video`;`--camera-cmd`(`spec.md` §7)仍然是文件裡的退路。

兩個 venv 是刻意的(`backend.md` §8.2):腳本跑在主程式 venv,sidecar 跑在自己的 venv,torch / MLX 不可能漏進後端環境。

**`--backend mlx` 只是驗證管道,不是 demo 路徑**,demo 一律 Orin 上的 `--backend cuda`。兩個 backend 共用 `Server` 與 §3.2 的 prompt,所以會漂掉的只有模型權重,不會是協議。

### 驗證量到的三件事

**一、`--ask-min-score 0.35` 對真 bge-m3 不會觸發,所以拒答責任已改判給 §3.2。** 那個預設值是照 mock 的 bag-of-bigrams 調的(無關句子 cosine 落在 0 附近)。真 bge-m3 的中文 cosine 有地板:實測 grounded 問句 `馬克杯放在哪` 拿 0.813,而**沒發生過**的 `有沒有人在跳舞` 拿 0.716 —— 兩者都遠高於 0.35,分離度只有 0.097。

**這不是待辦,是已經下的決定:不去調那個數字。** 0.097 的分離度撐不起單一門檻 —— 設 0.75 會把 0.716 擋掉但也開始誤拒 grounded 問句,而且門檻值會隨資料集、語言、鏡頭場景漂移,現場沒有時間重新量。所以 `spec.md` v1.5 §2.4 把語意拒答正式歸給 §3.2 的 system prompt,`--ask-min-score` 保留 `0.35` 當**下限護欄**:擋空庫、壞向量、以及 mock 那種無關句 cosine 近 0 的情形,真模型上幾乎不觸發。

實測 Qwen2.5-7B-4bit 對「沒發生過」的問題回了逐字相符的拒答句,而 `citations` 照實附上檢索到的事件 —— 這是常態路徑,不是退化。**驗收條件因此看 `answer` 而不看 `citations` 是否為 `[]`**(`backend.md` §8.8)。真要改門檻必須拿真資料重新量過,照 mock 的直覺設值只會讓它變裝飾。

**二、sidecar 的 venv 需要 torch,但只用來做前處理。** transformers 5.x 把 SmolVLM 的 image processor 綁在 torch 後面(連 PIL 版都會解析成 dummy),沒有它 `mlx_vlm.load` 直接失敗。推論全程在 MLX/Metal,torch 只負責 image preprocessing —— 這也正是 sidecar 要有自己 venv 的原因。

**三、SmolVLM2-2.2B 的中文句子品質勉強堪用,英文標籤可靠。** 真畫面實測:`objects` 準確(`person` / `curtain` / `wall` / `window`,對得上現場),summary 抓得到主體與動作(「一個男人在一個房間中拉着窗簾,窗簾是紅色的」),但**句子偶爾語法破碎或自我重複**,也會混簡體字。它同時會 OCR 畫面上的文字 —— 拿螢幕截圖測時,`objects` 直接吐出畫面裡的程式碼識別字。

這對 demo 有兩個意涵:**一是鏡頭別對著螢幕**,不然 summary 會變成 OCR 結果;**二是 summary 直接進 `/api/events` 不再加工(§3.2),所以句子品質就是使用者看到的品質**。要更好的中文得換更大的 VLM,那是模型選擇問題,不是協議問題 —— `--vlm` 可以直接換,`Server` 那層不用動。

實測(M5 Pro / 64GB):三個模型載入約 5s,`describe` 約 4.4s、`embed` 約 16ms、`answer` 約 0.3–1.1s,常駐約 9GB。
