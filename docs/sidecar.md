# 離線視覺記憶 — 推論 sidecar 契約 v1.3

VLM / LLM / embedding 三種推論的唯一真相來源。**Python sidecar 作者只需要看這一份**,不需要讀 Rust 端怎麼排程。

三份文件的節號連續、不重複:

| 文件 | 內容 |
|---|---|
| [`docs/spec.md`](./spec.md) | §0 全域約定、§2 對外 HTTP API、§5 分工、§6 時間表、§7 風險 |
| [`docs/backend.md`](./backend.md) | §1 SQLite schema、§3.3 pipeline 形狀、§4 seed、§8 後端實作契約(§8.5 除外) |
| 本文件 | §3.1 wire protocol、§3.2 prompt 契約、§8.5 mock sidecar |

**§3.1 的 enum 欄位名、§3.2 的 system prompt、§8.5 的 mock 行為都算契約**,改動要先在群組講一聲並更新版本號。

**v1.3 改了什麼**:從 `backend.md` v1.2 把 §3.1 / §3.2 / §8.5 原封不動搬到本文件。內容、節號、欄位名、prompt 文字都沒動,只把跨節引用改成跨檔引用。

---

## 邊界:sidecar 負責什麼

| 職責 | 誰做 |
|---|---|
| CUDA、模型載入、推論 | **sidecar(Python)** |
| 取樣、change filter、SQLite、檢索、HTTP、SSE | Rust,見 `backend.md` |
| 把 `Answer.context` 的時間轉成 `Asia/Taipei` 並排序號 | **Rust**,見 §3.2 |
| 拒答門檻(cosine `< 0.35`)判定 | **Rust**,sidecar 不做;見 `spec.md` §2.4 |
| 向量 L2 normalize | Rust 入庫前做(`backend.md` §8.7);sidecar 回原始向量即可 |

三個模型與版本以 `/api/health` 公告的為準(`spec.md` §2.1 範例:VLM `SmolVLM2-2.2B-Instruct`、LLM `qwen2.5-7b-instruct-q4`、embed `bge-m3`、`embed_dim` 1024)。**`Embedded.vec` 的長度必須等於 `--embed-dim`**(預設 1024,真相表在 `backend.md` §8.3),不一致 Rust 端會拒絕啟動。

程式放 `sidecar/server.py`,system prompt 放 `sidecar/prompts.py`,依賴寫 `sidecar/requirements.txt`(完整 repo 佈局見 `backend.md` §8.1)。

逾時由 Rust 端切:`Describe` / `Answer` 用 `--sidecar-timeout-ms`(預設 20000),`Embed` 固定 `5000`。超過 Rust 直接回 `spec.md` §2.6 的 `SIDECAR_TIMEOUT`,所以慢就是失敗,寧可回 `Failed` 也不要吊著。

---

## 3.1 Wire protocol

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

`serde(tag = "kind", rename_all = "snake_case")` 的意思是 wire 上長這樣,Python 端照這個 key 名寫:

```json
{"kind":"describe","req_id":"...","image_path":"frames/frm_01JBQ3M8ZK.jpg"}
{"kind":"described","req_id":"...","summary":"一個人把黑色後背包放在桌子左側,然後離開","objects":["person","backpack","desk"],"ms":812}
```

`image_path` 是相對 `--data-dir` 的路徑(`spec.md` §0「路徑」那條),sidecar 自己跟 `--data-dir` 拼。`ms` 是 sidecar 自己量的推論耗時,純粹給 log 和 demo 用。

連線與併發約定:

- Rust 只開**一條** socket 連線,而且**同時只有一個 in-flight 請求**。`req_id` 純粹用來對帳和 log 追蹤,不是為了多工 —— sidecar 端不需要做併發
- 每行一個 JSON,`\n` 結尾。字串裡不得有裸換行(`serde_json` 預設就會轉義,別自己拼字串)
- 收到不認識的 `kind` 回 `Failed { code: "UNKNOWN_KIND" }`,不要直接崩
- 斷線後 Rust 每 2 秒重連。期間 `/api/health.sidecar` = `down`、`status` = `degraded`,`/api/ask` 回 `SIDECAR_UNAVAILABLE`,但 `/api/events` 照常能讀

`Failed` 回來時 Rust 對外翻成 502 `SIDECAR_FAILED`(`spec.md` §2.6);`code` 字串自己取,會進 log,`message` 給人看。

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

`context` 最多 3 筆,也可能是空陣列(Rust 端判定過門檻才呼叫,但空的情況要能處理:照 system prompt 回固定拒答句)。`Describe` 產出的 `summary` 同樣是**一句中文**,直接進 DB 和 `/api/events`,不會再被加工。

## 8.5 mock sidecar

`--mock-sidecar` 時不開 socket,in-process 實作跟真 sidecar 同一個 trait:

- `Describe` → 從固定句庫挑一句(拿圖片灰階 mean 當索引),`objects` 用那句對應的固定清單,`ms` 回 `300`
- `Embed` → **deterministic**:用 `DefaultHasher` 吃 text 當 PRNG seed,產 `--embed-dim` 個 f32 再 L2 normalize。同一句話永遠同一個向量。語意相似度當然不像真模型,但**檢索、排序、拒答門檻、citations 組裝全都能測**
- `Answer` → 把 context 第一筆改寫成「最後一次是 <時間>,<summary>」;context 空的時候回 §3.2 的固定拒答句
- `health.sidecar` = `mock`

mock 的存在意義是讓 API 層在沒有 Orin 的機器上完整開發與測試。**它不是 demo 路徑**,`mode` / `sidecar` 兩個欄位一定要誠實反映,不准為了畫面好看寫成 `up`。

mock 是 Rust 端的 in-process 實作(`src/sidecar.rs`),Python sidecar 作者不需要實作它 —— 但**行為要對得上**:真 sidecar 換上去之後,`backend.md` §8.8 的驗收清單必須照樣全過,包含最後那條硬性拒答測試。
