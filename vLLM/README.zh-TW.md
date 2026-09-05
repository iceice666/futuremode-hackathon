# Orin 上的 vLLM — Qwen2.5-VL-3B

在 Jetson Orin(JetPack 6.2 / L4T 36.4.7 / CUDA 12.6,15GB 統一記憶體)上,用 vLLM 的
OpenAI 相容 API server 跑 Qwen2.5-VL-3B。透過
[dusty-nv/jetson-containers](https://github.com/dusty-nv/jetson-containers)(已 clone 到
`jetson-containers/`)取得針對 Jetson 修補過的 CUDA / PyTorch / vLLM。

**現況:可正常運作。** 實測單張圖片中文描述約 **4.3 秒**,記憶體用量約 10.5GB / 15.6GB。

為什麼用 Docker 而不是直接 pip install:vLLM 官方沒有為 Jetson 的 iGPU(sm_87)發布 wheel。

---

## 快速開始

環境都設定好之後,日常只需要兩個指令:

```bash
cd /home/jetson/futuremode/vLLM
./serve_qwen2_5_vl.sh                                    # 啟動(背景執行)
python3 test_client.py <圖片路徑> "用一句繁體中文描述這張圖片。"   # 測試
```

其他常用操作:

```bash
docker logs -f mneme-vllm     # 看即時 log
docker stop mneme-vllm        # 停止
curl http://127.0.0.1:8000/v1/models   # 確認 server 活著
```

Server 監聽 `0.0.0.0:8000`,同網段其他機器可以直接連。

---

## 首次安裝(一次性)

### 1. 安裝 Docker 並註冊 nvidia runtime

```bash
sudo apt update && sudo apt install -y docker.io
sudo usermod -aG docker $USER
sudo systemctl enable --now docker

# 這步很關鍵,少了它 docker run --runtime nvidia 會失敗
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

裝完後**登出再登入**(或執行 `newgrp docker`),否則現在這個 terminal 拿不到 `docker`
群組權限,會出現 `permission denied ... /var/run/docker.sock`。

`nvidia-container-toolkit` 本身 JetPack 已內建,不用另外裝,但**一定要**用 `nvidia-ctk`
註冊進 Docker,否則會看到 `unknown or invalid runtime name: nvidia`。

### 2. 下載模型權重

```bash
./download_model.sh
```

約 3.5GB,存進 `jetson-containers/data/models/huggingface`(容器內掛載為 `/data`)。
下載和啟動分開做是刻意的 —— 見下方「網路不穩」。

### 3. 啟動

```bash
./serve_qwen2_5_vl.sh
```

首次執行會先拉 5.7GB 的容器 image(`dustynv/vllm:0.8.6-r36.4-cu128-24.04`),之後就直接啟動。

---

## Jetson 特有的坑(照順序踩過的)

這台機器跟一般 x86 + 獨立顯卡的伺服器差很多,以下都是實際遇到並解掉的問題。

### `unknown or invalid runtime name: nvidia`

`nvidia-container-toolkit` 有裝不等於 Docker 認得它。要跑
`sudo nvidia-ctk runtime configure --runtime=docker` 再重啟 Docker。

### `NVML_SUCCESS == r INTERNAL ASSERT FAILED` + `NvMapMemAllocInternalTagged: error 12`

看起來像 NVML 的 bug,**實際上是記憶體不足**(`error 12` = ENOMEM)。Tegra 的 iGPU 不完整
支援 NVML,PyTorch 想回報 OOM 時反而先在 NVML 斷掉,所以真正的錯誤被蓋住了。
**看到 NVML assert,先當成記憶體不足來查。**

### bf16 版本塞不進去 → 改用 AWQ

`Qwen2.5-VL-3B-Instruct` 的 bf16 權重約 7.5GB,而系統本身就吃掉約 6GB,配置權重時直接
ENOMEM。改用 4-bit 的 **`Qwen2.5-VL-3B-Instruct-AWQ`**(權重僅 3.3GB),vLLM 會自動選用
`awq_marlin` kernel,Orin 的 sm_87 支援良好。

### `--gpu-memory-utilization` 在 Jetson 上的意義不同

統一記憶體架構下沒有獨立 VRAM,這個比例是對**整台機器的總記憶體**取值,而且
**作業系統已經佔用的部分也算在這個預算裡**。所以:

```
KV cache 可用量 ≈ 總記憶體 × util − (系統已用 + 模型權重 + 活化記憶體)
                ≈ 15.3GiB × util − (5.7 + 3.3 + 活化)
```

util 太低反而會讓 KV cache 分不到記憶體、直接啟動失敗 —— 這跟 x86 上的直覺相反。
目前設 `0.75`。

### `No available memory for the cache blocks` — 真正的元兇是**影片**

即使 util 調高仍然失敗,log 裡這行才是關鍵:

```
Encoder cache will be initialized with a budget of 4096 tokens,
and profiled with 2 video items of the maximum feature size.
```

`--limit-mm-per-prompt image=1` **只限制圖片,不限制影片**。Qwen2.5-VL 支援影片輸入,
vLLM 就拿「2 段最大尺寸的影片」去估活化記憶體,把預算整個吃光。這個專案只處理靜態畫面,
所以直接關掉:`--limit-mm-per-prompt image=1,video=0`。**這是讓它跑起來的決定性修正。**

同理,`--mm-processor-kwargs '{"max_pixels": 401408}'` 把單張圖片上限壓到約 512 個
視覺 token(預設允許到 16384 個),避免視覺編碼器的活化記憶體失控。

### 網路不穩會讓下載失敗且難以續傳

HF 新的 Xet/CAS 下載後端遇到連線中斷會直接噴
`CAS service error ... IncompleteBody`,而且整個 engine 啟動流程跟著失敗。
`download_model.sh` 用 `HF_HUB_DISABLE_XET=1` 改走傳統 HTTP(支援斷點續傳),
外面再包 100 次重試。把下載和啟動拆開,就不會因為抓權重失敗而連帶重跑整個啟動流程。

---

## 目前的設定

`serve_qwen2_5_vl.sh` 裡的參數,都可以用同名環境變數覆寫:

| 參數 | 值 | 原因 |
|---|---|---|
| `MODEL` | `Qwen/Qwen2.5-VL-3B-Instruct-AWQ` | 4-bit,3.3GB;bf16 塞不下 |
| `GPU_MEM_UTIL` | `0.75` | 需高於系統既有用量才有 KV cache |
| `MAX_MODEL_LEN` | `4096` | KV cache 實得 7,248 tokens |
| `MAX_NUM_SEQS` | `8` | 預設 256 會讓 KV 需求爆掉 |
| `MAX_PIXELS` | `401408` | 單圖上限 ~512 視覺 token |
| `DTYPE` | `float16` | |
| `--limit-mm-per-prompt` | `image=1,video=0` | **關掉影片 profiling** |
| `--enforce-eager` | 啟用 | 省下 CUDA graph 的額外記憶體 |

例如想改用 8192 的 context:

```bash
MAX_MODEL_LEN=8192 ./serve_qwen2_5_vl.sh
```

---

## 補充

- 這是獨立的 OpenAI 相容 HTTP 服務(port 8000),**尚未**接進 `orin-memory-spec.md` §3 的
  `/tmp/vlm.sock` sidecar 協定 —— 那不在這次任務範圍。之後要整合的話,sidecar 應該改成
  呼叫這個 server 的 `/v1/chat/completions`,而不是自己把模型載進 process。
- 記憶體很吃緊(剩約 4.7GB)。跑其他吃記憶體的東西之前,先確認 server 還活著。
