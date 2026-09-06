# Mneme — 部署與現場運維

**這份不是契約文件。** 對外 API 契約在 [`spec.md`](./spec.md),後端實作契約在
[`backend.md`](./backend.md),sidecar 的 wire protocol 與 prompt 在
[`sidecar.md`](./sidecar.md) —— 那三份有版本號,改動要先講。這一份是 README 從公開說明
裡拿掉的部分:首次安裝、`start.sh` 實際下的指令、驗證方式,以及在這台 Orin 上踩過的坑。

---

## 這台機器

JetPack 6.2 / L4T 36.4.7 / CUDA 12.6 / Python 3.10,**15.6GB 記憶體由 CPU 與 GPU 共用**。
下面幾乎每一條意外都是這個共用記憶體池的後果。

- fp16 預設值塞不下:SmolVLM2 + Qwen2.5-7B + bge-m3 約 21.6GB,對上 15.6GB。
  `sidecar/server.py --quantize` 會用 4-bit NF4 載 VLM 與 LLM;但真正的解法是把推論
  交給 vLLM(見下方「為什麼走 vLLM」)。
- `typing.Self` 要 3.11,這台是 3.10。`python3-venv` 也沒裝 —— 用
  `python3 -m virtualenv --system-site-packages`。

### 為什麼走 vLLM 而不是 transformers

同一顆 Qwen2.5-VL-3B,transformers + NF4 一次 describe 要 **105 秒**(bitsandbytes 每次
forward 都重新解量化),vLLM 的 fused `awq_marlin` kernel 約 **3 秒**。所以 sidecar 的
`--vlm-url` / `--llm-url` 指向本機的 OpenAI 相容 vLLM,describe 與 answer 共用同一個
engine;sidecar 自己只留 bge-m3,`--embed-device cpu` 再省掉第二個 CUDA context(約
1.5GB)。完整步驟見 [`vLLM/README.md`](../vLLM/README.md)。

---

## 首次安裝

```bash
# 主 venv —— 絕對不裝 torch(backend.md §8.2)。python3-venv 這台沒裝,用 virtualenv。
python3 -m virtualenv --system-site-packages .venv && .venv/bin/pip install -e .

# cv2:change filter 用它縮圖,所以主 venv 必須能 import。
# 絕對不要從 PyPI 裝 opencv-python —— 它可能從原始碼編譯,燒掉數小時(backend.md §8.2)。
# 那條規則假設 JetPack 附 Python bindings;這台只有 C++ 的 libopencv-dev 4.8.0,
# --system-site-packages 什麼都繼承不到,import cv2 直接失敗並帶走 mneme.seed。
# Jetson 的預編 wheel 才是可行路徑,而且永不編譯:
.venv/bin/pip install --only-binary=:all: \
    --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 opencv-python==4.10.0.84

# sidecar 自己的 venv,torch 只住在這裡
python3 -m virtualenv sidecar/.venv && sidecar/.venv/bin/pip install -r sidecar/requirements.txt

# vLLM:clone jetson-containers、註冊 nvidia runtime、抓 ~3.5GB 權重
git clone https://github.com/dusty-nv/jetson-containers vLLM/jetson-containers
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
cd vLLM && ./download_model.sh
```

`vLLM/jetson-containers/` 連權重約 30GB,已 gitignore。

### 依賴地雷

| 套件 | 問題 |
|---|---|
| `opencv-python` | 見上 —— 只能用 `pypi.jetson-ai-lab.io/jp6/cu126` 的預編 wheel。那顆 wheel **沒有 GStreamer 支援**,所以它開不了 CSI 相機;畫面一律由外部 GStreamer pipeline 提供 |
| `bitsandbytes` | 必須是為 **sm_87** 建的。PyPI 的 aarch64 wheel 針對資料中心 ARM,執行期死在 `named symbol not found`。同一個 jetson-ai-lab index 有正確的 |
| `transformers` | 必須 `<5`。v5 拒絕 JetPack 的 torch 2.5(`PyTorch was not found`),而且把 `from_pretrained` 的 `torch_dtype` 改名成 `dtype` |
| `hf_xet` | HuggingFace 的 Xet backend 在爛網路下會停住,續傳後寫出「大小完全正確、header 是垃圾」的檔案。`HF_HUB_DISABLE_XET=1` 對 huggingface_hub 0.30 無效,`pip uninstall hf_xet` 才真的切回傳統可續傳 HTTP。**驗權重請解析 safetensors header,不要比對檔案大小** |

---

## 相機

兩顆鏡頭,`start.sh` 裡的 `CAM_SRC` 選一顆(預設 `usb`)。**兩顆都不經過
`cv2.VideoCapture`**,畫面一律由 `--camera-cmd` 的 GStreamer pipeline 寫成 JPEG
(`spec.md` §7),透過 `multifilesink` 落地 —— `filesink` 配 `num-buffers=1` 不 flush 會掛住。

- **USB**(`CAM_SRC=usb`)—— `/dev/video1` 上的 Generic USB Camera,廣角,**原生吐
  MJPG** 1280x720。所以這條 pipeline 既不解碼也不編碼:
  `v4l2src ! image/jpeg ! jpegparse ! videorate ! multifilesink`。`videorate` 確實能
  協商 `image/jpeg`,但前面一定要有 `jpegparse`。
  一律以 `/dev/v4l/by-id/usb-Generic_USB_Camera_*-video-index0` 定址,**不要用
  `/dev/video1`** —— 重插 USB 後編號會變。
- **CSI**(`CAM_SRC=csi`)—— ribbon cable 上的 IMX219,吐 10-bit Bayer(`RG10`),
  `cv2.VideoCapture` 解不了。這條 pipeline 必須轉換(`nvvidconv`)並編碼(`jpegenc`)。
  sensor 下限是 21fps,所以用 `videorate` 節流。pipeline 被砍會把 Argus session 卡住
  (`Failed to create CaptureSession`),`sudo systemctl restart nvargus-daemon` 清掉。
  **demo day 值得先知道這條。**

兩條都跑 **21fps 而非 1fps**:web UI 把這些畫格當即時 MJPEG 串流播(`spec.md` §2.8),
backend 在送進 change filter 前自己降到約 2/s,所以 VLM 的負載不變。畫格以相機自己的
JPEG bytes 進串流 —— 不解碼、不重新編碼 —— 這才讓 21fps 在 vLLM 旁邊付得起。

---

## `start.sh` 實際下的四道指令

```bash
M=Qwen/Qwen2.5-VL-3B-Instruct-AWQ
export HF_HUB_CACHE=$PWD/vLLM/jetson-containers/data/models/huggingface

# 1. vLLM:一顆 VL 模型同時服務 describe 與 answer,約 40s 載入
#    先 drop page cache —— vLLM 的記憶體預檢把 page cache 算成已用,
#    長時間開機的機器會因此拒絕啟動。
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
( cd vLLM && GPU_MEM_UTIL=0.78 MAX_MODEL_LEN=2048 ./serve_qwen2_5_vl.sh )

# 2. sidecar:VLM/LLM 走 HTTP,只有 embedder 常駐,而且在 CPU 上
sidecar/.venv/bin/python sidecar/server.py --backend cuda \
    --vlm-url http://127.0.0.1:8000/v1 --vlm $M \
    --llm-url http://127.0.0.1:8000/v1 --llm $M \
    --embed-device cpu --socket /tmp/vlm.sock --data-dir ./data

# 3. backend:相機透過 GStreamer,不透過 cv2。--sidecar-timeout-ms 必須調高,
#    一次 describe 是兩輪 VLM 約 14s,撞到 20s 預設會斷線並卡住 pipeline。
.venv/bin/python -m mneme --data-dir ./data --sidecar /tmp/vlm.sock \
    --bind 0.0.0.0:8080 --sidecar-timeout-ms 60000 --capture-fps 21 \
    --camera-cmd 'gst-launch-1.0 v4l2src device=/dev/v4l/by-id/usb-Generic_USB_Camera_200901010001-video-index0 ! image/jpeg,width=1280,height=720,framerate=30/1 ! jpegparse ! videorate ! image/jpeg,framerate=21/1 ! multifilesink location=frame_%05d.jpg'

# 4. telegram bot(選用):自己從 .env 讀 TELEGRAM_API_KEY
.venv/bin/python bot/telegram_bot.py --api http://127.0.0.1:8080
```

順序不能換:sidecar 的 VLM/LLM 由 vLLM 提供,backend 又等 sidecar 的 socket。
log 落在 `./run/`。

### SIGTERM 是個真問題

`/api/stream` 與 `/api/frames/live.mjpg` 永遠不會自己結束,所以 uvicorn 的 *graceful*
shutdown 會無限等它們:process 已經放掉 port(下一個 start 綁得上、看起來很健康),卻
還握著相機、還在寫進同一個 `data/incoming`,於是兩條 capture pipeline 把兩個房間交錯
進同一條 timeline。`uvicorn.run` 因此把 `timeout_graceful_shutdown` 釘在 5 秒,
`./start.sh stop` 也會升級成 `kill -9`。

---

## 環境變數

全部 CLI flag 與 `MNEME_*` 環境變數見 [`backend.md` §8.3](./backend.md#83-cli-與環境變數)
—— 那裡是唯一真相,這裡不重複列表。現場最常要動的兩個:

- `MNEME_DIFF_THRESHOLD` —— change filter 閾值,預設 `12.0`。換場地的光線就要重調。
- `MNEME_SIDECAR_TIMEOUT_MS` —— 預設 `20000`,但一次 describe 約 14s,`start.sh` 已改成 `60000`。

---

## 驗證

### 沒有 Orin:mock sidecar

```bash
.venv/bin/python -m mneme.seed --out data/memory.db --data-dir ./data --hours 8 --count 60 --seed 42
.venv/bin/python -m mneme --no-camera --mock-sidecar
```

`--mock-sidecar` 用 in-process 的確定性假模型取代 CUDA 推論(規則見
[`sidecar.md` §8.5](./sidecar.md#85-mock-sidecar))。`/api/health` 必須誠實回
`sidecar: "mock"`、`mode: "seed-only"` —— 不要為了好看假造這兩個欄位。

[`backend.md` §8.8](./backend.md) 的驗收清單是目前最接近測試套件的東西;任何動到 API、
schema 或檢索的改動之後都要跑。最後兩條是硬性要求:拒答路徑,以及固定 `--seed` 下
`python -m mneme.seed` 的 byte-identical 可重現性。

### 真推論的 wire contract

`--mock-sidecar` 驗得到 API 層,但驗不到 §3.1 的 wire protocol —— mock 根本不開 socket。
`scripts/verify_sidecar.py` 驗的就是這一段,共 33 條檢查。

Apple silicon 上可以離開 Orin 跑真的:

```bash
python3 -m venv sidecar/.venv
sidecar/.venv/bin/pip install -r sidecar/requirements-mlx.txt
.venv/bin/python scripts/verify_sidecar.py
```

它會起一個 `sidecar/server.py --backend mlx`(MLX 版的同樣三個模型),把
[`sidecar.md` §3.1 / §3.2](./sidecar.md#31-wire-protocol) 的每一條與 §8.8 的硬性拒答
測完,**並用真的攝影機畫面驗 `describe`**(抓兩張要求 summary 不同,擋掉「無視像素、
每次吐同一句」的假通過),全過才 exit 0。沒鏡頭就 `--no-camera` 跑 33 條協議檢查。
細節與三則實測結論見 [`sidecar.md` §8.9](./sidecar.md#89-在-macos-上驗證真-sidecar)。

在 Orin 上驗的是**部署形狀**,所以要帶跟 `start.sh` 一樣的 URL,不要讓它自己 load 一份
塞不下 vLLM 旁邊的權重:

```bash
.venv/bin/python scripts/verify_sidecar.py --backend cuda --no-camera \
    --vlm-url http://127.0.0.1:8000/v1 --vlm $M \
    --llm-url http://127.0.0.1:8000/v1 --llm $M \
    --embed-device cpu --data-dir /tmp/verify --db /tmp/verify/memory.db
```

**要給它一份全新 seed 的、自己專用的資料庫。** 它拿最新 16 筆事件當語料,卻對其中一筆
seed 資料(馬克杯那筆)下斷言;真實 capture 跑過之後 seed 資料會被擠出去,於是報三個
失敗 —— 那是 fixture 的錯,不是程式的錯。這條燒掉過真實的 debug 時間:這支腳本最需要
被信任的時候,正是它最容易誤導人的時候。

---

## demo day 出事時先看這裡

| 症狀 | 真正的原因 |
|---|---|
| vLLM 說 `No available memory for the cache blocks` | `--gpu-memory-utilization` 是**開機前的預算檢查**,不是上限,而且分母是系統總記憶體、page cache 也算已用。設**低**反而起不來 —— 這跟 x86 的直覺相反。實際 KV 配置由 `--max-model-len` 界定。`sync && echo 3 \| sudo tee /proc/sys/vm/drop_caches` 之後再試(`start.sh` 已經會做)。 |
| torch 噴 `NVML_SUCCESS == r INTERNAL ASSERT FAILED` 或 `NvMapMemAllocInternalTagged: error 12` | 這是 OOM,不是 bug。Tegra 的 iGPU 不完整支援 NVML,所以 torch 在回報 OOM 的路上死在 NVML 裡。 |
| CSI 相機 `Failed to create CaptureSession` | 上一個 pipeline 被砍時把 Argus session 卡住了。`sudo systemctl restart nvargus-daemon`。 |
| capture 看起來跑兩倍快、timeline 交錯兩個房間 | 有第二個 `python -m mneme` 還活著(見上方「SIGTERM 是個真問題」)。先 `ps` 找,再看程式碼。 |
| 模型權重載入時報奇怪的格式錯誤 | HuggingFace 的 Xet backend 把中斷續傳的檔案寫成「大小正確、header 是垃圾」。`pip uninstall hf_xet`,並用解析 safetensors header 的方式驗權重。 |
| `import cv2` 失敗,`mneme.seed` 一起掛 | 主 venv 沒有 Python 的 cv2 —— 這台只有 C++ 的 `libopencv-dev`。裝上面那顆 jetson-ai-lab 的預編 wheel。 |
| bitsandbytes `named symbol not found` | 裝到 PyPI 的資料中心 ARM wheel。換 jetson-ai-lab 的 sm_87 版。 |
