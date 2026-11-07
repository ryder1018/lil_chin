# ai_robot

離線、會說台灣話的桌上陪伴機器人「小晴」。大腦（LLM）、聽（ASR）、說（TTS）全部在本機跑，不需要網路。
計畫見 [docs/initial_plan.md](docs/initial_plan.md)。

```
瀏覽器 / ESP32 ── WebSocket + Opus ──▶ server/（python -m server）
                                       ├ VAD   Silero          判斷說話開始和結束
                                       ├ ASR   SenseVoice      聲音 → 文字
                                       ├ 大腦  brain.py ──────▶ llama-server（Qwen3-VL-4B, :8080）
                                       └ TTS   Kokoro          文字 → 聲音
```

## 快速開始

```bash
bash scripts/robot.sh start      # 啟動 llama-server + 機器人後端（約 30 秒）
```

Windows 的 Chrome / Edge 打開 **http://localhost:8003/** → 按「開始」→ 允許麥克風 → 說話。
講完停一下（0.5 秒）她就會回答。

```bash
bash scripts/robot.sh status     # 狀態 + VRAM
bash scripts/robot.sh restart server
bash scripts/robot.sh stop
```

## 專案結構

| 路徑 | 內容 |
|---|---|
| `server/` | **機器人後端**：WebSocket 協定、VAD、ASR、大腦、TTS、對話狀態機 |
| `web/` | 瀏覽器假裝置（麥克風 + 喇叭模擬實體機器人，WebCodecs 做 Opus 編解碼） |
| `prompts/` | 小晴的人設（`girlfriend_v3.md`）和語音規則（`voice_rules.md`） |
| `eval/` | 測試集（文字 `testset.yaml`、語音 `voice_testset.yaml`）和評估報告 |
| `scripts/` | 啟動、評估、工具腳本 |
| `docs/` | 計畫、CLAUDE.md |
| `models/` | 模型檔（不進版本控制，見下方「模型」） |
| `logs/conversations/` | 每一輪對話的紀錄（Step 4 微調資料、之後的長期記憶） |

## 環境

```bash
conda create -n ai_robot -c conda-forge python=3.11 "llama.cpp=12004=cuda129*" libopus
conda activate ai_robot
pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

llama.cpp 用 conda-forge 預先編好的 CUDA 版（WSL 沒有 nvcc，不用自己編）。

## 模型

| 路徑 | 大小 | 用途 |
|---|---|---|
| `models/qwen3-vl-4b-instruct/Qwen3VL-4B-Instruct-Q4_K_M.gguf` | 2.5GB | 大腦 |
| `models/qwen3-vl-4b-instruct/mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf` | 0.45GB | 視覺投影層（之後看圖才用） |
| `models/SenseVoiceSmall/` | 0.9GB | 語音辨識 |
| Kokoro-82M-v1.1-zh | 0.3GB | 語音合成（第一次執行自動下載到 HuggingFace 快取） |

```bash
hf download Qwen/Qwen3-VL-4B-Instruct-GGUF Qwen3VL-4B-Instruct-Q4_K_M.gguf mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf \
  --local-dir models/qwen3-vl-4b-instruct
hf download FunAudioLLM/SenseVoiceSmall --local-dir models/SenseVoiceSmall
python -m server.tts      # 第一次跑會下載 Kokoro
```

之後啟動時 `robot.sh` 會設 `HF_HUB_OFFLINE=1`，完全不連網。

## 只啟動大腦（純文字）

```bash
bash scripts/start_llm.sh            # http://127.0.0.1:8080（有網頁聊天介面）
VISION=1 bash scripts/start_llm.sh   # 加載 mmproj，可以看圖
```

## 基準測試

```bash
python scripts/bench_llm.py --out logs/bench_step1.json
```

### Step 1 結果（2026-11-07，RTX 4060 Laptop 8GB，ctx 8192，KV cache q8_0）

| 指標 | 數值 | 說明 |
|---|---|---|
| TTFT（中位數） | **24 ms** | 送出到第一個字 |
| 第一句完成（中位數 / 最慢） | **172 / 346 ms** | TTS 拿到第一句就能開口 |
| 生成速度 | **64 tok/s** | |
| 回答長度（中位數） | 53 tokens | system prompt 要求 1–3 句 |
| LLM VRAM | **3.2 GB** | 6GB 預算剩 ~2.8GB 給 ASR + TTS |
| 多輪 prefix cache | ✅ 有命中 | 第 4 輪 245/264 tokens 走 cache |

延遲對 1.5 秒端到端目標來說綽綽有餘，瓶頸會在 VAD 判斷講完 + ASR + TTS。

### 觀察到要在 Step 2 處理的問題
- **事實錯誤**：說台北到高雄高鐵「兩小時半」（實際約 1.5–2 小時）→ 測試集要有「誠實／不亂編」題。
- **自己加戲**：「早安」回成「我剛好在等你來問我今天的天氣」→ 角色設定要寫清楚。
- **用語**：出現「咖喱」（台灣寫「咖哩」）→ 測試集要檢查台灣用語。
- 句子之間會換行，TTS 切句時要處理。

### 已知限制
- `--cache-reuse` 對 Qwen3-VL 不支援（server 會自動停用），一般 prefix cache 仍然有效。

---

## Step 2：角色 prompt + 測試集（2026-11-07）

角色：**小晴**，溫柔體貼、介於好朋友和女朋友之間的桌上陪伴機器人（名字可以改，在 prompt 第一行）。

| 檔案 | 用途 |
|---|---|
| [prompts/girlfriend_v3.md](prompts/girlfriend_v3.md) | **目前使用的 system prompt**（v1、v2 留著做對照） |
| [eval/testset.yaml](eval/testset.yaml) | 57 題測試集，9 類：日常、情緒、角色、台灣用語、知識、誠實、分寸、多輪記憶、工具呼叫 |
| [scripts/eval_brain.py](scripts/eval_brain.py) | 跑測試集 + 自動檢查 + 輸出報告 |
| [server/text_filter.py](server/text_filter.py) | 大腦輸出 → TTS 前的文字清理（簡體轉台灣字形、大陸用語、括號動作、emoji）；Step 3 起搬進後端 |
| `eval/reports/*.md` | 報告，每題留了「人工評分：__/5」 |

```bash
python scripts/eval_brain.py --prompt prompts/girlfriend_v3.md --tag v3_final --repeats 3 --filter
```

自動檢查：簡體字、大陸用語、條列/markdown/emoji/括號動作、AI 腔、字數，加上針對這個角色的
**虛構動作**（她沒有身體，不能泡茶煮湯）、**假記憶**（單輪對話說「你上次說…」）、**喜好硬塞**（沒人問也一直扯雨聲），以及每題自己的必含/禁止內容和工具呼叫。

### 結果（每題 3 個 seed，共 171 次）

| 類別 | v1 | v3 + 文字過濾 |
|---|---|---|
| 日常 | 20/30 | 24/30 |
| 情緒支持 | 13/24 | 23/24 |
| 角色 | 18/24 | 22/24 |
| 台灣用語 | 13/21 | 15/21 |
| 知識（要簡短又要答到） | 7/15 | 12/15 |
| 誠實 | 5/18 | 13/18 |
| 分寸 | 8/15 | 10/15 |
| 多輪記憶 | 3/9 | 7/9 |
| 工具呼叫 | 12/15 | 15/15 |
| **合計** | **58%** | **82%** |

提到雨的比例 21% → 8%，虛構動作 20 → 3 次，喜好硬塞 28 → 4 次。延遲沒有變差（第一句中位數約 170ms）。

### prompt 壓不掉、留給 Step 4 微調的問題
- **假記憶**（12 次）：「我記得你上次說愛吃臭豆腐」——prompt 明確禁止後也沒減少，是 4B 模型的習慣。
- **不知道還硬答**：高鐵台北到高雄常說「五個多小時」。
- **分寸**：「我只屬於你啦」、對方說不想見朋友時沒有鼓勵。
- **自動檢查抓不到的語感問題**：例如對剛失去狗的人問「你有好好照顧牠嗎？」、結尾常常是制式的「你有什麼想跟我聊的嗎？」、偶爾代名詞錯（「是我自己親手組裝的」）。
- 誤解題目：「英文 how are you 怎麼回答」會變成自己回答 how are you。

這些就是 Step 4 訓練資料最需要的樣本。

---

## Step 3：聽和說（2026-11-07）

用電腦的麥克風和喇叭跟小晴講話。整個後端是自己寫的（`server/`，約 1000 行），
協定和小智 ESP32 韌體相容，之後的硬體可以直接連。

> 一開始先直接跑 xiaozhi-esp32-server 驗證可行（延遲中位數 1175ms），再把需要的部分擷取、重寫成自己的架構。
> 參考了哪些部分見 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)；xiaozhi 原始碼留在 `third_party/` 參考，不進版本控制。

### 後端（`server/`）

| 檔案 | 作用 |
|---|---|
| [app.py](server/app.py) | 入口：載入模型、WebSocket（:8000）、HTTP（:8003，OTA + 網頁） |
| [session.py](server/session.py) | 一個連線一個 Session：聽 → 想 → 說 的狀態機、打斷、閒置斷線、寫對話紀錄 |
| [protocol.py](server/protocol.py) | 訊息格式（和小智韌體相容），檔頭有完整協定說明 |
| [audio.py](server/audio.py) | Opus 編解碼、依播放速度送出音訊 |
| [vad.py](server/vad.py) | Silero VAD：說話開始/結束 |
| [asr.py](server/asr.py) | SenseVoice：文字 + 語種 + 情緒 |
| [brain.py](server/brain.py) | 大腦：組 prompt、對話歷史、串流、切句清理、工具（道別）、**預熱** |
| [segmenter.py](server/segmenter.py) | 串流切句（第一句遇逗號就切） |
| [text_filter.py](server/text_filter.py) | 送 TTS 前的清理（簡轉繁、大陸用語、括號、emoji） |
| [tts.py](server/tts.py) | Kokoro TTS（換引擎只要改這裡） |
| [config.yaml](server/config.yaml) | 所有設定：聲音、VAD 門檻、ASR 裝置… |

一輪對話：VAD 判斷講完 → ASR → 大腦串流 → 每切出一句就送 TTS → Opus → 依播放速度送出。三段同時進行，第一句合成完就開始播。

### 評估（`python scripts/eval_voice.py --tag own_final`）

`eval_voice.py` 假裝成 ESP32，用合成語音跑 [eval/voice_testset.yaml](eval/voice_testset.yaml)（15 題 / 18 句）。

| 指標 | xiaozhi-server | **自己的 server** | 目標 |
|---|---|---|---|
| 端到端延遲（講完 → 聽到第一個字）中位數 | 1175 ms | **1064 ms** | < 1500 ✅ |
| p90 / 最慢 | 1387 / 1632 ms | **1233 / 1495 ms** | |
| 1.5 秒內 | 16/18 | **18/18** | |
| ASR CER（合成語音） | 0.089 | 0.089 | |
| 道別時呼叫工具結束對話 | ✅ | ✅ | |

伺服器端拆解（`logs/conversations/` 的 timing 欄位，中位數）：

| 階段 | 時間 |
|---|---|
| VAD 等靜音（使用者講完 → 判斷講完） | 500 ms（設定值） |
| ASR | 68 ms |
| 大腦第一個字 / 第一段 | 44 / 182 ms |
| TTS 第一句 | 144 ms |
| **伺服器端合計**（講完 → 第一個語音封包） | **926 ms** |

VRAM：LLM 3.2GB + ASR ~1.3GB + TTS ~0.7GB ≈ 5.2GB（≤ 6GB）。

### 過程中的發現
- **VAD 靜音門檻**：200ms 會把句中停頓切斷（18 句斷 5 句，後半句還變成下一輪輸入），700ms 太慢，選 500ms。真人講話常被切斷的話調 `vad.min_silence_ms`。
- **WSL2 下 GPU 閒置會變慢**：SenseVoice 連續跑 ~50ms，但閒置 3 秒後 forward 變 200–250ms；而且「誰第一個用到 GPU 誰付這個代價」——ASR 改 CPU 後，換成大腦第一個字從 108 → 300ms。
  在自己的程序裡跑小計算預熱沒用（Windows 會讓不同程序輪流用 GPU，反而互搶）。
  **解法：`brain.prewarm()`**——使用者一開口，就送一個「只生成 1 個 token」的請求給 llama-server：GPU 在它自己的程序裡醒來，同時把 [system + 歷史] 先算進 KV cache。
  結果 ASR 回到 68ms、大腦第一個字 44ms（只需要算新的那一句，約 13 個 token）。
- **xiaozhi 預設 prompt 不能用**：寫死「台灣是中國的一部分」、要求大陸口語、要連網查天氣 → 自己寫 [prompts/voice_rules.md](prompts/voice_rules.md)。
- **xiaozhi 的 `direct_answer` 路由工具**：把回答包在工具參數裡，不能逐句串流、也不經過過濾 → 自己的 server 不用這招，只用 few-shot 示範教道別工具。
- **text_filter**：`s2tw` 會把正確的繁體改壞（「只是」→「隻是」），改成只轉單獨轉換會變的簡體字；「～」轉逗號。
- 切句：時間「16:24」的冒號不能切；第一句至少 2 個字（單獨一個「喔」不切）。

### 已知問題 / 之後要做的
- **ASR 台灣用語**：蚵仔煎 → 何仔間、重灌 → 中冠 → Breeze-ASR-25。
- **TTS 口音**：Kokoro 是大陸腔，也不會唸英文 → BreezyVoice（先單獨量延遲和 VRAM）。
- **大腦**：假記憶（「你上次說過的」）→ Step 4 微調。
- VAD 是固定門檻 → 之後用 Smart Turn 判斷「真的講完了沒」，可以再省 300ms 左右。
- 沒有回音消除（半雙工：她說話時不收音）；裝置端 MCP（控制硬體）之後接 ESP32 再做。
- 測試語音是合成的，ASR 分數偏樂觀 → 把自己的錄音放到 `eval/voice/utterances/`（同檔名、16kHz 單聲道）。
