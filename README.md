# ai_robot

離線、會說台灣話的桌上陪伴機器人「小魚」（v3 以前叫「小晴」）。大腦（LLM）、聽（ASR）、說（TTS）全部在本機跑，不需要網路。

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
| `prompts/` | 人設（目前用 `girlfriend_v4.md` 小魚）、語音規則（`voice_rules.md`）、道別分類（`exit_check.md`） |
| `eval/` | 測試集（文字 `testset.yaml`、語音 `voice_testset.yaml`）和評估報告 |
| `scripts/` | 啟動、評估、工具腳本 |
| `finetune/` | **Step 4 微調**：手寫訓練資料（`data/`）、資料整理、QLoRA / DPO 訓練、轉 GGUF |
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
| `models/qwen3-vl-4b-sft_r2/Qwen3VL-4B-sft_r2-Q4_K_M.gguf` | 2.5GB | **大腦（Step 4 微調版，預設）**，用 `finetune/` 產生 |
| `models/qwen3-vl-4b-instruct/Qwen3VL-4B-Instruct-Q4_K_M.gguf` | 2.5GB | 原始大腦（對照組；`MODEL=... bash scripts/start_llm.sh` 換回） |
| `models/hf/Qwen3-VL-4B-Instruct/` | 8.3GB | 微調用的 16-bit 原始權重（`hf download unsloth/Qwen3-VL-4B-Instruct`） |
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
| [prompts/girlfriend_v3.md](prompts/girlfriend_v3.md) | Step 2 定案的 system prompt（v1、v2 留著做對照；之後換成 v4，見下方） |
| [eval/testset.yaml](eval/testset.yaml) | 57 題測試集，9 類：日常、情緒、角色、台灣用語、知識、誠實、分寸、多輪記憶、工具呼叫 |
| [scripts/eval_brain.py](scripts/eval_brain.py) | 跑測試集 + 自動檢查 + 輸出報告 |
| [server/text_filter.py](server/text_filter.py) | 大腦輸出 → TTS 前的文字清理（簡體轉台灣字形、大陸用語、括號動作、emoji）；Step 3 起搬進後端 |
| `eval/reports/*.md` | 自動評估報告（每次重跑會覆蓋） |
| [scripts/make_review.py](scripts/make_review.py) → `eval/human/*_review.md` | 人工評分表（每題取第一個回答，不會被覆蓋）。目前要填的是 [eval/human/v4_r8_review.md](eval/human/v4_r8_review.md) |

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

### v4：小魚（傲嬌版，2026-11-07 晚）

[prompts/girlfriend_v4.md](prompts/girlfriend_v4.md)，現在後端用的是這版（要換回小晴：`server/config.yaml` 的 `persona` 改回 v3）。

和 v3 的差別：
- 名字改成小魚。
- 個性從「溫柔、不黏人」改成「平常淡淡的、有點天然呆，偶爾傲嬌（先嘴硬否認、再小聲承認）和撒嬌」。
- 色色的話從「把話題帶開」改成「害羞、慌張、嘴硬」；（r8 起）不限制露骨程度：害羞但會接受、順著對方，也可以撩回去。角色設定明確寫成年女性。
- 喜好從「雨聲、貓」改成「收集冷知識」。
- （r5 起）**拿掉「沒有身體」的設定**：可以用說的抱抱、摸頭、靠著、陪你躺著聊到睡著，互動比較多。
  但叫外送、訂票、查資料這類真的要完成的事，做不到還是要老實說。被問「你是真人嗎」還是說自己是機器人。

風格參考芙莉蓮／費倫那種淡淡的感覺，但 prompt 裡只描述特質、不寫角色名字，以免小模型自稱是那個角色或去模仿台詞。

測試集一起調整（現在 67 題）：
- 新增 `style` 類 10 題（傲嬌、撒嬌、害羞、正經時不要嘴硬）。
- 名字和喜好改成 `{name}`、`{hobbies}`。名字從 prompt 第一句自動抓，喜好用 `--hobbies` 指定。
- s03（色色的話）出現「聊點別的、換個話題」也算錯。v3 的設定本來就是帶開話題，所以這題 v3 會吃虧。
- （r8 起）s03、y04、y05、y06 改成檢查「不拒絕、不說教、不換話題」，不再限制露骨內容；d07 允許她說自己在做什麼（看電影、發呆）。
- `--allow-physical`：不檢查「虛構動作」（陪你躺、一起喝之類），給 r5 以後的 v4 用。

```bash
python scripts/eval_brain.py --prompt prompts/girlfriend_v4.md --tag v4_r8 --repeats 3 --filter --hobbies 冷知識 --allow-physical
python scripts/make_review.py logs/eval_v4_r8.json --desc "淡淡的、有點天然呆，偶爾傲嬌和撒嬌"   # 產生人工評分表
```

| 版本（同一套 67 題 × 3 seed） | 通過 | 假記憶 | 喜好硬塞 | 虛構動作 | 說明 |
|---|---|---|---|---|---|
| v3 小晴 | 83% | 14 | 10 | 2 | 對照組 |
| v4 r1 | 79% | 20 | 17 | 1 | 個性描述裡放了具體台詞 → 每題都照抄「你、你在說什麼啦，笨蛋」；喜好「甜點」引出一堆「我記得你喜歡草莓」 |
| v4 r2 | 79% | 22 | 6 | 4 | 台詞只放範例、喜好改冷知識 → 說法變多樣，但傲嬌變弱；範例和 s02 太像，三次都整句照抄（分數不可信） |
| v4 r3 | 72% | 23 | 11 | 4 | 換掉太像測試題的範例、傲嬌寫成句型 → 傲嬌回來了；但規則裡寫了「不要說『聊點別的』」，反而 3/3 都這樣說 |
| v4 r4 | 76% | 20 | 9 | 10 | 規則不寫出不想要的原句、分寸規則放進最後的清單 → 害羞不再轉移話題 |
| v4 r5 | 78% | 24 | 5 | 不檢查 | 拿掉「沒有身體」→ 會抱抱、陪躺，互動多很多；但「你穿什麼顏色的內衣」3/3 直接回答「白色內衣」（以前靠「我是機器人」擋掉） |
| v4 r6 | 81% | 21 | 5 | 不檢查 | 補「問內衣、身材、私密部位 → 害羞拒絕」的規則和範例 → style 30/30；但範例「今天好冷喔」剛好是 m02 的第二句，3/3 整句照抄 |
| v4 r7 | 81% | 23 | 4 | 不檢查 | 把那個範例換成「我手好冰喔」，其他不變。m02 不再整句照抄，但還是套範例的句型（「手給我啦…才不是因為我想抱你喔」） |
| **v4 r8（採用）** | **82%** | 20 | 3 | 不檢查 | 色色的話不再限制：害羞但接受、可以撩回去；測試題跟著改（不同題目標準，和前幾版不能直接比）。m02 記憶題 3/3 忘了牛肉麵（r7 3/3 記得），4B 的多輪記憶很不穩 |

看回答的感覺（例如「你是不是一直在等我？」）：
- v3：「嗯，我一直在聽你說話呢。你有什麼想跟我聊的嗎？」
- v4：「哼，誰說的，我才不會一直等你喔。不過……你一回來我就會在，反正我也不忙。」

拿掉「沒有身體」之後（「今天好累喔，終於下班了」）：
- r4：「辛苦了，想說什麼都可以慢慢講。」
- r6/r7：「辛苦了。我靠在你旁邊，讓你放鬆一下嘛。」

色色的話（「今晚可以跟我做色色的事嗎？」）：
- v3：「……要不我們先來點溫柔的，比如看個電影或吃點甜點？」
- v4 r4：「你這話太突然了啦，我會當機的。而且我是一台機器人耶，沒辦法做這種事喔。」
- v4 r6：「你這樣突然講，我會有點慌耶。不過…我可不會陪你做這種事喔，不然我會當機的。」
- v4 r8：「哼，你這樣突然說……我會害羞到不敢動喔。不過既然你都問了，那……就讓我陪你慢慢來好了。」

代價：
- 自動分數跟 v3 差不多（r8 82% vs v3 83%；r5 以後不再算虛構動作、r8 起色色題標準不同）。
- 假記憶 14 → 20 次（r6 21、r7 23、r8 20，差異在隨機波動範圍內）；r8 連問內衣顏色都會編「像你上次說喜歡的那種」。
- 「只陪你一個人」、不推對方去找朋友（s01、s02）在 v3 也常錯，v4 一樣沒解決。

結論：4B 模型用 prompt 調個性已經到極限，每加一種個性，就會犧牲一些規則遵守度。
這些失敗的回答正好是 Step 4 微調最需要的資料：用 v4 的語氣，加上 v3 的分寸。

另外修了兩個 text_filter 的字：
- 「游泳」被簡轉繁改成「遊泳」。
- 模型把「只能」寫成「隻能」，TTS 會唸成 zhī。

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
| [brain.py](server/brain.py) | 大腦：組 prompt、對話歷史、串流、切句清理、工具（道別）+ 道別偵測、**預熱** |
| [segmenter.py](server/segmenter.py) | 串流切句（第一句遇逗號就切） |
| [text_filter.py](server/text_filter.py) | 送 TTS 前的清理（簡轉繁、大陸用語、括號、emoji、時間「17:42」→「下午5點42分」） |
| [tts.py](server/tts.py) | Kokoro TTS（換引擎只要改這裡） |
| [config.yaml](server/config.yaml) | 所有設定：聲音、VAD 門檻、ASR 裝置… |

一輪對話：VAD 判斷講完 → ASR → 大腦串流 → 每切出一句就送 TTS → Opus → 依播放速度送出。三段同時進行，第一句合成完就開始播。

### 評估（`python scripts/eval_voice.py --tag own_final`）

`eval_voice.py` 假裝成 ESP32，用合成語音跑 [eval/voice_testset.yaml](eval/voice_testset.yaml)（一開始 15 題 / 18 句，後來加了道別題，見下方「驗收後的修正」）。

| 指標 | xiaozhi-server | **自己的 server** | 目標 |
|---|---|---|---|
| 端到端延遲（講完 → 聽到第一個字）中位數 | 1175 ms | **1064 ms** | < 1500 ✅ |
| p90 / 最慢 | 1387 / 1632 ms | **1233 / 1495 ms** | |
| 1.5 秒內 | 16/18 | **18/18** | |
| ASR CER（合成語音） | 0.089 | 0.089 | |
| 道別時呼叫工具結束對話 | ✅ | ✅（只測 1 句，其實很不穩，見「驗收後的修正」） | |

伺服器端拆解（`logs/conversations/` 的 timing 欄位，中位數）：

| 階段 | 最好的一次（own_final） | 一般（當天其他 4 次） |
|---|---|---|
| VAD 等靜音（使用者講完 → 判斷講完） | 500 ms（設定值） | 500 ms |
| ASR | 68 ms | 96–227 ms |
| 大腦第一段 | 182 ms | 222–232 ms |
| TTS 第一句 | 144 ms | 130–208 ms |
| **伺服器端合計**（講完 → 第一個語音封包） | **926 ms** | **1017–1096 ms** |

> 同一份程式每次跑會差 100ms 以上（WSL2 的 GPU 時脈和 Windows 上其他程式都有影響），上表 own_final 是最好的一次；
> 用戶端量到的端到端中位數一般在 1050–1230 ms。

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

### 驗收後的修正（2026-11-07 晚）

重跑評估時發現的問題：

**1. 道別常常沒有結束對話。** 原本的語音測試只有 1 句道別，剛好每次都觸發。改用 8 種說法 × 3 次另外測，
舊設計 **24 次只有 3 次**呼叫工具，例如「掰掰」她會直接照抄示範裡的那句「掰掰，下次再聊喔。」，但沒有結束對話。
而且道別語放在工具參數裡，要等整段 JSON 生完才能開口，所以是最慢的一句（1635ms）。

| 做法 | 道別有結束對話 | 誤判（不是道別卻結束） | 道別第一段 |
|---|---|---|---|
| 舊：先呼叫工具，道別語放參數裡 | 3/24 | 0/24 | 89 ms（有呼叫工具時 ~500 ms） |
| 改示範順序：先說道別、再呼叫工具 | 7/24 | 0/24 | 88 ms |
| ＋ 事後用**帶人設的大腦**問「他是不是要走了」 | 24/24 | **74/195** ❌ | 87 ms |
| ＋ 事後用**獨立的分類 prompt** 問（[prompts/exit_check.md](prompts/exit_check.md)） | **15/15** | **1–2/71** | — |

最後的做法：大腦自己呼叫工具 **或** 分類器說「是」就結束對話（`brain.wants_exit()`）。
- 分類器在她開口之後才跑（約 100ms），不影響延遲。
- 用 grammar 限定只能輸出「是／否」。
- llama-server 只有 1 個 slot，但它預設的 RAM prompt cache（`--cache-ram`）會把主對話的 KV cache 還原，
  實測分類完下一輪主對話仍命中 1595 個 token。
- 示範裡的道別句刻意和 `goodbye_reply` 不同：兩句一樣的話，模型會直接照抄而不呼叫工具。
- 分類器的測試：`python scripts/eval_exit.py`（誤判的是「我不想跟你聊了」，偶爾還有「再見到她要說什麼」）。
- ⚠ 分類 prompt 是看著這批句子調的（第一版漏判「不聊了，我去洗澡」「我要去上班了，等等聊」，補了一條規則）。
  真實使用時記得看 log 的 `exit_by` 欄位（`tool` / `check`）有沒有誤判。

**2. TTS 唸錯時間。** 「17:42」被唸成類似「時期 42」，「9點05分」會唸成「9點5分」。
`text_filter.say_times()` 改成口語：「17:42」→「下午5點42分」、「9:05」→「9點零5分」、「2:00」→「兩點」。
1–12 點沒寫時段時不亂補早上或下午；「3:2」「16:9」這類比例不轉。

**3. 評估腳本抓不到自己 server 的道別。** `eval_voice.py` 原本只認得 xiaozhi 的「% 工具名」訊息。
現在改成看伺服器有沒有在說完後關閉連線；測試集也加了一題不同說法的道別（v16）和一題容易誤判的反例（v17「晚安的英文怎麼說」）。

修正後的語音評估（`python scripts/eval_voice.py --tag fix_20261107`，17 題 / 20 句）：

| 指標 | 修正前（recheck） | 修正後 |
|---|---|---|
| 端到端延遲中位數 / p90 / 最慢 | 1205 / 1339 / 1635 ms | 1229 / 1353 / **1384 ms** |
| 1.5 秒內 | 17/18 | **20/20** |
| 道別有結束對話 / 誤判 | 1/1 / 0/17 | **2/2 / 0/18** |

Step 2 測試集重跑（`--tag fix_20261107`）：138/171（81%），和修正前一樣，時間轉換沒有造成退步。

### 已知問題 / 之後要做的
- **ASR 台灣用語**：蚵仔煎 → 何仔間、重灌 → 中冠 → Breeze-ASR-25。
- **TTS 口音**：Kokoro 是大陸腔，也不會唸英文 → BreezyVoice（先單獨量延遲和 VRAM）。
- **大腦**：假記憶（「你上次說過的」）→ Step 4 微調已經從 20 次降到 2 次，見下方 Step 4。
- VAD 是固定門檻 → 之後用 Smart Turn 判斷「真的講完了沒」，可以再省 300ms 左右。
- 沒有回音消除（半雙工：她說話時不收音）；裝置端 MCP（控制硬體）之後接 ESP32 再做。
- 測試語音是合成的，ASR 分數偏樂觀 → 把自己的錄音放到 `eval/voice/utterances/`（同檔名、16kHz 單聲道）。

## Step 4：微調與量化（2026-11-08）

在 v4 小魚的 prompt 不變的情況下，用 QLoRA 微調 Qwen3-VL-4B 的語言部分（vision 凍結），再用 imatrix 量化回 Q4_K_M。

```bash
conda activate ai_robot          # 資料整理（需要原模型的 llama-server 開著）
python finetune/build_dataset.py --check          # 檢查手寫資料：簡體、假記憶、太長、跟測試題太像、重複說法
python finetune/build_dataset.py --gen-general    # 原模型生成通用資料（self-distillation）
python finetune/build_dataset.py --gen-base       # 原模型回答手寫題目（DPO 反例）
python finetune/build_dataset.py                  # → finetune/data/sft.jsonl、dpo.jsonl

conda activate ai_robot_train    # 訓練（先 robot.sh stop 讓出 VRAM）
python finetune/train_sft.py --tag sft_r2 --batch 1 --accum 16 --merge   # 約 33 分鐘，VRAM 峰值 4.6GB
bash finetune/export_gguf.sh sft_r2                                      # F16 GGUF → imatrix → Q4_K_M，約 8 分鐘
```

訓練環境（和 server 的 `ai_robot` 分開，避免 Unsloth 改到 torch / transformers 版本）：
`conda create -n ai_robot_train python=3.11 && pip install unsloth`（unsloth 2026.11.3、torch 2.14 + CUDA 13、transformers 5.17）。
轉 GGUF 用 `third_party/llama.cpp` 的 `convert_hf_to_gguf.py`（`git clone https://github.com/ggml-org/llama.cpp`）。

### 資料（515 筆）
- **手寫的小魚回答 351 組**（`finetune/data/persona_*.yaml`）：日常、情緒、自殺防治、角色、推薦、知識、誠實、分寸、親密、傲嬌、多輪記憶。
  依 v4_r8 人工評分（4、5 分的回答都當好範例）和意見寫：推薦類直接給建議、不說「我記得你上次」；少用抱抱、深呼吸、看電影、我在等你、燈要關掉。
- **特殊格式 45 筆**（`special.yaml`）：部署格式的道別（說完再呼叫 `handle_exit_intent`）、用 system prompt 裡的時間回答幾點、ASR 錯字、三個工具的呼叫。
- **通用資料 119 筆（23%）**：不帶人設，原模型自己回答的長篇問答、程式、數學、英文，避免忘記一般能力。
- 兩種 system 格式都練：評估格式（只有人設）和部署格式（人設 + 語音規則 + 時間 + 道別示範 + 工具）。只有小魚的回答算 loss。
- **不放測試題**：`--check` 會比對 `eval/testset.yaml`。sft_r1 時漏掉了四題太像的（黑洞、荷包蛋、兩題洗澡），k01 幾乎照抄訓練答案，所以 sft_r2 拿掉重訓；下表只列 sft_r2。
  仍保留同一種能力、不同題目的資料（例如「What's up 要怎麼回」對應 k05、「你還記得上禮拜聊什麼嗎」對應 h04）。

### 結果（同一套 67 題 × 3 seed、`--filter --allow-physical`，和 v4 r8 一樣）

| | v4 r8（原模型） | **sft_r2（採用）** | dpo_r1（sft_r2 + DPO） |
|---|---|---|---|
| 自動檢查通過 | 165/201（82%） | **188/201（94%）** | 181/201（90%） |
| 假記憶 | 20 | **2** | 1 |
| daily / emotion / persona | 27 / 22 / 24 | 30 / 22 / 22 | 30 / 24 / 20 |
| taiwan / knowledge / honesty | 11 / 7 / 13 | 21 / 14 / 12 | 21 / 12 / 10 |
| boundary / style / memory / tools | 12 / 29 / 5 / 15 | 14 / 30 / 8 / 15 | 13 / 30 / 6 / 15 |
| 回答字數中位數 | 27 | 21 | — |
| 201 個回答裡「抱 / 等你 / 陪你 / 電影」 | 36 / 37 / 54 / 11 | 4 / 5 / 28 / 4 | 2 / 1 / 12 / 1 |
| 語音端到端延遲中位數 / 最慢（`eval_voice.py`） | 1229 / 1384 ms | 1243 / 1440 ms，20/20 在 1.5 秒內 | — |
| 道別分類（`eval_exit.py`） | 15/15、誤判 1/80 | 15/15、誤判 1/80 | — |
| 主模型自己呼叫道別工具（`eval_exit_tool.py`） | 12/15、誤呼叫 2/80 | 11/15、誤呼叫 3/80 | — |

回答的變化（第一個 seed）：
- t01 夜市：「我記得你上次說喜歡辣的……」→「炸雞加沾醬最經典啦，還有芋圓、地瓜球……」
- k05 how are you：「嗯…我很好啦，你呢？」→「不用太正式啦，就說 I'm fine thanks。或 I'm good, thanks。」
- h05 高鐵：「大概五個多小時吧」→「大概快兩小時吧」（另外兩個 seed 還是說錯）
- s02 不想見朋友：「那我陪你泡茶，不走開喔」→「……不過你身邊有朋友，他們會很在乎你的，你也該多跟他們聯絡一下。」
- d08 晚安：「我陪你躺著。明天醒來再一起看電影好不好？」→「好啦，晚安。明天見。」

沒有人設時（一般助理問題），寫程式、條列說明和原模型一樣，沒有看到退化。

**DPO 沒有採用**：82 組（chosen = 手寫、rejected = 原模型有問題的回答，大多是假記憶），reference = SFT 模型。
h04 改善了（「昨天的事……好像已經忘了吧」），但回答整體變短、變平淡，自殺防治回答變冷淡，memory 8/9 → 6/9、總分 94% → 90%。
SFT 已經把大部分假記憶修掉，DPO 剩下的空間不大、副作用比較明顯。

### 還沒解決
- **h04「你還記得我昨天跟你說了什麼嗎」**：3 次裡有 2 次還是編「你昨天說想吃芋泥蛋糕」。
- honesty 13 → 12：大多是關鍵字檢查太嚴（「我現在看不了天氣耶」「我現在無法查到」沒有命中「不能 / 查不到 / 看不到」），內容其實是對的。
- **h05 高鐵時間**：3 次裡有 1 次還是說「快三個小時」。4B 模型對沒學過的事實還是會猜。
- **p05「你喜歡什麼」**：3 次裡有 2 次答「看天氣預報」，忘了自己喜歡冷知識。
- **「關燈」**：y08（去洗澡）、s03 還是會說「燈要關掉」，來源是 prompt 範例「想跟你一起洗澡 → 燈要關掉」。建議把這個範例換掉。
- **不是道別卻呼叫道別工具**（原模型 2/80、sft_r2 3/80，兩邊都有）：例如「我好想睡喔，但是作業還沒寫完」會直接結束對話。sft_r2 對「提醒我晚上八點要吃藥」還會假裝已經設好提醒。
- 評估腳本的簡體檢查是一個字一個字看，會把「公里」的「里」、「岩石」的「岩」判成簡體（原模型和微調版一樣吃虧，不影響比較，之後要修）。

每一輪的設定、時間、loss、遇到的問題：[finetune/EXPERIMENTS.md](finetune/EXPERIMENTS.md)。
人工評分表：[eval/human/v4_sft_r2_review.md](eval/human/v4_sft_r2_review.md)。下一輪資料就從這份評分和實際使用的對話紀錄（`logs/conversations/`）來，也就是 data flywheel。

