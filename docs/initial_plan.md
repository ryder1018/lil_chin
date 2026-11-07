# 🤖 離線台灣話桌上機器人：專案計畫總覽

## 0. 專案目標
做一台**完全離線、會說台灣話、之後能看見並主動互動**的桌上型聊天機器人。

- 硬體和韌體沿用成熟的開源專案 xiaozhi。
- **差異化放在大腦**：自己微調、量化本地模型，並做台灣在地化。

---

## 1. 系統架構

### 第一版（不含視覺）
```
┌─────────── 機器人本體 ───────────┐        Wi-Fi         ┌──────────── 筆電（WSL）────────────┐
│ ESP32-S3（xiaozhi 韌體）          │   WebSocket + Opus   │ xiaozhi-esp32-server（Python）       │
│  ├ INMP441 麥克風 → 收音          │ ───────────────────▶ │  VAD → ASR → LLM 大腦 → TTS         │
│  ├ MAX98357A + 喇叭 → 播放        │ ◀─────────────────── │           ↑                         │
│  ├ OLED → 表情                    │   音訊 / 表情 / 指令  │  llama-server（自己的量化模型）      │
│  └ 按鈕 / ESP-SR 喚醒詞           │                      │  RTX 4060 8GB                       │
└──────────────────────────────────┘                      └─────────────────────────────────────┘
```

### 最終版
```
鏡頭 + 麥克風 + 喇叭 + 伺服馬達
        ⇅
Jetson（大腦，所有模型在本機）
   ├ System 1（快）：VAD、喚醒詞、人臉偵測與追蹤、判斷對方講完了沒
   └ System 2（慢）：VLM 負責理解、對話、決定呼叫哪個工具
        ⇅ USB serial
ESP32（小腦）：伺服馬達、LED 眼睛
```

---

## 2. 硬體

### 2.1 第一版：麵包板（約 NT$900–1,300）
| 品項 | 規格 / 注意事項 | 約略價格 |
|---|---|---|
| ESP32-S3-DevKitC-1 | ⚠️ 一定要 **N16R8**（16MB Flash + 8MB PSRAM） | NT$300–450 |
| INMP441 I2S 麥克風 | 建議多買 1 個備用 | NT$80–150 |
| MAX98357A I2S 功放 | 建議多買 1 個備用 | NT$100–150 |
| 喇叭 | 4Ω 或 8Ω、2–3W、直徑 28–40mm | NT$50–100 |
| 0.96 吋 OLED（SSD1306）| 選配，顯示表情 | NT$100 |
| 按鈕 ×2、830 孔麵包板 ×2、杜邦線 | | NT$200 |
| USB-C **傳輸**線 | ⚠️ 只能充電的線沒辦法燒錄 | NT$100 |
| 工具：烙鐵、三用電表 | 沒有的話再買 | NT$600–900 |

接線照 xiaozhi 原始碼的 `main/boards/bread-compact-wifi/config.h`。

**已經有、不用買的：** 筆電（RTX 4060）、手機熱點或便宜路由器（避開校園 Wi-Fi 的 AP isolation）。

### 2.2 之後依階段加購
| 階段 | 品項 | 約略價格 |
|---|---|---|
| 成品化 | 洞洞板（或自己畫 PCB 給 JLCPCB 打樣）、M2/M2.5 螺絲和銅柱、3D 列印外殼 | NT$300–800 |
| 無線 | 鋰電池 + TP4056 充電模組 | NT$200 |
| 視覺 | XIAO ESP32S3 Sense 或 ESP32-S3-EYE | NT$500–1,500 |
| 轉頭 | MG90S ×2 + PCA9685 + 5V 獨立電源 | NT$600 |
| 回音消除 / 全雙工 | ReSpeaker，或改用有 AEC 的整合板（S3-BOX-3） | NT$1,000–2,500 |
| 邊緣化 | Jetson Orin Nano Super 8GB + NVMe SSD | NT$9,500–11,500 |

---

## 3. 軟體

### 3.1 裝置端韌體：沿用 xiaozhi-esp32
| 項目 | 技術 |
|---|---|
| 框架 / 語言 | ESP-IDF 6.x、C++、FreeRTOS |
| 音訊 | Opus，16kHz 單聲道，每個 frame 60ms |
| 喚醒詞 | ESP-SR（WakeNet），可以自訂 |
| 通訊 | WebSocket（hello 握手、`listen`、`stt`、`tts`、`llm` 情緒、`abort`） |
| 控制硬體 | 裝置端 MCP（JSON-RPC 2.0），把 LED、伺服馬達、GPIO 包成工具 |
| 開發工具 | VS Code + ESP-IDF 外掛 |

### 3.2 後端：xiaozhi-esp32-server，各模組逐步換成本地模型
| 模組 | 第一版 | 進階版（差異化） |
|---|---|---|
| VAD | SileroVAD | + Smart Turn（判斷對方是不是真的講完了） |
| ASR | FunASR / **SenseVoice-Small**（本地） | **Breeze-ASR-25**（台灣口音、中英夾雜）+ LoRA 微調；嘗試台語 |
| 情緒 | — | SenseVoice 的情緒標籤，從聲音聽出情緒 |
| **LLM 大腦** | 現成的 **Qwen3-VL-4B** 或 **Breeze2-3B**，Q4_K_M | 自己 QLoRA 微調 + DPO，用 imatrix 量化 |
| 看圖 | 暫時不用 | 自己寫一個 VLLM provider，接本地 VLM |
| TTS | 本地 GPT-SoVITS / FishSpeech / Piper | **BreezyVoice**（台灣腔）、CosyVoice2 + 情緒標籤 |
| 記憶 | mem_local_short | bge-m3 / Qwen3-Embedding + sqlite-vec，做跟人綁定的長期記憶 |
| 工具 | function_call | 用 GBNF 約束 JSON 輸出 + 自訂 MCP 工具 |

第一版雖然還不看圖，但**一開始就用 VLM 當大腦**（VLM 本身也能純文字對話）。這樣之後加上視覺時，微調好的模型可以直接沿用，不用重練。

### 3.3 推論與部署
| 項目 | 技術 |
|---|---|
| 推論引擎 | **llama.cpp `llama-server`**（GGUF 格式，提供 OpenAI 相容 API） |
| 加速 | 串流輸出、prefix / KV cache 重用、推測解碼、KV cache 量化 |
| 環境 | WSL2：在 `.wslconfig` 設 `networkingMode=mirrored`、把記憶體調大，並開放 Windows 防火牆的對應 port |
| 打包 | Docker、systemd |
| 模型大小上限 | 全部加起來 **6GB 以內**，對齊之後 Jetson 的記憶體 |
| 最終版 | Jetson：JetPack 6.2、jetson-containers；進階可用 TensorRT-LLM |

### 3.4 模型訓練與量化
| 步驟 | 工具 / 技術 |
|---|---|
| 資料 | 合成對話（大模型生成）+ 人工審核；混入 20–30% 通用資料，避免模型忘記原本的能力 |
| SFT | **Unsloth** 或 LLaMA-Factory。QLoRA：r=16、lr 2e-4、1–3 epochs，凍結 vision 部分 |
| 對齊 | DPO / ORPO：教模型講話簡短口語，看不清楚時誠實說，不要亂編 |
| 合併與轉檔 | 合併 LoRA → `convert_hf_to_gguf.py`（語言模型和 `--mmproj` 視覺投影層分開轉） |
| 量化 | `llama-imatrix`（用自己的繁中資料校準）→ `llama-quantize` Q4_K_M / Q5_K_M；mmproj 保持 FP16 或 Q8 |
| 算力 | 2B–4B 用筆電的 4060 訓練；7B 以上租雲端 GPU |

---

## 4. 開發流程

**原則：** 先在電腦上把大腦（聽、想、說）做好，不用任何硬體；用電腦的麥克風和喇叭模擬機器人。順序是「跑起來 → 打分數 → 實際用 → 才微調」，沒有基準分數前不做微調。

### 階段 A：電腦上的大腦（現在要做）

| 步驟 | 內容 | 完成標準 | 預估時間 |
|---|---|---|---|
| **Step 1 跑起現成模型** | 用 Docker 架 llama-server（WSL 沒有 nvcc，用官方 `server-cuda` image）；下載 **Qwen3-VL-4B-Instruct** Q4_K_M（用 Instruct 不用 Thinking，避免延遲）；第一版先不載 `--mmproj` 省 VRAM | `curl` 能拿到回答；記下 TTFT、tok/s、VRAM | 1–2 天 |
| **Step 2 system prompt + 測試集** | 決定角色（先做中性的台灣口語助理，之後再往女友/助理方向調）；寫 system prompt；建 50 題測試集（口語簡短、台灣用語、角色一致、誠實、工具呼叫、多輪記憶）；寫自動評分腳本（長度、簡體字（OpenCC）、markdown 符號、TTFT） | 有一份**基準分數**，之後每次改動都用同一套題目比 | 2–3 天 | in-context learning
| **Step 3 加上聽和說** | 在 WSL 架 xiaozhi-esp32-server，LLM 指向自己的 llama-server；ASR 用 SenseVoice-Small，TTS 先挑輕量快速的；用 Windows 瀏覽器的測試頁（或 py-xiaozhi）當假裝置，避開 WSL 音訊問題；LLM 串流輸出、一句一句送 TTS；**所有對話存 log** | 用電腦麥克風跟它講話、喇叭有回應；端到端延遲 < 1.5 秒；VRAM 合計 ≤ 6GB | 3–5 天 |
| **Step 4 微調與量化** | 從 log 挑出答不好的對話，人工修正 + 合成資料（混 20–30% 通用資料）→ Unsloth QLoRA → DPO → imatrix 量化 → 跑 Step 2 測試集 → 換模型檔重新部署 | 微調版在測試集上明顯勝過原始模型 | 2–4 週 |

做完 Step 4 後持續循環：**部署 → 實際使用 → 記錄答不好的對話 → 整理成資料重新訓練 → 再部署（data flywheel）**。

### 之後的階段（先列大概，到時再細拆）

| 階段 | 大概內容 |
|---|---|
| **差異化功能** | 換成台灣腔的 ASR / TTS（Breeze-ASR-25、BreezyVoice）、聲音情緒、長期記憶，每項都要有前後對比數據 |
| **硬體 v1** | 買零件、接麵包板、燒 xiaozhi 韌體，把連線位址指到電腦上的後端（後端不用改） |
| **視覺與表情** | 加鏡頭、打開 mmproj 讓大腦看圖、人臉偵測主動打招呼；OLED 表情 |
| **成品化** | 洞洞板、3D 列印外殼 |
| **動作與邊緣化** | 伺服馬達轉頭追人；整個大腦搬到 Jetson |

---

## 5. 評估指標
| 面向 | 指標 |
|---|---|
| 延遲 | 端到端延遲（說完話到開始出聲，目標 < 1.5 秒）、TTFT、tok/s |
| ASR | CER（字錯誤率） |
| 大腦 | 自建的 200 題測試集（人工評分或 LLM-as-judge）、TMMLU+ |
| 量化影響 | 同一套測試集上 FP16、Q8、Q5、Q4 的分數差異 |
| 視覺（P6 之後） | 自建看圖問答、POPE（測視覺幻覺） |
| TTS | MOS（主觀聽感評分） |
| 資源 | VRAM / RAM 峰值 |

---

## 6. 常見坑檢查清單
- [ ] ESP32-S3 買的是 **N16R8**
- [ ] USB-C 線能傳資料
- [ ] 韌體用 **ESP-IDF 6.x**（5.x 已經不支援）
- [ ] 不要用校園 Wi-Fi，改用手機熱點
- [ ] WSL 有設 mirrored networking，Windows 防火牆有開 port
- [ ] 模型總大小在 6GB 以內，之後 Jetson 才放得下
- [ ] 沒有 AEC 的話用半雙工，或第一版先戴耳機測試
- [ ] 伺服馬達接獨立電源，並且共地
- [ ] 用 API 生成訓練資料前先看使用條款

---

## 7. 預算總表
| 階段 | 累計 |
|---|---|
| P1–P5（麵包板 + 外殼） | 約 NT$1,500–3,000 |
| + P6 視覺 + 轉頭 | 約 NT$3,000–5,000 |
| + P7 Jetson | 約 NT$13,000–17,000 |

---
