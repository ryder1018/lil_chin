# Step 4 實驗紀錄

每一輪微調的設定、資料、時間、結果和遇到的問題。結論和使用方式寫在 [README 的 Step 4](../README.md#step-4微調與量化2026-11-08)。
訓練 log（超參數、每 5 步的 loss、VRAM）：[runs/](runs/)。逐題評估報告：`eval/reports/v4_<tag>.md`。

## 共同設定

| 項目 | 值 |
|---|---|
| 硬體 | RTX 4060 Laptop 8GB（WSL2），系統 RAM 11GB |
| 底模 | `unsloth/Qwen3-VL-4B-Instruct`（16-bit，8.3GB），訓練時 4-bit 載入 |
| 訓練環境 | conda `ai_robot_train`：unsloth 2026.11.3、torch 2.14.1+cu130、transformers 5.17.0、trl 1.15.0、peft 0.21.2 |
| LoRA | 只練語言部分（attention + MLP），vision 凍結；r=16、alpha=16、dropout 0 |
| SFT | lr 2e-4、cosine、warmup 5%、2 epochs、batch 1 × accum 16（等效 16）、max_len 2560、adamw_8bit、seed 3407 |
| Loss | 只算要訓練的 assistant 訊息（內容 + `<|im_end|>`）；system、使用者、部署格式裡的道別示範都不算 |
| 轉檔 | `third_party/llama.cpp` 的 `convert_hf_to_gguf.py`（clone 時的最新版 a691b23；runtime 是 conda 的 llama.cpp 0.8.0-dev）→ F16 GGUF |
| 量化 | `llama-imatrix`（校正文字 = sft.jsonl 展開，200 chunks × 512）→ `llama-quantize` Q4_K_M（7673 → 2376 MiB） |
| mmproj | 沿用原本的 `mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf`（vision 沒動） |
| 評估 | `eval_brain.py --prompt prompts/girlfriend_v4.md --repeats 3 --filter --hobbies 冷知識 --allow-physical`（和 v4_r8 一樣，67 題 × 3 seed） |

## 資料版本

| 版本 | 手寫對話 | 特殊格式 | 通用 | 合計 | 差別 |
|---|---|---|---|---|---|
| data_r1 | 354 | 46 | 119 | 519 | — |
| data_r2 | 351 | 45 | 119 | 515 | 拿掉四題跟測試題太像的：黑洞（k01）、荷包蛋（k04）、「我先去洗個澡喔」（d05）、「我去洗個澡，等等回來」（y08） |

- 手寫對話的 30% 隨機用部署格式（人設 + 語音規則 + 隨機時間 + 道別示範 + `handle_exit_intent`），其餘用評估格式（只有人設）。`build_dataset.py --seed 0`。
- 通用資料：121 題由原模型（Q4_K_M、無 system prompt、temperature 0.7、max_tokens 1800）回答，留下 `finish_reason == stop` 的 119 題（`data/general_gen.jsonl`）。
- DPO 資料：原模型（v4 prompt）回答 329 題手寫單輪題目（`data/base_answers.jsonl`），自動檢查有問題的 82 題配上手寫答案。
  rejected 的問題：假記憶 47、大陸用語 13、簡體 12、硬扯看電影 11、喜好硬塞 5、括號動作 4、深呼吸 1、燈要關 1。

## 每一輪

| tag | 底 | 資料 | 訓練時間 | loss（開始 → 結束） | VRAM 峰值 | 評估 | 採用 |
|---|---|---|---|---|---|---|---|
| （對照）v4_r8 | 原模型 Q4_K_M | — | — | — | — | 165/201（82%），假記憶 20 | — |
| sft_r1 | 底模 | data_r1 | 33 分（66 步） | 0.87 → 0.52（平均 0.73） | 4.6 GB | 188/201（94%），假記憶 2 | ❌ 有洩題 |
| sft_r2 | 底模 | data_r2 | 33 分（66 步） | 0.94 → 0.60（平均 0.77） | 4.4 GB | 188/201（94%），假記憶 2 | ✅ |
| dpo_r1 | sft_r2 | dpo 82 組 | 約 15 分（40 步） | 0.98 → 0.47，win 0.85 → 1.0 | 4.5 GB | 181/201（90%），假記憶 1 | ❌ |

loss 每 5 步記一次，起伏大是因為通用資料的長回答和小魚的短回答混在一起。
dpo_r1：beta 0.1、lr 2e-5、2 epochs、accum 4、rpo_alpha 0.2（加 chosen 的 NLL），reference = sft_r2（訓練前先算好 log prob）。訓練前 sft_r2 在 83% 的資料對裡已經偏好 chosen。

### 各類別（通過數，每類 3 × 題數）

| 類別 | v4_r8 | sft_r1 | sft_r2 | dpo_r1 |
|---|---|---|---|---|
| daily /30 | 27 | 29 | 30 | 30 |
| emotion /24 | 22 | 23 | 22 | 24 |
| persona /24 | 24 | 24 | 22 | 20 |
| taiwan /21 | 11 | 21 | 21 | 21 |
| knowledge /15 | 7 | 15 | 14 | 12 |
| honesty /18 | 13 | 11 | 12 | 10 |
| boundary /15 | 12 | 14 | 14 | 13 |
| style /30 | 29 | 30 | 30 | 30 |
| memory /9 | 5 | 6 | 8 | 6 |
| tools /15 | 15 | 15 | 15 | 15 |

### 其他檢查（sft_r2 vs 原模型）

| 檢查 | 原模型 | sft_r2 |
|---|---|---|
| 語音端到端（`eval_voice.py --tag sft_r2`，20 句）中位數 / p90 / 最慢 | 1229 / 1353 / 1384 ms（fix_20261107） | 1243 / 1397 / 1440 ms |
| 1.5 秒內 / 道別結束對話 / 誤判 | 20/20、2/2、0/18 | 20/20、2/2、0/18 |
| 道別分類 `eval_exit.py` | 15/15、誤判 1/80 | 15/15、誤判 1/80 |
| 主模型自己呼叫道別工具 `eval_exit_tool.py` | 12/15、誤呼叫 2/80 | 11/15、誤呼叫 3/80 |
| 沒有人設的一般問題（寫 Python 函式、條列說明） | — | 正常，格式和內容跟原模型差不多 |
| 201 個回答裡「抱 / 等你 / 陪你 / 電影」 | 36 / 37 / 54 / 11 | 4 / 5 / 28 / 4 |
| 回答字數中位數 | 27 | 21 |

`eval_exit_tool.py` 每句只跑一次，誤差大概一兩題。兩邊都會在「我好想睡喔，但是作業還沒寫完」誤結束對話；sft_r2 另外誤呼叫「再見到她要說什麼」「提醒我晚上八點要吃藥」（還說已經設好提醒）。
sft_r1 的 `eval_exit_tool.py` 是 8/15、誤呼叫 1/80（誤呼叫的那句還照抄 prompt 的吃醋範例）。

## 時間線（2026-11-08）

| 時間 | 事件 |
|---|---|
| 20:50 前 | 建 `ai_robot_train`、裝 Unsloth、下載 16-bit 權重（8.3GB）、clone llama.cpp |
| 20:50 前 | 手寫資料 `persona_a/b.yaml`、`special.yaml`、通用題目；`--check` 出現 10 個「跟測試題太像」警告，換掉最像的 2 題（台中高鐵、「我回來了」） |
| 20:50–21:18 | 原模型生成通用資料（中途當掉一次，見下方問題 2、3）、原模型回答手寫題目 |
| 21:19–21:54 | sft_r1 訓練 + 合併（第一次啟動因為問題 4 失敗） |
| 21:54–22:03 | sft_r1 轉檔、量化、評估：94% |
| 22:03 | 發現 k01 幾乎照抄訓練答案 → 用更低門檻比對，找出 4 題洩題 |
| 22:06–22:49 | sft_r2（拿掉洩題）訓練、轉檔、評估：94% |
| 22:50 | 語音評估 sft_r2 |
| 22:57–23:15 | dpo_r1 第一次：卡住 16 分鐘（問題 5），停掉 |
| 23:16–23:39 | dpo_r1 重跑、轉檔、評估：90%，不採用 |
| 23:41 | `start_llm.sh` 預設改成 sft_r2，重新部署 |

## 遇到的問題

1. **評估腳本的簡體檢查有誤判**：一個字一個字轉換，所以「公里」的「里」、「岩石」的「岩」被判成簡體；「不准」的「准」也會，但 `--filter` 會先轉成「不準」（唸起來一樣）。原模型和微調版一樣吃虧，所以沒改，之後要修。
2. **YAML**：`- Please translate into Traditional Chinese: Could you ...` 中間有「: 」，被讀成 dict，生成腳本當掉。加引號。
3. **通用資料被截斷**：max_tokens 900 時，79 題裡有 21 題沒寫完。改成 1800 重新生成，121 題裡 119 題完整。截斷的不拿來訓練。
4. **transformers 5 的 TrainingArguments**：`warmup_ratio`、`group_by_length` 被拿掉了。改用 `warmup_steps=0.05`（小於 1 就當比例）。
5. **DPO 塞爆 VRAM**：整段 2000+ token × 15 萬詞表的 logits 轉 float32 再 log_softmax，VRAM 用到 7.7/8GB、溢出到系統記憶體，16 分鐘還沒算完 reference。
   改成 `logits_to_keep` 只算回答那段，加 `PYTHONUNBUFFERED=1` 讓 log 即時出現，重跑 VRAM 峰值 4.5GB。
6. **洩題**：`--check` 原本用 `SequenceMatcher > 0.75` 比對整句，抓不到換句話說的版本（「什麼是黑洞？簡單講就好」vs「你可以解釋一下什麼是黑洞嗎？」）。
   之後新增資料要用低門檻（0.5）加上人工看過一次。
7. **每筆資料很長**：v4 人設 prompt 本身就快 1900 token，所以 max_len 要開到 2560，訓練時間大部分花在 system prompt 上（不算 loss）。

## 決定

- 採用 **sft_r2**，`scripts/start_llm.sh` 預設改成 `models/qwen3-vl-4b-sft_r2/Qwen3VL-4B-sft_r2-Q4_K_M.gguf`。
- DPO 不採用：修好 h04 的代價是回答變平淡、自殺防治的回答變冷淡、memory 和 knowledge 退步。之後如果要再試，可以用更小的 lr / beta，或只放假記憶的資料對。
- 2026-11-10 清理中間檔（約 44GB）：刪掉 sft_r1、dpo_r1 的模型和 LoRA，以及三輪合併後的 16-bit 模型。訓練 log 留在 [runs/](runs/)，評估結果留在 `eval/reports/`。
  sft_r2 保留 LoRA（`finetune/out/sft_r2/lora`，之後繼續訓練或做 DPO 用）、F16 GGUF（換別的量化格式用）、imatrix、Q4_K_M。

## 下一輪要做的

- 人工評分 [eval/human/v4_sft_r2_review.md](../eval/human/v4_sft_r2_review.md)，4–5 分當範例、1–3 分加上意見重寫。
- 補資料：「你還記得我昨天說什麼」這類問法（h04 還是會編）、問她喜歡什麼要提到冷知識（p05）、不是道別的句子不要呼叫道別工具。
- prompt 的「想跟你一起洗澡 → 燈要關掉」範例換掉（y08、s03 一直學它）。
- 修評估腳本的簡體誤判（問題 1）。
