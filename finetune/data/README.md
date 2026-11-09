# finetune/data

| 檔案 | 類型 | 內容 |
|---|---|---|
| `persona_a.yaml` | 手寫 | 小魚的示範對話：daily、emotion、self_harm、persona、style |
| `persona_b.yaml` | 手寫 | taiwan、knowledge、honesty、boundary、intimate、memory（多輪） |
| `special.yaml` | 手寫 | 需要特殊格式的：部署格式（道別工具、現在時間、ASR 錯字）、只用評估格式的、三個工具的呼叫 |
| `general_prompts.yaml` | 手寫 | 通用資料的題目（不帶人設，程式、數學、英文、長篇說明） |
| `general_gen.jsonl` | 原模型生成 | 原模型回答 `general_prompts.yaml`（`build_dataset.py --gen-general`） |
| `base_answers.jsonl` | 原模型生成 | 原模型（v4 prompt）回答手寫單輪題目，DPO 的反例來源（`--gen-base`） |
| `sft.jsonl` | 產生（不進 git） | `build_dataset.py` 組出來的訓練資料 |
| `dpo.jsonl` | 產生（不進 git） | DPO 資料對 |
| `calib.txt` | 產生（不進 git） | imatrix 校正文字（`export_gguf.sh` 會自動產生） |

原模型生成的兩個檔案有隨機性，重新生成不會一模一樣，所以進 git；其他產生的檔案用 `python finetune/build_dataset.py` 就能重建（`--seed 0`）。

新增手寫資料後先跑 `python finetune/build_dataset.py --check`。它會檢查簡體、大陸用語、假記憶、太長、重複說法，以及跟測試題太像的題目。
相似度檢查只比字面，換句話說的版本抓不到（sft_r1 就這樣漏掉黑洞、荷包蛋），所以還要人工對照 `eval/testset.yaml` 看一次。
