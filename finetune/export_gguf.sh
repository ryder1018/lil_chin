#!/usr/bin/env bash
# Step 4：合併後的 16-bit 模型 → GGUF（f16）→ imatrix → Q4_K_M
# 用法：bash finetune/export_gguf.sh sft_r1
# 輸出：models/qwen3-vl-4b-<tag>/Qwen3VL-4B-<tag>-Q4_K_M.gguf（vision 沒動，mmproj 沿用原本的）
set -euo pipefail

TAG=$1
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/finetune/out/$TAG/merged"
DST="$ROOT/models/qwen3-vl-4b-$TAG"
F16="$DST/Qwen3VL-4B-$TAG-F16.gguf"
Q4="$DST/Qwen3VL-4B-$TAG-Q4_K_M.gguf"
IMAT="$DST/imatrix.gguf"
CALIB="$ROOT/finetune/data/calib.txt"
LLAMA="$ROOT/third_party/llama.cpp"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
mkdir -p "$DST"

conda activate ai_robot_train
[[ -f "$F16" ]] || PYTHONPATH="$LLAMA/gguf-py" python "$LLAMA/convert_hf_to_gguf.py" "$SRC" --outtype f16 --outfile "$F16"
python "$ROOT/finetune/make_calib.py" "$CALIB"

conda activate ai_robot
# imatrix：用訓練資料（人設對話 + 通用資料）當校正文字，量化時保護這些內容最常用到的權重
[[ -f "$IMAT" ]] || llama-imatrix -m "$F16" -f "$CALIB" -o "$IMAT" -ngl 20 -c 512 --chunks 200
llama-quantize --imatrix "$IMAT" "$F16" "$Q4" Q4_K_M
ln -sf "$ROOT/models/qwen3-vl-4b-instruct/mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf" "$DST/"
ls -la "$DST"
