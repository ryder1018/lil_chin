#!/usr/bin/env bash
# 啟動機器人大腦：llama-server（OpenAI 相容 API）
# 用法：bash scripts/start_llm.sh            （純文字，第一版）
#       VISION=1 bash scripts/start_llm.sh   （載入 mmproj，可以看圖，P6 之後）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-$ROOT/models/qwen3-vl-4b-instruct}"
MODEL="${MODEL:-$MODEL_DIR/Qwen3VL-4B-Instruct-Q4_K_M.gguf}"
MMPROJ="${MMPROJ:-$MODEL_DIR/mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
CTX="${CTX:-8192}"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate ai_robot

ARGS=(
  -m "$MODEL"
  --host "$HOST" --port "$PORT"
  -ngl 99                  # 全部 layer 放 GPU
  -c "$CTX"                # context 長度
  -np 1                    # 只有一個使用者（機器人），省 KV cache
  -fa on                   # flash attention
  -ctk q8_0 -ctv q8_0      # KV cache 量化，省 VRAM
  # 註：--cache-reuse 對 Qwen3-VL 不支援（server 會自動停用）；一般 prefix cache 仍有效
  --jinja                  # 用模型內建 chat template（工具呼叫需要）
  # Qwen3-VL Instruct 官方建議的取樣參數
  --temp 0.7 --top-p 0.8 --top-k 20 --presence-penalty 1.5
)

if [[ "${VISION:-0}" == "1" ]]; then
  ARGS+=(--mmproj "$MMPROJ")
else
  ARGS+=(--no-mmproj)
fi

exec llama-server "${ARGS[@]}" "$@"
