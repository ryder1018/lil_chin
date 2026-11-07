#!/usr/bin/env bash
# 一次管理機器人的服務
#   bash scripts/robot.sh start          # 全部啟動
#   bash scripts/robot.sh stop           # 全部停止
#   bash scripts/robot.sh status         # 看狀態 + VRAM
#   bash scripts/robot.sh restart server # 只重啟某一個（llm / server）
#
#   llm     llama-server（小晴的大腦）                       :8080
#   server  機器人後端（VAD/ASR/TTS/對話，python -m server）  :8000 WebSocket、:8003 網頁 + OTA
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$ROOT/logs/services"
mkdir -p "$LOG"
source "$HOME/miniconda3/etc/profile.d/conda.sh"

SERVICES=(llm server)

port_of() { case $1 in llm) echo 8080 ;; server) echo 8003 ;; esac; }
health_url() { case $1 in llm) echo "http://127.0.0.1:8080/health" ;; server) echo "http://127.0.0.1:8003/xiaozhi/ota/" ;; esac; }
is_up() { curl -sf -o /dev/null -m 2 "$(health_url "$1")"; }
pid_on_port() { ss -ltnp 2>/dev/null | awk -v p=":$1" '$4 ~ p"$"' | grep -oP 'pid=\K[0-9]+' | head -1; }

start_one() {
  local s=$1
  if is_up "$s"; then echo "  $s 已經在跑"; return; fi
  case $s in
    llm) (bash "$ROOT/scripts/start_llm.sh" >"$LOG/llm.log" 2>&1 &) ;;
    # HF_HUB_OFFLINE：模型都已經下載好，不要每次啟動都連網檢查（完全離線）
    server) (conda activate ai_robot && cd "$ROOT" && HF_HUB_OFFLINE=1 python -m server >"$LOG/server.log" 2>&1 &) ;;
  esac
  for _ in $(seq 1 180); do
    is_up "$s" && { echo "  ✅ $s (:$(port_of "$s"))"; return; }
    sleep 1
  done
  echo "  ❌ $s 啟動逾時，看 $LOG/$s.log"
}

stop_one() {
  local pid
  pid=$(pid_on_port "$(port_of "$1")")
  if [[ -n "$pid" ]]; then kill "$pid" && echo "  停止 $1 (pid $pid)"; else echo "  $1 沒在跑"; fi
}

status() {
  for s in "${SERVICES[@]}"; do
    if is_up "$s"; then echo "  ✅ $s (:$(port_of "$s"))"; else echo "  ⬜ $s"; fi
  done
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | sed 's/^/  VRAM /'
}

cmd=${1:-status}
targets=("${@:2}")
[[ ${#targets[@]} -eq 0 ]] && targets=("${SERVICES[@]}")

case $cmd in
  start) for s in "${targets[@]}"; do start_one "$s"; done
         echo; echo "打開瀏覽器（Windows 也可以）：http://localhost:8003/" ;;
  stop) for s in "${targets[@]}"; do stop_one "$s"; done ;;
  restart) for s in "${targets[@]}"; do stop_one "$s"; done; sleep 2
           for s in "${targets[@]}"; do start_one "$s"; done ;;
  status) status ;;
  *) echo "用法：$0 start|stop|restart|status [llm|server]"; exit 1 ;;
esac
