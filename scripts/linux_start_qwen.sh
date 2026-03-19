#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-$HOME/derm_agent}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-derm-qwen}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-VL-7B-Instruct}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
API_KEY="${API_KEY:-EMPTY}"
LOG_FILE="${LOG_FILE:-$PROJECT_DIR/qwen_vllm.log}"

cd "$PROJECT_DIR"

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  # shellcheck disable=SC1090
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV_NAME"
else
  echo "[error] conda not found."
  exit 1
fi

SERVICE_URL="http://$HOST:$PORT/v1/models"

if curl -s "$SERVICE_URL" -H "Authorization: Bearer $API_KEY" >/tmp/qwen_models.json 2>/dev/null; then
  echo "[ok] Qwen service is already ready."
  cat /tmp/qwen_models.json
  exit 0
fi

EXISTING_PIDS="$(pgrep -f "vllm serve .*${PORT}" || true)"
if [[ -n "$EXISTING_PIDS" ]]; then
  echo "[warn] found stale vLLM process(es): $EXISTING_PIDS"
  pkill -f "vllm serve .*${PORT}" || true
  sleep 3
fi

if pgrep -f "vllm serve .*${PORT}" >/dev/null 2>&1; then
  echo "[error] stale vLLM process is still alive after kill attempt."
  echo "[error] inspect with: ps -fp $(pgrep -f 'vllm serve .*${PORT}' | tr '\n' ' ')"
  exit 1
else
  nohup vllm serve "$MODEL_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --api-key "$API_KEY" \
    --trust-remote-code \
    --max-model-len 4096 \
    --limit-mm-per-prompt.image 1 \
    > "$LOG_FILE" 2>&1 &
fi

echo "[info] waiting for Qwen service on $SERVICE_URL"
for i in $(seq 1 60); do
  if curl -s "$SERVICE_URL" -H "Authorization: Bearer $API_KEY" >/tmp/qwen_models.json 2>/dev/null; then
    echo "[ok] Qwen service is ready."
    cat /tmp/qwen_models.json
    exit 0
  fi
  sleep 5
done

echo "[error] Qwen service did not become ready within 5 minutes."
echo "[error] check log: $LOG_FILE"
tail -n 50 "$LOG_FILE" || true
exit 1
