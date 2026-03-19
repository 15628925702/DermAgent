#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-$HOME/derm_agent}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-derm-qwen}"
TRAIN_HOURS="${TRAIN_HOURS:-8}"
SAVE_DIR="${SAVE_DIR:-outputs/train_runs/server_v2_8h}"
SPLIT_JSON="${SPLIT_JSON:-outputs/splits/pad_ufes20.json}"
CONTROLLER_IN="${CONTROLLER_IN:-outputs/controller_v2.json}"
BANK_IN="${BANK_IN:-outputs/experience_bank.json}"

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

if ! curl -s "http://127.0.0.1:8000/v1/models" -H "Authorization: Bearer EMPTY" >/dev/null 2>&1; then
  echo "[error] Qwen service is not reachable on 127.0.0.1:8000"
  echo "[error] start it first with bash scripts/linux_start_qwen.sh"
  exit 1
fi

mkdir -p "$SAVE_DIR"

timeout "${TRAIN_HOURS}h" python scripts/train_server.py \
  --dataset-root data/pad_ufes_20 \
  --epochs 999999 \
  --eval-every 1 \
  --split-json "$SPLIT_JSON" \
  --save-dir "$SAVE_DIR" \
  --controller-state-in "$CONTROLLER_IN" \
  --bank-state-in "$BANK_IN"

TRAIN_EXIT_CODE=$?
if [[ "$TRAIN_EXIT_CODE" -ne 0 && "$TRAIN_EXIT_CODE" -ne 124 ]]; then
  echo "[error] training exited with code $TRAIN_EXIT_CODE"
  exit "$TRAIN_EXIT_CODE"
fi

echo "[ok] training window finished"
python scripts/run_eval_brief.py \
  --dataset-root data/pad_ufes_20 \
  --split-json "$SPLIT_JSON" \
  --split-name test \
  --controller-state-in "$SAVE_DIR/best_controller.json" \
  --bank-state-in "$SAVE_DIR/best_bank.json" \
  --freeze-learning
