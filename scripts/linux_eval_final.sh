#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-$HOME/derm_agent}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-derm-qwen}"
SPLIT_JSON="${SPLIT_JSON:-outputs/splits/pad_ufes20_full.json}"
SAVE_DIR="${SAVE_DIR:-outputs/train_runs/final_v3_8h}"
DATASET_ROOT="${DATASET_ROOT:-data/pad_ufes_20}"

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

if [[ ! -f "$SPLIT_JSON" ]]; then
  echo "[error] split file not found: $SPLIT_JSON"
  echo "[error] build it first with bash scripts/linux_build_full_split.sh"
  exit 1
fi

if [[ ! -f "$SAVE_DIR/best_controller.json" ]]; then
  echo "[error] checkpoint not found: $SAVE_DIR/best_controller.json"
  echo "[error] run training first with bash scripts/linux_train_final.sh"
  exit 1
fi

python scripts/run_eval_brief.py \
  --dataset-root "$DATASET_ROOT" \
  --limit 999999 \
  --split-json "$SPLIT_JSON" \
  --split-name test \
  --controller-state-in "$SAVE_DIR/best_controller.json" \
  --bank-state-in "$SAVE_DIR/best_bank.json" \
  --freeze-learning
