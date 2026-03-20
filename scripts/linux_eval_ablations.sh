#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-$HOME/derm_agent}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-derm-qwen}"
DATASET_ROOT="${DATASET_ROOT:-data/pad_ufes_20}"
SPLIT_JSON="${SPLIT_JSON:-outputs/splits/pad_ufes20_full.json}"
SPLIT_NAME="${SPLIT_NAME:-test}"
LIMIT="${LIMIT:-40}"
SAVE_DIR="${SAVE_DIR:-outputs/train_runs/quick_iter}"
OUTPUT_PATH="${OUTPUT_PATH:-outputs/quick_compare/ablations.json}"

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

if [[ ! -f "$SAVE_DIR/best_controller.json" ]]; then
  echo "[error] checkpoint not found: $SAVE_DIR/best_controller.json"
  exit 1
fi

python scripts/run_ablations.py \
  --dataset-root "$DATASET_ROOT" \
  --limit "$LIMIT" \
  --split-json "$SPLIT_JSON" \
  --split-name "$SPLIT_NAME" \
  --controller-state-in "$SAVE_DIR/best_controller.json" \
  --bank-state-in "$SAVE_DIR/best_bank.json" \
  --output "$OUTPUT_PATH"

echo "[ok] ablations saved to $OUTPUT_PATH"
