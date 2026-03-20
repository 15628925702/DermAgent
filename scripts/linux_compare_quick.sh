#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-$HOME/derm_agent}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-derm-qwen}"
DATASET_ROOT="${DATASET_ROOT:-data/pad_ufes_20}"
SPLIT_JSON="${SPLIT_JSON:-outputs/splits/pad_ufes20_full.json}"
SPLIT_NAME="${SPLIT_NAME:-test}"
SPLIT_LIMIT="${SPLIT_LIMIT:-40}"
PRE_LIMIT="${PRE_LIMIT:-}"
SAVE_DIR="${SAVE_DIR:-outputs/train_runs/final_v3_8h}"
OUTPUT_PATH="${OUTPUT_PATH:-outputs/quick_compare/latest.json}"

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
  echo "[error] run training first."
  exit 1
fi

PRE_LIMIT_ARGS=()
if [[ -n "$PRE_LIMIT" ]]; then
  PRE_LIMIT_ARGS=(--pre-limit "$PRE_LIMIT")
fi

python scripts/run_compare_quick.py \
  --dataset-root "$DATASET_ROOT" \
  "${PRE_LIMIT_ARGS[@]}" \
  --split-json "$SPLIT_JSON" \
  --split-name "$SPLIT_NAME" \
  --split-limit "$SPLIT_LIMIT" \
  --controller-state-in "$SAVE_DIR/best_controller.json" \
  --bank-state-in "$SAVE_DIR/best_bank.json" \
  --output "$OUTPUT_PATH"

echo "[ok] quick compare saved to $OUTPUT_PATH"
