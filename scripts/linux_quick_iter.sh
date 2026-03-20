#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-$HOME/derm_agent}"
TRAIN_HOURS="${TRAIN_HOURS:-0.2}"
EPOCHS="${EPOCHS:-1}"
SAVE_DIR="${SAVE_DIR:-outputs/train_runs/quick_iter}"
BASE_RUN_DIR="${BASE_RUN_DIR:-outputs/train_runs/final_v3_8h}"
COMPARE_OUTPUT="${COMPARE_OUTPUT:-outputs/quick_compare/quick_iter.json}"
SPLIT_LIMIT="${SPLIT_LIMIT:-40}"
SPLIT_NAME="${SPLIT_NAME:-test}"
SPLIT_JSON="${SPLIT_JSON:-outputs/splits/pad_ufes20_full.json}"
DATASET_ROOT="${DATASET_ROOT:-data/pad_ufes_20}"
PRE_LIMIT="${PRE_LIMIT:-}"

cd "$PROJECT_DIR"

echo "[step] start local Qwen service"
bash scripts/linux_start_qwen.sh "$PROJECT_DIR"

echo "[step] quick training"
TRAIN_HOURS="$TRAIN_HOURS" \
EPOCHS="$EPOCHS" \
SAVE_DIR="$SAVE_DIR" \
BASE_RUN_DIR="$BASE_RUN_DIR" \
SPLIT_JSON="$SPLIT_JSON" \
bash scripts/linux_train_quick.sh "$PROJECT_DIR"

echo "[step] quick compare"
SAVE_DIR="$SAVE_DIR" \
OUTPUT_PATH="$COMPARE_OUTPUT" \
SPLIT_JSON="$SPLIT_JSON" \
SPLIT_NAME="$SPLIT_NAME" \
SPLIT_LIMIT="$SPLIT_LIMIT" \
DATASET_ROOT="$DATASET_ROOT" \
PRE_LIMIT="$PRE_LIMIT" \
bash scripts/linux_compare_quick.sh "$PROJECT_DIR"

echo "[step] summarize verdict"
python scripts/summarize_quick_compare.py --input "$COMPARE_OUTPUT"
