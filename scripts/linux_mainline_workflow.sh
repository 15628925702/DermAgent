#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-$HOME/derm_agent}"
STAGE="${STAGE:-smoke}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-derm-qwen}"
DATASET_ROOT="${DATASET_ROOT:-data/pad_ufes_20}"
FULL_SPLIT_JSON="${FULL_SPLIT_JSON:-outputs/splits/pad_ufes20_full.json}"
SPLIT_NAME="${SPLIT_NAME:-test}"
SMOKE_LIMIT="${SMOKE_LIMIT:-80}"
SMOKE_EPOCHS="${SMOKE_EPOCHS:-1}"
FULL_EPOCHS="${FULL_EPOCHS:-7}"
RUN_NAME="${RUN_NAME:-}"
PROMOTE_DIR="${PROMOTE_DIR:-outputs/train_runs/mainline_current}"
DELETE_REJECTED="${DELETE_REJECTED:-0}"
MIN_TOP1_GAIN="${MIN_TOP1_GAIN:-0.0}"
MIN_TOP3_GAIN="${MIN_TOP3_GAIN:-0.0}"
MIN_MALIGNANT_RECALL_GAIN="${MIN_MALIGNANT_RECALL_GAIN:-0.0}"
MIN_CONFUSION_GAIN="${MIN_CONFUSION_GAIN:-0.0}"
START_QWEN="${START_QWEN:-1}"

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

if [[ -d "$PROMOTE_DIR" && -f "$PROMOTE_DIR/best_controller.json" && -f "$PROMOTE_DIR/best_bank.json" ]]; then
  BASELINE_RUN_DIR="$PROMOTE_DIR"
else
  BASELINE_RUN_DIR="${BASELINE_RUN_DIR:-outputs/train_runs/final_v3_8h}"
fi

if [[ -z "$RUN_NAME" ]]; then
  RUN_NAME="mainline_${STAGE}_$(date +%Y%m%d_%H%M%S)"
fi

SAVE_DIR="outputs/train_runs/$RUN_NAME"

run_review() {
  DELETE_REJECTED="$DELETE_REJECTED" \
  PROMOTE_DIR="$PROMOTE_DIR" \
  BASELINE_RUN_DIR="$BASELINE_RUN_DIR" \
  CANDIDATE_RUN_DIR="$SAVE_DIR" \
  DATASET_ROOT="$DATASET_ROOT" \
  SPLIT_JSON="$FULL_SPLIT_JSON" \
  SPLIT_NAME="$SPLIT_NAME" \
  MIN_TOP1_GAIN="$MIN_TOP1_GAIN" \
  MIN_TOP3_GAIN="$MIN_TOP3_GAIN" \
  MIN_MALIGNANT_RECALL_GAIN="$MIN_MALIGNANT_RECALL_GAIN" \
  MIN_CONFUSION_GAIN="$MIN_CONFUSION_GAIN" \
  bash scripts/linux_review_run.sh "$PROJECT_DIR"
}

start_qwen_if_needed() {
  if [[ "$START_QWEN" != "1" ]]; then
    echo "[info] skip Qwen startup because START_QWEN=$START_QWEN"
    return
  fi
  echo "[stage] ensure local Qwen service"
  CONDA_ENV_NAME="$CONDA_ENV_NAME" bash scripts/linux_start_qwen.sh "$PROJECT_DIR"
}

case "$STAGE" in
  smoke)
    start_qwen_if_needed
    echo "[stage] smoke train"
    python scripts/train_server.py \
      --dataset-root "$DATASET_ROOT" \
      --limit "$SMOKE_LIMIT" \
      --epochs "$SMOKE_EPOCHS" \
      --eval-every 1 \
      --save-dir "$SAVE_DIR"
    echo "[stage] smoke review"
    run_review
    ;;
  overnight)
    start_qwen_if_needed
    echo "[stage] overnight train"
    python scripts/train_server.py \
      --dataset-root "$DATASET_ROOT" \
      --epochs "$FULL_EPOCHS" \
      --eval-every 1 \
      --split-json "$FULL_SPLIT_JSON" \
      --save-dir "$SAVE_DIR"
    echo "[stage] overnight review"
    run_review
    ;;
  review)
    if [[ -n "${CANDIDATE_RUN_DIR:-}" ]]; then
      SAVE_DIR="$CANDIDATE_RUN_DIR"
    fi
    echo "[stage] review only: $SAVE_DIR"
    run_review
    ;;
  *)
    echo "[error] unsupported STAGE: $STAGE"
    echo "        use STAGE=smoke | STAGE=overnight | STAGE=review"
    exit 1
    ;;
esac

echo "[ok] workflow finished for $SAVE_DIR"
