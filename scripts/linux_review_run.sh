#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-$HOME/derm_agent}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-derm-qwen}"
DATASET_ROOT="${DATASET_ROOT:-data/pad_ufes_20}"
SPLIT_JSON="${SPLIT_JSON:-outputs/splits/pad_ufes20_full.json}"
SPLIT_NAME="${SPLIT_NAME:-test}"
BASELINE_RUN_DIR="${BASELINE_RUN_DIR:-outputs/train_runs/final_v3_8h}"
CANDIDATE_RUN_DIR="${CANDIDATE_RUN_DIR:-outputs/train_runs/quick_iter}"
MIN_TOP1_GAIN="${MIN_TOP1_GAIN:-0.0}"
MIN_TOP3_GAIN="${MIN_TOP3_GAIN:-0.0}"
MIN_MALIGNANT_RECALL_GAIN="${MIN_MALIGNANT_RECALL_GAIN:-0.0}"
MIN_CONFUSION_GAIN="${MIN_CONFUSION_GAIN:-0.0}"
PROMOTE_DIR="${PROMOTE_DIR:-}"
DELETE_REJECTED="${DELETE_REJECTED:-0}"
LIMIT="${LIMIT:-}"

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

CMD=(
  python scripts/review_train_run.py
  --baseline-run-dir "$BASELINE_RUN_DIR"
  --candidate-run-dir "$CANDIDATE_RUN_DIR"
  --dataset-root "$DATASET_ROOT"
  --split-json "$SPLIT_JSON"
  --split-name "$SPLIT_NAME"
  --min-top1-gain "$MIN_TOP1_GAIN"
  --min-top3-gain "$MIN_TOP3_GAIN"
  --min-malignant-recall-gain "$MIN_MALIGNANT_RECALL_GAIN"
  --min-confusion-gain "$MIN_CONFUSION_GAIN"
)

if [[ -n "$PROMOTE_DIR" ]]; then
  CMD+=(--promote-dir "$PROMOTE_DIR")
fi

if [[ "$DELETE_REJECTED" == "1" ]]; then
  CMD+=(--delete-rejected)
fi

if [[ -n "$LIMIT" ]]; then
  CMD+=(--limit "$LIMIT")
fi

echo "[info] reviewing candidate run: $CANDIDATE_RUN_DIR"
"${CMD[@]}"
