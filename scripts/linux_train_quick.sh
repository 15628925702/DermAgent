#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-$HOME/derm_agent}"
TRAIN_HOURS="${TRAIN_HOURS:-0.25}"
SPLIT_JSON="${SPLIT_JSON:-outputs/splits/pad_ufes20_full.json}"
SAVE_DIR="${SAVE_DIR:-outputs/train_runs/quick_iter}"
BASE_RUN_DIR="${BASE_RUN_DIR:-outputs/train_runs/final_v3_8h}"
DEFAULT_CONTROLLER_IN="outputs/controller_v2.json"
DEFAULT_BANK_IN="outputs/experience_bank.json"
SKILL_EVOLUTION_START_EPOCH="${SKILL_EVOLUTION_START_EPOCH:-999999}"
SKILL_EVOLUTION_EVERY="${SKILL_EVOLUTION_EVERY:-999999}"

cd "$PROJECT_DIR"

bash scripts/linux_build_full_split.sh "$PROJECT_DIR"

if [[ -z "${CONTROLLER_IN:-}" ]]; then
  if [[ -f "$BASE_RUN_DIR/best_controller.json" ]]; then
    CONTROLLER_IN="$BASE_RUN_DIR/best_controller.json"
  else
    CONTROLLER_IN="$DEFAULT_CONTROLLER_IN"
  fi
fi

if [[ -z "${BANK_IN:-}" ]]; then
  if [[ -f "$BASE_RUN_DIR/best_bank.json" ]]; then
    BANK_IN="$BASE_RUN_DIR/best_bank.json"
  else
    BANK_IN="$DEFAULT_BANK_IN"
  fi
fi

echo "[info] quick warm start controller: $CONTROLLER_IN"
echo "[info] quick warm start bank: $BANK_IN"

mkdir -p "$SAVE_DIR"
if [[ -f "$CONTROLLER_IN" ]]; then
  cp "$CONTROLLER_IN" "$SAVE_DIR/best_controller.json"
fi
if [[ -f "$BANK_IN" ]]; then
  cp "$BANK_IN" "$SAVE_DIR/best_bank.json"
fi
echo "[info] quick seed checkpoint copied into $SAVE_DIR"

SAVE_DIR="$SAVE_DIR" \
SPLIT_JSON="$SPLIT_JSON" \
CONTROLLER_IN="$CONTROLLER_IN" \
BANK_IN="$BANK_IN" \
TRAIN_HOURS="$TRAIN_HOURS" \
SKILL_EVOLUTION_START_EPOCH="$SKILL_EVOLUTION_START_EPOCH" \
SKILL_EVOLUTION_EVERY="$SKILL_EVOLUTION_EVERY" \
bash scripts/linux_train_8h.sh "$PROJECT_DIR"
