#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-$HOME/derm_agent}"
TRAIN_HOURS="${TRAIN_HOURS:-0.25}"
SPLIT_JSON="${SPLIT_JSON:-outputs/splits/pad_ufes20_full.json}"
SAVE_DIR="${SAVE_DIR:-outputs/train_runs/quick_iter}"
CONTROLLER_IN="${CONTROLLER_IN:-outputs/controller_v2.json}"
BANK_IN="${BANK_IN:-outputs/experience_bank.json}"
SKILL_EVOLUTION_START_EPOCH="${SKILL_EVOLUTION_START_EPOCH:-999999}"
SKILL_EVOLUTION_EVERY="${SKILL_EVOLUTION_EVERY:-999999}"

cd "$PROJECT_DIR"

bash scripts/linux_build_full_split.sh "$PROJECT_DIR"

SAVE_DIR="$SAVE_DIR" \
SPLIT_JSON="$SPLIT_JSON" \
CONTROLLER_IN="$CONTROLLER_IN" \
BANK_IN="$BANK_IN" \
TRAIN_HOURS="$TRAIN_HOURS" \
SKILL_EVOLUTION_START_EPOCH="$SKILL_EVOLUTION_START_EPOCH" \
SKILL_EVOLUTION_EVERY="$SKILL_EVOLUTION_EVERY" \
bash scripts/linux_train_8h.sh "$PROJECT_DIR"
