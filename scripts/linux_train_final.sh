#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-$HOME/derm_agent}"
SPLIT_JSON="${SPLIT_JSON:-outputs/splits/pad_ufes20_full.json}"
SAVE_DIR="${SAVE_DIR:-outputs/train_runs/final_v3_8h}"
CONTROLLER_IN="${CONTROLLER_IN:-outputs/checkpoints/seed/controller_v2.json}"
BANK_IN="${BANK_IN:-outputs/checkpoints/seed/experience_bank.json}"
TRAIN_HOURS="${TRAIN_HOURS:-8}"
SKILL_EVOLUTION_START_EPOCH="${SKILL_EVOLUTION_START_EPOCH:-12}"
SKILL_EVOLUTION_EVERY="${SKILL_EVOLUTION_EVERY:-3}"

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
