#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-$HOME/derm_agent}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-derm-qwen}"
TRAIN_HOURS="${TRAIN_HOURS:-8}"
EPOCHS="${EPOCHS:-}"
SAVE_DIR="${SAVE_DIR:-outputs/train_runs/server_v2_8h}"
SPLIT_JSON="${SPLIT_JSON:-outputs/splits/pad_ufes20_full.json}"
LIMIT="${LIMIT:-}"
INIT_MODE="${INIT_MODE:-clean}"
BASE_RUN_DIR="${BASE_RUN_DIR:-}"
CONTROLLER_IN="${CONTROLLER_IN:-}"
BANK_IN="${BANK_IN:-}"
NOTES="${NOTES:-}"
ALLOW_DIR_REUSE="${ALLOW_DIR_REUSE:-0}"
SKILL_EVOLUTION_START_EPOCH="${SKILL_EVOLUTION_START_EPOCH:-12}"
SKILL_EVOLUTION_EVERY="${SKILL_EVOLUTION_EVERY:-3}"

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

if [[ ! -f "$SPLIT_JSON" ]]; then
  echo "[info] split file not found, building full split at $SPLIT_JSON"
  python scripts/build_split.py \
    --dataset-root data/pad_ufes_20 \
    --seed 42 \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15 \
    --output "$SPLIT_JSON"
fi

resolve_init_inputs() {
  case "$INIT_MODE" in
    clean)
      ;;
    seed)
      if [[ -z "$CONTROLLER_IN" ]]; then
        CONTROLLER_IN="outputs/checkpoints/seed/controller_v2.json"
      fi
      if [[ -z "$BANK_IN" ]]; then
        BANK_IN="outputs/checkpoints/seed/experience_bank.json"
      fi
      ;;
    resume)
      if [[ -z "$CONTROLLER_IN" ]]; then
        CONTROLLER_IN="$SAVE_DIR/latest_controller.json"
      fi
      if [[ -z "$BANK_IN" ]]; then
        BANK_IN="$SAVE_DIR/latest_bank.json"
      fi
      ;;
    run_best)
      if [[ -z "$BASE_RUN_DIR" ]]; then
        echo "[error] INIT_MODE=run_best requires BASE_RUN_DIR"
        exit 1
      fi
      if [[ -z "$CONTROLLER_IN" ]]; then
        CONTROLLER_IN="$BASE_RUN_DIR/best_controller.json"
      fi
      if [[ -z "$BANK_IN" ]]; then
        BANK_IN="$BASE_RUN_DIR/best_bank.json"
      fi
      ;;
    *)
      echo "[error] unsupported INIT_MODE: $INIT_MODE"
      echo "        use INIT_MODE=clean | seed | resume | run_best"
      exit 1
      ;;
  esac
}

assert_optional_input_exists() {
  local label="$1"
  local path="$2"
  if [[ -n "$path" && ! -f "$path" ]]; then
    echo "[error] missing $label checkpoint: $path"
    exit 1
  fi
}

resolve_init_inputs
assert_optional_input_exists "controller" "$CONTROLLER_IN"
assert_optional_input_exists "bank" "$BANK_IN"

echo "[info] init_mode=$INIT_MODE"
if [[ -n "$BASE_RUN_DIR" ]]; then
  echo "[info] base_run_dir=$BASE_RUN_DIR"
fi
if [[ -n "$CONTROLLER_IN" ]]; then
  echo "[info] controller_init=$CONTROLLER_IN"
else
  echo "[info] controller_init=<clean>"
fi
if [[ -n "$BANK_IN" ]]; then
  echo "[info] bank_init=$BANK_IN"
else
  echo "[info] bank_init=<clean>"
fi
if [[ -n "$NOTES" ]]; then
  echo "[info] notes=$NOTES"
fi

TRAIN_CMD=(
  python scripts/train_server.py
  --dataset-root data/pad_ufes_20
  --eval-every 1
  --split-json "$SPLIT_JSON"
  --save-dir "$SAVE_DIR"
  --run-name "$(basename "$SAVE_DIR")"
  --init-mode "$INIT_MODE"
  --skill-evolution-start-epoch "$SKILL_EVOLUTION_START_EPOCH"
  --skill-evolution-every "$SKILL_EVOLUTION_EVERY"
)

if [[ -n "$BASE_RUN_DIR" ]]; then
  TRAIN_CMD+=(--base-run-dir "$BASE_RUN_DIR")
fi
if [[ -n "$NOTES" ]]; then
  TRAIN_CMD+=(--notes "$NOTES")
fi
if [[ -n "$LIMIT" ]]; then
  TRAIN_CMD+=(--limit "$LIMIT")
fi
if [[ -n "$CONTROLLER_IN" ]]; then
  TRAIN_CMD+=(--controller-state-in "$CONTROLLER_IN")
fi
if [[ -n "$BANK_IN" ]]; then
  TRAIN_CMD+=(--bank-state-in "$BANK_IN")
fi
if [[ "$INIT_MODE" == "resume" ]]; then
  TRAIN_CMD+=(--resume)
fi
if [[ "$ALLOW_DIR_REUSE" == "1" ]]; then
  TRAIN_CMD+=(--allow-dirty-save-dir)
fi

if [[ -n "$EPOCHS" ]]; then
  echo "[info] training mode: epoch_based epochs=$EPOCHS"
  "${TRAIN_CMD[@]}" --epochs "$EPOCHS"
else
  echo "[info] training mode: time_based train_hours=$TRAIN_HOURS"
  set +e
  timeout "${TRAIN_HOURS}h" "${TRAIN_CMD[@]}" --epochs 999999

  TRAIN_EXIT_CODE=$?
  set -e
  if [[ "$TRAIN_EXIT_CODE" -ne 0 && "$TRAIN_EXIT_CODE" -ne 124 ]]; then
    echo "[error] training exited with code $TRAIN_EXIT_CODE"
    exit "$TRAIN_EXIT_CODE"
  fi
fi

echo "[ok] training window finished"
python scripts/run_eval_brief.py \
  --dataset-root data/pad_ufes_20 \
  --limit 999999 \
  --split-json "$SPLIT_JSON" \
  --split-name test \
  --controller-state-in "$SAVE_DIR/best_controller.json" \
  --bank-state-in "$SAVE_DIR/best_bank.json" \
  --freeze-learning
