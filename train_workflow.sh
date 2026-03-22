#!/usr/bin/env bash
set -euo pipefail

# DermAgent 统一训练工作流
# 使用配置文件驱动，支持多种训练模式

PROJECT_DIR="${1:-$HOME/derm_agent}"
CONFIG_FILE="${CONFIG_FILE:-config.ini}"
STAGE="${STAGE:-train}"
RUN_NAME="${RUN_NAME:-}"

cd "$PROJECT_DIR"

# 激活conda环境
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate derm-qwen 2>/dev/null || conda activate derm-agent 2>/dev/null || true
fi

# 读取配置文件
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "[error] Config file not found: $CONFIG_FILE"
  exit 1
fi

# 解析配置
TRAIN_BATCH_SIZE=$(grep "^batch_size" config.ini | cut -d'=' -f2 | tr -d ' ')
TRAIN_EVAL_INTERVAL=$(grep "^eval_interval" config.ini | cut -d'=' -f2 | tr -d ' ')
TRAIN_SAVE_INTERVAL=$(grep "^save_interval" config.ini | cut -d'=' -f2 | tr -d ' ')
DATA_LIMIT=$(grep "^data_limit" config.ini | cut -d'=' -f2 | tr -d ' ')

case "$STAGE" in
  train)
    echo "[stage] Training DermAgent"
    if [[ -z "$RUN_NAME" ]]; then
      RUN_NAME="train_$(date +%Y%m%d_%H%M%S)"
    fi

    python train_agent.py \
      --run-name "$RUN_NAME" \
      --batch-size "${TRAIN_BATCH_SIZE:-100}" \
      --eval-interval "${TRAIN_EVAL_INTERVAL:-1}" \
      --save-interval "${TRAIN_SAVE_INTERVAL:-5}" \
      --data-limit "${DATA_LIMIT:-1000}"

    echo "[success] Training completed: $RUN_NAME"
    ;;

  quick_test)
    echo "[stage] Quick test run"
    RUN_NAME="${RUN_NAME:-quick_test_$(date +%Y%m%d_%H%M%S)}"

    python train_agent.py \
      --run-name "$RUN_NAME" \
      --epochs 1 \
      --batch-size 10 \
      --eval-interval 1 \
      --save-interval 1

    echo "[success] Quick test completed: $RUN_NAME"
    ;;

  resume)
    echo "[stage] Resuming training"
    if [[ -z "$RUN_NAME" ]]; then
      echo "[error] RUN_NAME required for resume"
      exit 1
    fi

    python train_agent.py \
      --run-name "$RUN_NAME" \
      --resume-from "$RUN_NAME" \
      --batch-size "${TRAIN_BATCH_SIZE:-100}" \
      --eval-interval "${TRAIN_EVAL_INTERVAL:-1}" \
      --save-interval "${TRAIN_SAVE_INTERVAL:-5}"

    echo "[success] Resume training completed: $RUN_NAME"
    ;;

  evaluate)
    echo "[stage] Evaluating model"
    if [[ -z "$RUN_NAME" ]]; then
      echo "[error] RUN_NAME required for evaluation"
      exit 1
    fi

    python -c "
from evaluation.run_eval import run_evaluation
from memory.weights_manager import weights_manager
from memory.skill_index import build_default_skill_index

# 加载训练好的组件
skill_index = build_default_skill_index()
learning_components = weights_manager.initialize_components(skill_index)
weights_manager.load_checkpoint(learning_components, '$RUN_NAME')

# 运行评估
result = run_evaluation(
    cases_limit=200,
    learning_components=learning_components,
    skill_index=skill_index
)

print('Evaluation Results:')
print(f'  Accuracy: {result[\"accuracy\"]:.3f}')
print(f'  Top-3 Accuracy: {result[\"top3_accuracy\"]:.3f}')
print(f'  Malignant Recall: {result[\"malignant_recall\"]:.3f}')
print(f'  Confusion Accuracy: {result[\"confusion_accuracy\"]:.3f}')
"

    echo "[success] Evaluation completed"
    ;;

  *)
    echo "[error] Unknown stage: $STAGE"
    echo "Available stages: train, quick_test, resume, evaluate"
    exit 1
    ;;
esac