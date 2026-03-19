#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${1:-}"
TARGET_DIR="${2:-$HOME/derm_agent}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-derm-qwen}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

if [[ -z "$REPO_URL" ]]; then
  echo "Usage: bash scripts/linux_clone_setup.sh <repo_url> [target_dir]"
  exit 1
fi

if [[ -d "$TARGET_DIR/.git" ]]; then
  echo "[info] repo already exists at $TARGET_DIR"
else
  git clone "$REPO_URL" "$TARGET_DIR"
fi

cd "$TARGET_DIR"

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  # shellcheck disable=SC1090
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true
  if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV_NAME"; then
    conda create -n "$CONDA_ENV_NAME" "python=$PYTHON_VERSION" -y
  fi
  conda activate "$CONDA_ENV_NAME"
else
  echo "[error] conda not found. Please install Miniconda or Anaconda first."
  exit 1
fi

python -m pip install --upgrade pip
pip install -r requirements.txt

if [[ -f "data/pad_ufes_20/metadata.csv" ]]; then
  echo "[ok] dataset already present"
else
  echo "[info] dataset not found yet"
  echo "[info] next step: bash scripts/linux_download_data.sh $TARGET_DIR"
fi

if [[ -f "data/pad_ufes_20/metadata.csv" ]]; then
  python scripts/build_split.py \
    --dataset-root data/pad_ufes_20 \
    --output outputs/splits/pad_ufes20.json
fi

echo "[ok] repo and environment are ready at $TARGET_DIR"
echo "[ok] conda env: $CONDA_ENV_NAME"
if [[ -f "outputs/splits/pad_ufes20.json" ]]; then
  echo "[ok] split file: outputs/splits/pad_ufes20.json"
fi
