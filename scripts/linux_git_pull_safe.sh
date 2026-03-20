#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-$HOME/derm_agent}"

cd "$PROJECT_DIR"

bash scripts/linux_protect_checkpoints.sh "$PROJECT_DIR"

git fetch origin
git pull --no-rebase

echo "[ok] code synced with local checkpoints protected."
