#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-$HOME/derm_agent}"

cd "$PROJECT_DIR"

if ! command -v git >/dev/null 2>&1; then
  echo "[error] git not found."
  exit 1
fi

git config pull.rebase false

PROTECT_PATHS=(
  "outputs/train_runs"
  "outputs/checkpoints_backup"
  "outputs/logs"
)

for path in "${PROTECT_PATHS[@]}"; do
  if [[ -e "$path" ]]; then
    while IFS= read -r tracked_file; do
      [[ -n "$tracked_file" ]] || continue
      git update-index --skip-worktree "$tracked_file" 2>/dev/null || true
    done < <(git ls-files "$path" 2>/dev/null || true)
  fi
done

echo "[ok] local checkpoint protection enabled."
echo "[info] protected paths:"
for path in "${PROTECT_PATHS[@]}"; do
  echo "  - $path"
done
echo "[info] future git pull will prefer keeping your local training artifacts untouched."
