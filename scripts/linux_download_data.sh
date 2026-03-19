#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-$HOME/derm_agent}"
DATA_DIR="$PROJECT_DIR/data"
TARGET_DIR="$DATA_DIR/pad_ufes_20"
ZIP_PATH="$DATA_DIR/pad_ufes_20.zip"
SOURCE_PAGE="${SOURCE_PAGE:-https://data.mendeley.com/datasets/zr7vgbcyr2/1}"
DIRECT_URL="${DIRECT_URL:-}"

mkdir -p "$DATA_DIR"

if [[ -f "$TARGET_DIR/metadata.csv" ]]; then
  echo "[ok] dataset already exists at $TARGET_DIR"
  exit 0
fi

if [[ -z "$DIRECT_URL" ]]; then
  echo "[info] trying to resolve PAD-UFES-20 official download link from Mendeley"
  DIRECT_URL="$(
    python - <<'PY'
import re
import sys
import urllib.request

source_page = "https://data.mendeley.com/datasets/zr7vgbcyr2/1"
request = urllib.request.Request(
    source_page,
    headers={
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    },
)
html = urllib.request.urlopen(request, timeout=60).read().decode("utf-8", errors="ignore")
patterns = [
    r'https://data\.mendeley\.com/public-files/datasets/[^"\']+/file_downloaded',
    r'https://prod-dcd-datasets-public-files-eu-west-1\.s3\.eu-west-1\.amazonaws\.com/[^"\']+',
]
for pattern in patterns:
    match = re.search(pattern, html)
    if match:
        print(match.group(0))
        sys.exit(0)
sys.exit(1)
PY
  )" || true
fi

if [[ -z "$DIRECT_URL" ]]; then
  echo "[error] could not automatically resolve the official dataset file URL"
  echo "[error] open this page in a browser and copy the final zip link if needed:"
  echo "        $SOURCE_PAGE"
  echo "[error] then rerun with:"
  echo "        DIRECT_URL='<real_zip_url>' bash scripts/linux_download_data.sh $PROJECT_DIR"
  exit 1
fi

echo "[info] downloading dataset zip"
echo "[info] source: $DIRECT_URL"
curl -L \
  -A "Mozilla/5.0" \
  -H "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8" \
  "$DIRECT_URL" -o "$ZIP_PATH"

if [[ ! -s "$ZIP_PATH" ]]; then
  echo "[error] downloaded zip is empty: $ZIP_PATH"
  exit 1
fi

echo "[info] extracting dataset"
rm -rf "$TARGET_DIR"
unzip -q "$ZIP_PATH" -d "$DATA_DIR"

if [[ ! -f "$TARGET_DIR/metadata.csv" ]]; then
  echo "[error] extraction finished but metadata.csv is missing"
  echo "[error] inspect contents under $DATA_DIR"
  exit 1
fi

cd "$PROJECT_DIR"
python scripts/build_split.py \
  --dataset-root data/pad_ufes_20 \
  --output outputs/splits/pad_ufes20.json

echo "[ok] dataset is ready at $TARGET_DIR"
