#!/usr/bin/env bash
# Pull results / artifacts back from the DGX Spark. Optional; not run automatically.
# Fills the local artifacts/ and OUTPUT_DIR from the Spark copy. Does not overwrite .env.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -f .env ]]; then
  echo "No .env in $ROOT."
  exit 1
fi

set -a
# shellcheck disable=SC1091
source .env
set +a

: "${SPARK_HOST:?Set SPARK_HOST in .env}"
: "${SPARK_USER:?Set SPARK_USER in .env}"
: "${SPARK_PROJECT_DIR:?Set SPARK_PROJECT_DIR in .env}"
SPARK_SSH_PORT="${SPARK_SSH_PORT:-22}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts}"

if [[ "$SPARK_HOST" == *"XXXX"* ]]; then
  echo "SPARK_HOST still has the tailXXXX placeholder."
  exit 1
fi

mkdir -p "$ROOT/$OUTPUT_DIR"
echo "Rsync results from ${SPARK_USER}@${SPARK_HOST}:${SPARK_PROJECT_DIR}/${OUTPUT_DIR}/ -> $ROOT/$OUTPUT_DIR/"
rsync -avz --progress \
  -e "ssh -p ${SPARK_SSH_PORT}" \
  --exclude '.env' \
  "${SPARK_USER}@${SPARK_HOST}:${SPARK_PROJECT_DIR}/${OUTPUT_DIR}/" \
  "$ROOT/$OUTPUT_DIR/"

echo "Done. Local copy is at $ROOT/$OUTPUT_DIR/"
