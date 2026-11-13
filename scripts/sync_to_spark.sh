#!/usr/bin/env bash
# Copy this project to the DGX Spark over Tailscale SSH/rsync.
# Do not run this script unless you mean to; fill .env first.
# .env itself is EXCLUDED from the rsync — copy it separately/securely
# or create it on the Spark (recommended: scp once, or paste values on the Spark).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -f .env ]]; then
  echo "No .env in $ROOT. Copy .env.example to .env and fill SPARK_HOST / SPARK_USER / SPARK_PROJECT_DIR first."
  exit 1
fi

# Load .env without printing values. Only KEY=VALUE lines; ignore comments.
set -a
# shellcheck disable=SC1091
source .env
set +a

: "${SPARK_HOST:?Set SPARK_HOST in .env (Tailscale MagicDNS name, e.g. spark.tailXXXX.ts.net)}"
: "${SPARK_USER:?Set SPARK_USER in .env}"
: "${SPARK_PROJECT_DIR:?Set SPARK_PROJECT_DIR in .env (absolute path on the Spark)}"
SPARK_SSH_PORT="${SPARK_SSH_PORT:-22}"

if [[ "$SPARK_HOST" == *"XXXX"* ]]; then
  echo "SPARK_HOST still has the tailXXXX placeholder. Put your real MagicDNS name in .env."
  exit 1
fi

echo "Rsync $ROOT -> ${SPARK_USER}@${SPARK_HOST}:${SPARK_PROJECT_DIR}"
echo "Excluding .git, .env, logs, caches, and artifacts. .env must be copied separately."

# trailing slash on the destination would also be fine; we create the dir first.
ssh -p "$SPARK_SSH_PORT" "${SPARK_USER}@${SPARK_HOST}" "mkdir -p '$SPARK_PROJECT_DIR'"

rsync -avz --progress \
  -e "ssh -p ${SPARK_SSH_PORT}" \
  --exclude '.git/' \
  --exclude '.env' \
  --exclude '.env.local' \
  --exclude 'artifacts/' \
  --exclude 'outputs/' \
  --exclude 'checkpoints/' \
  --exclude '__pycache__/' \
  --exclude '.venv/' \
  --exclude 'venv/' \
  --exclude '.ipynb_checkpoints/' \
  --exclude '.DS_Store' \
  --exclude 'hf_cache/' \
  --exclude 'wandb/' \
  --exclude 'logs/' \
  "$ROOT/" "${SPARK_USER}@${SPARK_HOST}:${SPARK_PROJECT_DIR}/"

echo
echo "Code copied. Next:"
echo "  # copy env securely (example; review before running)"
echo "  scp -P ${SPARK_SSH_PORT} .env ${SPARK_USER}@${SPARK_HOST}:${SPARK_PROJECT_DIR}/.env"
echo "  # or ssh in and create .env on the Spark from .env.example"
echo "See scripts/spark_setup.md for container / uv setup and how to pull results back."
