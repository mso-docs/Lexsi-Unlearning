#!/usr/bin/env bash
# Pull results / artifacts back from the DGX Spark. Optional; not run automatically.
# Does not overwrite .env. Does not start, stop, or otherwise touch running units
# or containers — it only rsyncs files out.
#
# Usage:
#   scripts/sync_from_spark.sh            # full ${OUTPUT_DIR:-artifacts}/ (includes weights)
#   scripts/sync_from_spark.sh --results  # traces, results, interp, circuitkit, logs,
#                                         # and data/translations.json; excludes weights
#                                         # and checkpoints
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RESULTS_ONLY=0
usage() {
  cat <<'USAGE'
Usage: scripts/sync_from_spark.sh [--results]

  --results   Copy traces, results, interp, circuitkit, logs, and
              data/translations.json only. Exclude model weights and
              checkpoints (*.safetensors, checkpoint-*, rmu_unlearned/,
              inject_lora/, npo_unlearned/).

  (default)   Rsync the entire ${OUTPUT_DIR:-artifacts}/ tree, including
              weights. Prefer --results unless you really need a checkpoint.

Does not overwrite .env. Read-only with respect to running jobs.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --results) RESULTS_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

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

REMOTE="${SPARK_USER}@${SPARK_HOST}"
SSH=(ssh -p "${SPARK_SSH_PORT}")
RSYNC_SSH="ssh -p ${SPARK_SSH_PORT}"

if [[ "$RESULTS_ONLY" -eq 0 ]]; then
  mkdir -p "$ROOT/$OUTPUT_DIR"
  echo "Rsync results from ${REMOTE}:${SPARK_PROJECT_DIR}/${OUTPUT_DIR}/ -> $ROOT/$OUTPUT_DIR/"
  rsync -avz --progress \
    -e "$RSYNC_SSH" \
    --exclude '.env' \
    "${REMOTE}:${SPARK_PROJECT_DIR}/${OUTPUT_DIR}/" \
    "$ROOT/$OUTPUT_DIR/"
  echo "Done. Local copy is at $ROOT/$OUTPUT_DIR/"
  exit 0
fi

# --results: selected subtrees only, no weights / checkpoints.
WEIGHT_EXCLUDES=(
  --exclude '.env'
  --exclude '*.safetensors'
  --exclude '*.bin'
  --exclude '*.pt'
  --exclude '*.pth'
  --exclude '*.ckpt'
  --exclude '*.gguf'
  --exclude 'checkpoint-*/'
  --exclude 'rmu_unlearned/'
  --exclude 'inject_lora/'
  --exclude 'npo_unlearned/'
)

echo "Rsync --results from ${REMOTE}:${SPARK_PROJECT_DIR} (no weights or checkpoints)"

for sub in traces results interp circuitkit; do
  # Skip remote dirs that do not exist yet (e.g. circuitkit mid-run).
  if "${SSH[@]}" "$REMOTE" "test -d '${SPARK_PROJECT_DIR}/${OUTPUT_DIR}/${sub}'"; then
    mkdir -p "$ROOT/$OUTPUT_DIR/$sub"
    echo "  ${OUTPUT_DIR}/${sub}/"
    rsync -avz --progress \
      -e "$RSYNC_SSH" \
      "${WEIGHT_EXCLUDES[@]}" \
      "${REMOTE}:${SPARK_PROJECT_DIR}/${OUTPUT_DIR}/${sub}/" \
      "$ROOT/$OUTPUT_DIR/$sub/"
  else
    echo "  skip ${OUTPUT_DIR}/${sub}/ (not on the Spark yet)"
  fi
done

# Selected loose files under artifacts/ (review sheet, never weights).
if "${SSH[@]}" "$REMOTE" "test -f '${SPARK_PROJECT_DIR}/${OUTPUT_DIR}/translations_review.md'"; then
  mkdir -p "$ROOT/$OUTPUT_DIR"
  rsync -avz --progress -e "$RSYNC_SSH" \
    "${REMOTE}:${SPARK_PROJECT_DIR}/${OUTPUT_DIR}/translations_review.md" \
    "$ROOT/$OUTPUT_DIR/translations_review.md"
fi

if "${SSH[@]}" "$REMOTE" "test -d '${SPARK_PROJECT_DIR}/logs'"; then
  mkdir -p "$ROOT/logs"
  echo "  logs/"
  rsync -avz --progress \
    -e "$RSYNC_SSH" \
    --exclude '.env' \
    "${REMOTE}:${SPARK_PROJECT_DIR}/logs/" \
    "$ROOT/logs/"
else
  echo "  skip logs/ (not on the Spark yet)"
fi

if "${SSH[@]}" "$REMOTE" "test -f '${SPARK_PROJECT_DIR}/data/translations.json'"; then
  mkdir -p "$ROOT/data"
  echo "  data/translations.json"
  rsync -avz --progress -e "$RSYNC_SSH" \
    "${REMOTE}:${SPARK_PROJECT_DIR}/data/translations.json" \
    "$ROOT/data/translations.json"
else
  echo "  skip data/translations.json (not on the Spark yet)"
fi

echo "Done. Scrub before publishing: python3 scripts/redact_release.py --help"
