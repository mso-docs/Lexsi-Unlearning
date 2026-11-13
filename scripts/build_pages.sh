#!/usr/bin/env bash
# Build the public GitHub Pages snapshot: optional Spark sync, redact, ingest, export.
# Does not commit, push, or change GitHub Pages settings. Does not touch running jobs
# except for a read-only rsync when --sync is passed.
#
#   scripts/build_pages.sh              # redact local artifacts/logs, export docs/
#   scripts/build_pages.sh --sync       # rsync --results from the Spark first
#   scripts/build_pages.sh --dry-run    # redact report only; do not write release/ or docs/data
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SYNC=0
DRY=0
TRIM_BYTES="${TRIM_BYTES:-250000}"
EVALBOARD_ROOT="${EVALBOARD_ROOT:-}"

usage() {
  cat <<'USAGE'
Usage: scripts/build_pages.sh [--sync] [--dry-run]

  --sync      Run scripts/sync_from_spark.sh --results first (read-only rsync).
  --dry-run   Write the redaction report only; do not write release/ or docs/.

  EVALBOARD_ROOT   Checkout of modular_dashboard (auto-detected if unset).
  TRIM_BYTES       Max interp artifact size in the Pages bundle (default 250000).
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sync) SYNC=1; shift ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

detect_evalboard() {
  local c
  if [[ -n "$EVALBOARD_ROOT" && -f "$EVALBOARD_ROOT/evalboard/export_static.py" ]]; then
    return 0
  fi
  for c in \
      "${HOME}/Documents/Code/modular_dashboard" \
      "${HOME}/Documents/code/modular_dashboard" \
      "$(dirname "$ROOT")/modular_dashboard" \
      "$(dirname "$ROOT")/Code/modular_dashboard"
  do
    if [[ -f "$c/evalboard/export_static.py" ]]; then
      EVALBOARD_ROOT="$c"
      return 0
    fi
  done
  echo "Could not find the Evalboard toolkit (modular_dashboard). Set EVALBOARD_ROOT." >&2
  exit 2
}

if [[ "$SYNC" -eq 1 ]]; then
  "$ROOT/scripts/sync_from_spark.sh" --results
fi

REPORT="${ROOT}/artifacts/redact_report.md"
if [[ "$DRY" -eq 1 ]]; then
  python3 "$ROOT/scripts/redact_release.py" --root "$ROOT" --dry-run --report "$REPORT"
  exit $?
fi

python3 "$ROOT/scripts/redact_release.py" \
  --root "$ROOT" \
  --hf-out "$ROOT/release/hf_dataset" \
  --pages-out "$ROOT/release/pages_src" \
  --report "$REPORT"

detect_evalboard
export PYTHONPATH="${EVALBOARD_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export EVALBOARD_ROOT

DB="$ROOT/release/pages_workspace.sqlite3"
rm -f "$DB"
shopt -s nullglob
traces=("$ROOT/release/pages_src/traces"/*.jsonl)
if [[ ${#traces[@]} -eq 0 ]]; then
  echo "No scrubbed traces in release/pages_src/traces/. Nothing to export." >&2
  exit 1
fi

for f in "${traces[@]}"; do
  run_id="$(basename "$f" .jsonl)"
  echo "Ingest $run_id"
  python3 "$ROOT/scripts/ingest_evalboard.py" \
    --db "$DB" \
    --run "$run_id" \
    --output-dir "$ROOT/release/pages_src"
done

mkdir -p "$ROOT/docs"
python3 -m evalboard.export_static \
  --db "$DB" \
  --out "$ROOT/docs" \
  --trim-bytes "$TRIM_BYTES"

echo "Pages bundle is at $ROOT/docs/. Preview with:"
echo "  python3 -m http.server -d $ROOT/docs 8080"
echo "Do not commit docs/data/ until the weekend runs are finished and scrubbed."
