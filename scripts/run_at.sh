#!/usr/bin/env bash
# Schedule a pipeline docker run on the DGX Spark at a local clock time.
# Default time is 17:00 in America/New_York. This script only *schedules*;
# it does not start training immediately.
#
# Usage (run on the Spark, from the project root or via this script's path):
#   ./scripts/run_at.sh                 # 17:00 today (or tomorrow if already past), stage=smoke
#   ./scripts/run_at.sh 17:00 smoke
#   ./scripts/run_at.sh 17:30 all
#   ./scripts/run_at.sh 17:00 smoke,translate   # smoke first, then the helper-model translations
#   ./scripts/run_at.sh 17:00 inject,verify
#
# Logs: logs/<run_id>.log
# Traces (written by the pipeline): ${OUTPUT_DIR:-artifacts}/traces/<run_id>.jsonl
#
# Prefers systemd-run --user --on-calendar (no sudo). The `at` daemon is not
# required and is often missing. User lingering may be off: a timer only fires
# while this user has an active session (keep an SSH or tmux login, or ask an
# admin for `loginctl enable-linger "$USER"`).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

TZ="${TZ:-America/New_York}"
export TZ

TIME_SPEC="${1:-17:00}"
STAGE_LIST="${2:-smoke}"
CONFIG_PATH="${3:-configs/spark.yaml}"

if [[ ! "$TIME_SPEC" =~ ^([0-9]|0[0-9]|1[0-9]|2[0-3]):[0-5][0-9]$ ]]; then
  echo "Time must be HH:MM (24-hour), got: $TIME_SPEC" >&2
  exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 1
fi
if [[ ! -f .env ]]; then
  echo "No .env in $ROOT. The scheduled docker run uses --env-file .env." >&2
  exit 1
fi
if ! command -v docker >/dev/null 2>&1; then
  echo "docker is not on PATH. Run this script on the Spark." >&2
  exit 1
fi

# Target the next occurrence of TIME_SPEC today, or tomorrow if that time has passed.
if target_epoch=$(date -d "today $TIME_SPEC:00" +%s 2>/dev/null); then
  :
else
  echo "GNU date is required (this script is meant for the Spark)." >&2
  exit 1
fi
now_epoch=$(date +%s)
if (( target_epoch <= now_epoch + 30 )); then
  target_epoch=$(date -d "tomorrow $TIME_SPEC:00" +%s)
fi
CALENDAR="$(date -d "@$target_epoch" "+%Y-%m-%d %H:%M:%S")"
WHEN_HUMAN="$(date -d "@$target_epoch" "+%Y-%m-%d %H:%M:%S %Z")"

mkdir -p logs
# Run id matches the pipeline --run-name so the log file and traces file share a name.
# Hash the config file bytes (the pipeline will use this --run-name as-is).
TS="$(date -u -d "@$target_epoch" +%Y%m%dT%H%M%SZ)"  # name runs by scheduled start, not scheduling time
SAFE_STAGE="${STAGE_LIST//,/+}"
SAFE_STAGE="$(echo "$SAFE_STAGE" | tr -c 'A-Za-z0-9._+-' '-')"
HASH="$(sha256sum "$CONFIG_PATH" | cut -c1-8)"
RUN_ID="${TS}-${SAFE_STAGE}-${HASH}"

LOG="$ROOT/logs/${RUN_ID}.log"
UNIT="lexsi-$(echo "$RUN_ID" | tr ':/' '--' | tr -c 'A-Za-z0-9._=-' '-' | cut -c1-80)"

# The command the timer will run. No training starts until the calendar time.
# --network host lets the container reach host Ollama at http://127.0.0.1:11434
# (verified with GET /api/tags). If you drop it, add
# --add-host=host.docker.internal:host-gateway and set
# OLLAMA_BASE_URL=http://host.docker.internal:11434/v1 instead.
RUN_CMD=$(cat <<EOF
set -euo pipefail
cd '$ROOT'
echo "[\$(date -Is)] starting run_id=$RUN_ID stage=$STAGE_LIST" >> '$LOG'
rc=0
docker run --rm --gpus all --ipc=host --network host --env-file .env \\
  -v "\$HOME/.cache/huggingface:/root/.cache/huggingface" \\
  -v "\$PWD:/workspace" -w /workspace \\
  lexsi-unlearning:latest \\
  python scripts/run_pipeline.py --config '$CONFIG_PATH' --stage '$STAGE_LIST' --run-name '$RUN_ID' \\
  >> '$LOG' 2>&1 || rc=\$?
echo "[\$(date -Is)] finished exit=\$rc" >> '$LOG'
exit \$rc
EOF
)

if command -v systemd-run >/dev/null 2>&1; then
  if ! systemctl --user is-system-running >/dev/null 2>&1; then
    echo "Warning: systemd --user does not look running. The timer may not fire until a user session is active."
  fi
  linger="$(loginctl show-user "$USER" -p Linger --value 2>/dev/null || true)"
  if [[ "$linger" != "yes" ]]; then
    echo "Note: linger is ${linger:-unknown}. Keep an SSH/tmux session open until $WHEN_HUMAN, or enable lingering."
  fi
  systemd-run --user \
    --unit="$UNIT" \
    --description="Lexsi pipeline $STAGE_LIST at $WHEN_HUMAN (run_id=$RUN_ID)" \
    --on-calendar="$CALENDAR" \
    --timer-property=AccuracySec=1s \
    --timer-property=Persistent=false \
    --working-directory="$ROOT" \
    --collect \
    /bin/bash -lc "$RUN_CMD"
  echo
  echo "Scheduled: $WHEN_HUMAN"
  echo "  unit:    $UNIT.timer"
  echo "  run_id:  $RUN_ID"
  echo "  stages:  $STAGE_LIST"
  echo "  log:     $LOG"
  echo "  traces:  \${OUTPUT_DIR:-artifacts}/traces/${RUN_ID}.jsonl"
  echo
  echo "Inspect:  systemctl --user list-timers --all | grep -E 'lexsi|$UNIT' || true"
  echo "Status:   systemctl --user status ${UNIT}.timer"
  echo "Cancel:   systemctl --user stop ${UNIT}.timer ${UNIT}.service; systemctl --user disable ${UNIT}.timer 2>/dev/null || true"
elif command -v at >/dev/null 2>&1; then
  echo "$RUN_CMD" | at "$TIME_SPEC"
  echo "Scheduled with at at $TIME_SPEC ($WHEN_HUMAN)."
  echo "  run_id: $RUN_ID"
  echo "  log:    $LOG"
  echo "List/cancel with: atq   and   atrm <job>"
else
  echo "Neither systemd-run nor at is available. Cannot schedule." >&2
  exit 1
fi
