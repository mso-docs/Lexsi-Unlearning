#!/usr/bin/env bash
# Create the two transient systemd user timers for the overnight queue. Run ON the Spark.
#   lexsi-overnight-queue.timer     -> .service runs scripts/overnight_queue.sh (it waits for the chain itself)
#   lexsi-overnight-deadline.timer  -> .service runs scripts/overnight_queue_deadline.sh at the hard deadline
# Starting a queue unit before 17:05 ET is harmless: the script will not start anything before its floor.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export TZ="${TZ:-America/New_York}"
QUEUE_START="${QUEUE_START:-2026-11-12 17:03:00}"
DEADLINE="${DEADLINE_LOCAL:-2026-11-13 07:30:00}"
systemd-run --user --unit=lexsi-overnight-queue \
  --description="Lexsi overnight job queue (waits for the full-run chain; deadline $DEADLINE ET)" \
  --on-calendar="$QUEUE_START" --timer-property=AccuracySec=1s --timer-property=Persistent=false \
  --working-directory="$ROOT" --property=TimeoutStopSec=150 --setenv=TZ="$TZ" --setenv=HOME="$HOME" \
  --setenv=DEADLINE_LOCAL="$DEADLINE" --collect \
  /bin/bash "$ROOT/scripts/overnight_queue.sh"
systemd-run --user --unit=lexsi-overnight-deadline \
  --description="Lexsi overnight queue hard stop at $DEADLINE ET" \
  --on-calendar="$DEADLINE" --timer-property=AccuracySec=1s --timer-property=Persistent=false \
  --working-directory="$ROOT" --setenv=TZ="$TZ" --setenv=HOME="$HOME" --collect \
  /bin/bash "$ROOT/scripts/overnight_queue_deadline.sh"
systemctl --user list-timers --all --no-pager | grep -E 'lexsi|NEXT' || true
