#!/usr/bin/env bash
# Hard 07:30 ET backstop, run by lexsi-overnight-deadline.timer. Independent of the queue script, so it still
# works if that script died. Stops the queue service (its trap docker-stops the job), then docker-stops any
# remaining lexsi container (queue jobs lexsi-q-*, and the main chain's lexsi-full-* if it is still running),
# keeps all partial output, and logs what it did. Nothing is deleted.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export TZ="${TZ:-America/New_York}"
GRACE="${STOP_GRACE_S:-45}"
LOG="$ROOT/logs/${DEADLINE_LOG_PREFIX:-overnight-deadline}-$(date +%Y%m%d).log"
log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }
log "deadline backstop fired"
UNITS="${DEADLINE_UNITS:-lexsi-overnight-queue.timer lexsi-overnight-queue.service lexsi-full-after-smoke.timer lexsi-full-after-smoke.service lexsi-full-fix1.service lexsi-full-fix2.service}"
for u in $UNITS; do
  st=$(systemctl --user is-active "$u" 2>/dev/null || true)
  if [[ "$st" == active || "$st" == activating || "$st" == waiting ]]; then log "stopping $u (was $st)"; systemctl --user stop "$u" 2>&1 | tee -a "$LOG"; fi
done
for c in $(docker ps --format '{{.Names}}' --filter name=lexsi-q- --filter name=lexsi-full-); do
  log "docker stop -t $GRACE $c (partial results kept)"
  docker stop -t "$GRACE" "$c" 2>&1 | tee -a "$LOG"
done
left=$(docker ps -q --filter "ancestor=lexsi-unlearning:latest" | wc -l | tr -d ' ')
log "done; lexsi containers still running: $left"
