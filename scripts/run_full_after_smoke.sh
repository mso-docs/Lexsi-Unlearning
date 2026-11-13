#!/usr/bin/env bash
# Chain the full submission-grade pipeline behind the scheduled smoke+translate run.
# Runs ON the Spark, normally launched by a systemd user timer (see scripts/run_at.sh for the smoke one).
#
# - Never starts before FLOOR (default 17:00 ET on FULL_DATE).
# - Waits until the smoke service is inactive, its log has a "finished exit=" line, and no
#   lexsi-unlearning docker container is running, so the two never share the GPU.
# - Starts whether smoke passed or failed; the smoke exit code is logged.
# - run_id is stamped with the REAL start time. Log: logs/<run_id>.log
# - After the pipeline exits 0 it runs the RMU sweep (scripts/sweep.py), best effort, unless RUN_SWEEP=0.
#
# Env (all optional except SMOKE_UNIT / SMOKE_RUN_ID):
#   SMOKE_UNIT, SMOKE_RUN_ID, FLOOR_LOCAL="2026-11-12 17:00:00", STAGES, CONFIG_PATH, RUN_SWEEP, MAX_WAIT_S
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export TZ="${TZ:-America/New_York}"

: "${SMOKE_UNIT:?}" "${SMOKE_RUN_ID:?}"
FLOOR_LOCAL="${FLOOR_LOCAL:-2026-11-12 17:00:00}"
STAGES="${STAGES:-inject,verify,unlearn,eval,probes,interp,relearn}"
CONFIG_PATH="${CONFIG_PATH:-configs/spark.yaml}"
RUN_SWEEP="${RUN_SWEEP:-1}"
MAX_WAIT_S="${MAX_WAIT_S:-14400}"          # give up (without running) after 4 h of waiting
IMAGE="lexsi-unlearning:latest"
JUDGE_BUDGET_S="${JUDGE_BUDGET_S:-5400}"   # per judge batch; the 1800 s default can truncate ~540-item batches

mkdir -p logs
SMOKE_LOG="$ROOT/logs/${SMOKE_RUN_ID}.log"
WAIT_LOG="$ROOT/logs/full-wait-$(date +%Y%m%dT%H%M%S).log"
log() { echo "[$(date -Is)] $*" | tee -a "$WAIT_LOG"; }

floor_epoch=$(date -d "$FLOOR_LOCAL" +%s)
deadline=$(( $(date +%s) + MAX_WAIT_S ))
log "waiting: floor=$FLOOR_LOCAL smoke_unit=$SMOKE_UNIT smoke_log=$SMOKE_LOG stages=$STAGES"

while :; do
  now=$(date +%s)
  if (( now < floor_epoch )); then sleep 10; continue; fi
  active=$(systemctl --user is-active "$SMOKE_UNIT" 2>/dev/null || true)
  busy=$(docker ps -q --filter "ancestor=$IMAGE" 2>/dev/null | wc -l | tr -d ' ')
  done_line=$(grep -o 'finished exit=[0-9]*' "$SMOKE_LOG" 2>/dev/null | tail -1 || true)
  started=$(grep -c 'starting run_id' "$SMOKE_LOG" 2>/dev/null || true)
  if [[ "$active" != "active" && "$active" != "activating" && "$busy" == "0" ]]; then
    if [[ -n "$done_line" ]]; then SMOKE_RC="${done_line#finished exit=}"; break; fi
    if [[ "${started:-0}" == "0" ]] && (( now > floor_epoch + 900 )); then
      SMOKE_RC="never-started"; break
    fi
    if [[ "${started:-0}" != "0" ]] && (( now > floor_epoch + 900 )); then
      # started, but service gone and no finish line: it died without logging (killed/crashed)
      SMOKE_RC="died-without-exit-line"; break
    fi
  fi
  if (( now > deadline )); then
    log "gave up waiting (active=$active busy=$busy); NOT starting the full run"
    exit 1
  fi
  sleep 15
done

TS="$(date -u +%Y%m%dT%H%M%SZ)"            # real start time
SAFE_STAGE="$(echo "${STAGES//,/+}" | tr -c 'A-Za-z0-9._+-' '-')"
HASH="$(sha256sum "$CONFIG_PATH" | cut -c1-8)"
RUN_ID="${TS}-${SAFE_STAGE}-${HASH}"
LOG="$ROOT/logs/${RUN_ID}.log"
{ cat "$WAIT_LOG"; } >> "$LOG"
echo "[$(date -Is)] smoke exit code: ${SMOKE_RC} (smoke run_id=$SMOKE_RUN_ID, log=$SMOKE_LOG)" | tee -a "$LOG"
echo "[$(date -Is)] starting run_id=$RUN_ID stage=$STAGES config=$CONFIG_PATH" >> "$LOG"

CNAME="lexsi-full-${TS}"
trap 'docker stop -t 30 "$CNAME" >/dev/null 2>&1; docker stop -t 30 "${CNAME}-sweep" >/dev/null 2>&1; echo "[$(date -Is)] stopped by signal" >> "$LOG"; exit 143' TERM INT

drun() { # name, then command
  local name="$1"; shift
  if [[ "${DRY_RUN:-0}" == "1" ]]; then echo "DRY_RUN would run: docker run --rm --init --name $name --gpus all --ipc=host --network host --env-file .env -e JUDGE_MAX_SECONDS=$JUDGE_BUDGET_S -v \$HOME/.cache/huggingface:/root/.cache/huggingface -v \$PWD:/workspace -w /workspace $IMAGE $*" >> "$LOG"; return 0; fi
  docker run --rm --init --name "$name" --gpus all --ipc=host --network host --env-file .env \
    -e JUDGE_MAX_SECONDS="$JUDGE_BUDGET_S" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -v "$PWD:/workspace" -w /workspace \
    "$IMAGE" "$@" >> "$LOG" 2>&1 &
  wait $!
}

drun "$CNAME" python scripts/run_pipeline.py --config "$CONFIG_PATH" --stage "$STAGES" --run-name "$RUN_ID"
rc=$?
echo "[$(date -Is)] finished exit=$rc" >> "$LOG"

if [[ "$RUN_SWEEP" == "1" && $rc -eq 0 ]]; then
  echo "[$(date -Is)] starting sweep (scripts/sweep.py, default combinations)" >> "$LOG"
  drun "${CNAME}-sweep" python scripts/sweep.py --config "$CONFIG_PATH"
  src=$?
  echo "[$(date -Is)] sweep finished exit=$src" >> "$LOG"
fi
exit $rc
