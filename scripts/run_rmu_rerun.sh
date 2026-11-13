#!/usr/bin/env bash
# Re-run the main Global pipeline FROM RMU onward, reusing the injected adapter and the pre-RMU results.
# Runs ON the Spark, normally as a transient unit:
#   systemd-run --user --unit=lexsi-full-fix2 --working-directory=$PWD --collect \
#     -E RESUME_RESULTS=... -E RESUME_TRACES=... /bin/bash scripts/run_rmu_rerun.sh
#
# - Skips inject/verify (artifacts/inject_lora is reused); base + injected + before_unlearn numbers are carried over
#   from RESUME_RESULTS / RESUME_TRACES (see src/lexsi_unlearn/resume.py).
# - Waits for any other lexsi-* GPU container to finish first, so nothing shares the GPU.
# - After the pipeline exits 0 it runs scripts/sweep.py (same config, so the same forget_mask), unless RUN_SWEEP=0.
# Env: RESUME_RESULTS, RESUME_TRACES (required); STAGES, CONFIG_PATH, RUN_SWEEP, MAX_WAIT_S, JUDGE_BUDGET_S
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export TZ="${TZ:-America/New_York}"
: "${RESUME_RESULTS:?}" "${RESUME_TRACES:?}"
STAGES="${STAGES:-unlearn,eval,probes,interp,relearn}"
CONFIG_PATH="${CONFIG_PATH:-configs/spark.yaml}"
RUN_SWEEP="${RUN_SWEEP:-1}"
MAX_WAIT_S="${MAX_WAIT_S:-7200}"
IMAGE="lexsi-unlearning:latest"
JUDGE_BUDGET_S="${JUDGE_BUDGET_S:-5400}"
mkdir -p logs

TS="$(date -u +%Y%m%dT%H%M%SZ)"
SAFE_STAGE="$(echo "${STAGES//,/+}" | tr -c 'A-Za-z0-9._+-' '-')"
HASH="$(sha256sum "$CONFIG_PATH" | cut -c1-8)"
RUN_ID="${TS}-${SAFE_STAGE}-${HASH}"
LOG="$ROOT/logs/${RUN_ID}.log"
CNAME="lexsi-full-${TS}"
echo "[$(date -Is)] waiting for the GPU (no other lexsi container) run_id=$RUN_ID" >> "$LOG"
deadline=$(( $(date +%s) + MAX_WAIT_S ))
while [[ "$(docker ps -q --filter name=lexsi- | wc -l | tr -d ' ')" != "0" ]]; do
  if (( $(date +%s) > deadline )); then echo "[$(date -Is)] gave up waiting for the GPU" >> "$LOG"; exit 1; fi
  sleep 10
done
echo "[$(date -Is)] starting run_id=$RUN_ID stage=$STAGES config=$CONFIG_PATH resume=$RESUME_RESULTS" >> "$LOG"
trap 'docker stop -t 30 "$CNAME" >/dev/null 2>&1; docker stop -t 30 "${CNAME}-sweep" >/dev/null 2>&1; echo "[$(date -Is)] stopped by signal" >> "$LOG"; exit 143' TERM INT

drun() { # name, then command
  local name="$1"; shift
  docker run --rm --init --name "$name" --gpus all --ipc=host --network host --env-file .env \
    -e JUDGE_MAX_SECONDS="$JUDGE_BUDGET_S" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -v "$PWD:/workspace" -w /workspace \
    "$IMAGE" "$@" >> "$LOG" 2>&1 &
  wait $!
}

drun "$CNAME" python scripts/run_pipeline.py --config "$CONFIG_PATH" --stage "$STAGES" --run-name "$RUN_ID" \
  --resume-from-results "$RESUME_RESULTS" --resume-traces "$RESUME_TRACES"
rc=$?
echo "[$(date -Is)] finished exit=$rc" >> "$LOG"
if [[ "$RUN_SWEEP" == "1" && $rc -eq 0 ]]; then
  echo "[$(date -Is)] starting sweep (scripts/sweep.py, default combinations)" >> "$LOG"
  drun "${CNAME}-sweep" python scripts/sweep.py --config "$CONFIG_PATH"
  src=$?
  echo "[$(date -Is)] sweep finished exit=$src" >> "$LOG"
fi
exit $rc
