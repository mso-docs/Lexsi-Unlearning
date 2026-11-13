#!/usr/bin/env bash
# Overnight job queue for the DGX Spark. Runs ON the Spark, launched by the systemd user timer
# lexsi-overnight-queue.timer (see scripts/schedule_overnight_queue.sh).
#
# - Waits until the full-run chain (smoke+translate, then lexsi-full-after-smoke: pipeline + RMU sweep)
#   is finished and no lexsi container holds the GPU. Never starts anything before FLOOR (17:05 ET).
# - Runs jobs from scripts/overnight_queue.jobs ONE AT A TIME (a single docker container at a time).
# - Before each job it picks the first job in priority order whose estimated end (+ margin) is before the
#   DEADLINE (07:30 ET); estimates are rescaled from the measured duration of finished jobs.
# - At the DEADLINE it `docker stop`s the running job (grace period), keeps partial output, logs it and exits.
# - Every job: its own run_id <UTC start time>-<stages>-<config hash> (stamped at real start), own log
#   logs/<run_id>.log, own OUTPUT_DIR artifacts/queue/<slug>/ (checkpoints, traces, interp, results).
# - A failing job is logged and the queue moves on.
#
# Modes:  (no args)          run the queue
#         --plan             simulate only: print the plan, start/end times, skips (no docker, no waiting)
#         TEST_MODE=1        fake jobs (sleep) with no GPU, no floor, no chain wait. Never used by the unit.
# Env:    FLOOR_LOCAL, DEADLINE_LOCAL, CHAIN_UNITS, JOBS_FILE, MARGIN_MIN, STOP_GRACE_S, CLEAN_FREE_GB,
#         SIM_START_LOCAL (for --plan), FAKE_SPEC="slug=sleep_s:rc ..." (TEST_MODE)
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export TZ="${TZ:-America/New_York}"

FLOOR_LOCAL="${FLOOR_LOCAL:-2026-11-12 17:05:00}"
DEADLINE_LOCAL="${DEADLINE_LOCAL:-2026-11-13 07:30:00}"
CHAIN_UNITS="${CHAIN_UNITS:-lexsi-full-after-smoke.service lexsi-20261112T220000Z-smoke-translate--64148c72-.service}"
JOBS_FILE="${JOBS_FILE:-$ROOT/scripts/overnight_queue.jobs}"
MARGIN_MIN="${MARGIN_MIN:-10}"
STOP_GRACE_S="${STOP_GRACE_S:-45}"
POLL_S="${POLL_S:-15}"
CLEAN_FREE_GB="${CLEAN_FREE_GB:-500}"         # delete finished jobs' bulky checkpoints when free disk drops below this
JUDGE_BUDGET_S="${JUDGE_BUDGET_S:-5400}"      # same per-batch judge cap as the main full run
IMAGE="${IMAGE:-lexsi-unlearning:latest}"
HF_CACHE="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"
TEST_MODE="${TEST_MODE:-0}"
MODE="run"; [[ "${1:-}" == "--plan" ]] && MODE="plan"

mkdir -p logs
STAMP="$(date +%Y%m%dT%H%M%S)"
PFX="overnight-queue"; [[ "$TEST_MODE" == "1" ]] && PFX="overnight-queue-TEST"
QLOG="$ROOT/logs/${PFX}-${STAMP}.log"
INDEX="$ROOT/logs/${PFX/-queue/-queue-index}.tsv"
[[ "$MODE" == "plan" ]] && QLOG=/dev/null
log() { echo "[$(date -Is)] $*" | tee -a "$QLOG" >&2; }

deadline_epoch=$(date -d "$DEADLINE_LOCAL" +%s)
floor_epoch=$(date -d "$FLOOR_LOCAL" +%s)
SIM_NOW=""
now_epoch() { if [[ -n "$SIM_NOW" ]]; then echo "$SIM_NOW"; else date +%s; fi; }
hhmm() { date -d "@$1" "+%a %H:%M"; }

# ---------- jobs ----------
declare -a J_SLUG J_KIND J_CFG J_STAGES J_EST J_SEED J_CAL J_WHY J_SEC
load_jobs() {
  local line
  while IFS= read -r line; do
    [[ -z "${line// }" || "$line" == \#* ]] && continue
    IFS='|' read -r s k c st e sd cal why sec <<<"$line"
    J_SLUG+=("$s"); J_KIND+=("$k"); J_CFG+=("$c"); J_STAGES+=("$st"); J_EST+=("$e"); J_SEED+=("$sd"); J_CAL+=("$cal"); J_WHY+=("$why"); J_SEC+=("$sec")
  done < "$JOBS_FILE"
}
load_jobs
N=${#J_SLUG[@]}
declare -a J_STATE; for ((i=0;i<N;i++)); do J_STATE[i]=pending; done
SCALE="1.0"; CALIBRATED=0

est_s() { awk -v e="$1" -v s="$SCALE" 'BEGIN{printf "%d", e*s*60}'; }

PICK=""
pick_next() {   # sets PICK to the index of the first pending job that fits before the deadline; returns 1 if none
  local i now need
  now=$(now_epoch)
  for ((i=0;i<N;i++)); do
    [[ "${J_STATE[i]}" == pending ]] || continue
    need=$(( $(est_s "${J_EST[i]}") + MARGIN_MIN*60 ))
    if (( now + need <= deadline_epoch )); then PICK="$i"; return 0; fi
    J_STATE[i]="skipped:does-not-fit"
    log "skip ${J_SLUG[i]}: est $(est_s "${J_EST[i]}")s (+${MARGIN_MIN}m margin) from $(hhmm "$now") ends $(hhmm $((now+need))) after deadline $(hhmm "$deadline_epoch")"
  done
  return 1
}

# ---------- plan simulation ----------
if [[ "$MODE" == "plan" ]]; then
  SIM_NOW=$(date -d "${SIM_START_LOCAL:-2026-11-13 00:45:00}" +%s)
  echo "Plan (uncalibrated nominal estimates, scale=$SCALE). Queue start $(hhmm "$SIM_NOW"), deadline $(hhmm "$deadline_epoch") ET."
  printf '%-12s %-8s %-18s %-18s %s\n' slug kind est_start est_end note
  while pick_next 2>/dev/null; do
    idx=$PICK
    d=$(est_s "${J_EST[idx]}")
    printf '%-12s %-8s %-18s %-18s %s\n' "${J_SLUG[idx]}" "${J_KIND[idx]}" "$(hhmm "$SIM_NOW")" "$(hhmm $((SIM_NOW+d)))" "${J_WHY[idx]:0:70}"
    SIM_NOW=$((SIM_NOW+d)); J_STATE[idx]=done
  done
  for ((i=0;i<N;i++)); do [[ "${J_STATE[i]}" == pending || "${J_STATE[i]}" == skipped* ]] && printf '%-12s %-8s %-18s %-18s %s\n' "${J_SLUG[i]}" "${J_KIND[i]}" "-" "-" "NOT RUN: does not fit (est ${J_EST[i]} min)"; done
  exit 0
fi

# ---------- helpers ----------
CNAME=""
CUR_LOG=""
stop_current() {
  [[ -n "$CNAME" ]] || return 0
  if docker ps -q --filter "name=^${CNAME}$" 2>/dev/null | grep -q .; then
    docker stop -t "$STOP_GRACE_S" "$CNAME" >/dev/null 2>&1
  fi
}
on_signal() {
  log "signal received: stopping current job container ${CNAME:-none}"
  [[ -n "$CUR_LOG" ]] && echo "[$(date -Is)] queue stopped by signal; container stopped, partial results kept" >> "$CUR_LOG"
  stop_current
  exit 143
}
trap on_signal TERM INT

gpu_busy() {
  local u active busy
  for u in $CHAIN_UNITS; do
    active=$(systemctl --user is-active "$u" 2>/dev/null || true)
    [[ "$active" == active || "$active" == activating || "$active" == deactivating ]] && return 0
  done
  busy=$(docker ps -q --filter "ancestor=$IMAGE" 2>/dev/null | wc -l | tr -d ' ')
  [[ "$busy" != 0 ]] && return 0
  busy=$(docker ps -q --filter "name=lexsi-" 2>/dev/null | wc -l | tr -d ' ')
  [[ "$busy" != 0 ]] && return 0
  return 1
}

wait_for_idle() {
  local idle=0 now
  log "waiting for the full-run chain to finish (units: $CHAIN_UNITS), floor=$FLOOR_LOCAL, deadline=$DEADLINE_LOCAL"
  while :; do
    now=$(date +%s)
    if (( now >= deadline_epoch )); then log "deadline reached while waiting; nothing started"; return 1; fi
    if [[ "$TEST_MODE" != "1" ]] && (( now < floor_epoch )); then idle=0; sleep 30; continue; fi
    if [[ "$TEST_MODE" == "1" ]] || ! gpu_busy; then idle=$((idle+1)); else idle=0; fi
    if (( idle >= 3 )); then return 0; fi    # idle on three consecutive polls
    sleep "$POLL_S"
  done
}

free_gb() { df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9'; }

cleanup_job_dir() {   # slug: drop bulky temp checkpoints of a finished job when disk is getting low
  local slug="$1" free
  free=$(free_gb)
  if (( free < CLEAN_FREE_GB )); then
    log "disk low (${free} GB free < ${CLEAN_FREE_GB}): removing rmu_unlearned/relearn_new of $slug"
    docker run --rm --entrypoint bash -v "$ROOT:/workspace" "$IMAGE" -c "rm -rf /workspace/artifacts/queue/$slug/rmu_unlearned /workspace/artifacts/queue/$slug/relearn_new" >/dev/null 2>&1
  fi
  # LoRA epoch checkpoints (with optimizer state) are never needed after the final adapter is saved.
  docker run --rm --entrypoint bash -v "$ROOT:/workspace" "$IMAGE" -c "rm -rf /workspace/artifacts/queue/$slug/inject_lora/checkpoint-*" >/dev/null 2>&1 || true
}

fake_spec_for() { # slug -> "sleep:rc"
  local slug="$1" item
  for item in ${FAKE_SPEC:-}; do [[ "${item%%=*}" == "$slug" ]] && { echo "${item#*=}"; return; }; done
  echo "5:0"
}

run_job() {   # index
  local i="$1" slug kind cfg stages seed ts safe hash run_id logf out start_ep end_ep rc status dur_min cmd_desc
  slug="${J_SLUG[i]}"; kind="${J_KIND[i]}"; cfg="${J_CFG[i]}"; stages="${J_STAGES[i]}"; seed="${J_SEED[i]}"
  ts="$(date -u +%Y%m%dT%H%M%SZ)"                      # REAL start time
  hash="$(sha256sum "$cfg" | cut -c1-8)"
  if [[ "$kind" == sweep ]]; then safe="sweep-extra"; else safe="$(echo "${stages//,/+}" | tr -c 'A-Za-z0-9._+-' '-')"; fi
  run_id="${ts}-${safe}-${hash}"
  logf="$ROOT/logs/${run_id}.log"; CUR_LOG="$logf"
  out="artifacts/queue/$slug"
  CNAME="lexsi-q-${slug}-${ts}"
  echo "[$(date -Is)] starting queue job slug=$slug kind=$kind run_id=$run_id model_cfg=$cfg stages=${stages:-sweep} seed=${seed:-default} OUTPUT_DIR=$out est=$(est_s "${J_EST[i]}")s deadline=$DEADLINE_LOCAL" | tee -a "$logf" >> "$QLOG"
  log "START $slug run_id=$run_id (est $(( $(est_s "${J_EST[i]}") / 60 )) min, scale $SCALE) log=$logf"

  # Secondary (OpenRouter) judge: blanked for every job EXCEPT those flagged `secondary` in the jobs file (9th field),
  # for which SECONDARY_JUDGE_MODEL / SECONDARY_JUDGE_FRACTION / OPENROUTER_API_KEY / OPENROUTER_BUDGET_USD come from .env.
  # All OpenRouter spend is capped by src/lexsi_unlearn/spend.py against ONE shared ledger (logs/openrouter_spend.jsonl, the same
  # host file the main run writes, since every container mounts the repo at /workspace).
  local -a envs=(-e "OUTPUT_DIR=$out" -e "JUDGE_MAX_SECONDS=$JUDGE_BUDGET_S")
  local sec="${J_SEC[i]:-}"
  if [[ "$sec" == secondary ]]; then
    envs+=(-e "OPENROUTER_SPEND_LEDGER=/workspace/logs/openrouter_spend.jsonl")
    log "secondary judge ENABLED for $slug (from .env); $(python3 scripts/spend_report.py 2>/dev/null | tail -1 || echo 'spend report unavailable')"
  else
    envs+=(-e "SECONDARY_JUDGE_MODEL=")
  fi
  [[ -n "$seed" ]] && envs+=(-e "LEXSI_SEED=$seed")
  local -a cmd
  local -a gpu=(--gpus all)
  if [[ "$TEST_MODE" == "1" ]]; then
    local fs sl frc; fs="$(fake_spec_for "$slug")"; sl="${fs%%:*}"; frc="${fs##*:}"
    gpu=()
    cmd=(python -c "import os,sys,time; print('FAKE job OUTPUT_DIR=',os.environ['OUTPUT_DIR'],'LEXSI_SEED=',os.environ.get('LEXSI_SEED'),'SECONDARY_JUDGE_MODEL=',repr(os.environ.get('SECONDARY_JUDGE_MODEL')),flush=True); time.sleep(float(sys.argv[1])); sys.exit(int(sys.argv[2]))" "$sl" "$frc")
  elif [[ "$kind" == sweep ]]; then
    cmd=(python scripts/sweep.py --config "$cfg")
  else
    cmd=(python scripts/run_pipeline_seeded.py --config "$cfg" --stage "$stages" --run-name "$run_id")
  fi

  start_ep=$(date +%s)
  docker run --rm --init --name "$CNAME" "${gpu[@]}" --ipc=host --network host --env-file .env "${envs[@]}" \
    -v "$HF_CACHE:/root/.cache/huggingface" -v "$ROOT:/workspace" -w /workspace \
    "$IMAGE" "${cmd[@]}" >> "$logf" 2>&1 &
  local pid=$!
  status=""
  while kill -0 "$pid" 2>/dev/null; do
    if (( $(date +%s) >= deadline_epoch )); then
      log "DEADLINE $DEADLINE_LOCAL reached: stopping $slug (docker stop -t $STOP_GRACE_S $CNAME); partial output kept in $out"
      echo "[$(date -Is)] DEADLINE reached: docker stop -t $STOP_GRACE_S $CNAME (partial results kept; traces in $out/traces/ are appended as stages finish)" >> "$logf"
      stop_current
      status="deadline-stopped"
      break
    fi
    sleep 5
  done
  wait "$pid"; rc=$?
  end_ep=$(date +%s)
  [[ -z "$status" ]] && { [[ $rc -eq 0 ]] && status="ok" || status="failed"; }
  dur_min=$(awk -v s="$start_ep" -v e="$end_ep" 'BEGIN{printf "%.1f", (e-s)/60}')
  echo "[$(date -Is)] finished exit=$rc status=$status duration_min=$dur_min" >> "$logf"
  log "END $slug status=$status exit=$rc duration=${dur_min} min (est ${J_EST[i]} nominal)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$run_id" "$slug" "${cfg}" "${seed:-default}" "$status" "$rc" \
    "$(date -d "@$start_ep" -Is)" "$(date -d "@$end_ep" -Is)" "$dur_min" "$out" >> "$INDEX"
  J_STATE[i]="$status"
  CNAME=""; CUR_LOG=""
  cleanup_job_dir "$slug"
  # Calibrate remaining estimates from measured duration of a successful comparable job.
  if [[ "$status" == ok && "${J_CAL[i]}" == 1 ]]; then
    SCALE=$(awk -v d="$dur_min" -v e="${J_EST[i]}" -v s="$SCALE" -v c="$CALIBRATED" 'BEGIN{r=d/e; if(c>0) r=0.5*s+0.5*r; if(r<0.6) r=0.6; if(r>3) r=3; printf "%.3f", r}')
    CALIBRATED=1
    log "calibration: remaining estimates scaled x$SCALE (measured ${dur_min} min vs nominal ${J_EST[i]})"
  fi
  [[ "$status" == deadline-stopped ]] && return 99
  return 0
}

precheck() {   # index -> 0 ok / 1 skip
  local i="$1"
  if [[ "${J_KIND[i]}" == sweep && "$TEST_MODE" != "1" ]]; then
    if ! ls artifacts/inject_lora/adapter_model.safetensors >/dev/null 2>&1; then
      log "skip ${J_SLUG[i]}: main run's artifacts/inject_lora/adapter_model.safetensors not found"; return 1
    fi
  fi
  [[ -f "${J_CFG[i]}" ]] || { log "skip ${J_SLUG[i]}: config ${J_CFG[i]} missing"; return 1; }
  return 0
}

# ---------- main ----------
log "queue start. jobs=$N file=$JOBS_FILE index=$INDEX test_mode=$TEST_MODE"
wait_for_idle || exit 0
log "GPU idle; starting queue. free disk: $(free_gb) GB"
while pick_next; do
  idx=$PICK
  if ! precheck "$idx"; then J_STATE[idx]="skipped:precheck"; continue; fi
  run_job "$idx"; r=$?
  [[ $r -eq 99 ]] && { log "deadline stop complete; queue exiting"; break; }
  (( $(date +%s) >= deadline_epoch )) && { log "deadline passed; queue exiting"; break; }
done
log "queue finished. Summary:"
for ((i=0;i<N;i++)); do st="${J_STATE[i]}"; [[ "$st" == pending ]] && st="not started (deadline reached or queue stopped)"; log "  ${J_SLUG[i]}: $st"; done
exit 0
