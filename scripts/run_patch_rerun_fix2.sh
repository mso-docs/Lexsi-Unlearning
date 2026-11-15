#!/usr/bin/env bash
# Launch the redesigned fix2 activation-patching rerun in the project container.
# Logs to logs/<run_id>.log. Intended to be started with systemd-run --user.
# Must not share the GPU with any other lexsi job.
set -euo pipefail
export TZ="${TZ:-America/New_York}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
IMAGE="${IMAGE:-lexsi-unlearning:latest}"
RUN_ID="${RUN_ID:?}"
LOG="$ROOT/logs/${RUN_ID}.log"
CNAME="lexsi-q-patch-rerun-${RUN_ID}"
mkdir -p logs artifacts/interp
touch "$LOG"
exec >>"$LOG" 2>&1
echo "[$(date -Is)] starting patch-rerun-fix2 run_id=$RUN_ID kind=patch-rerun slug=patch-rerun"
cleanup() {
  docker stop -t 30 "$CNAME" >/dev/null 2>&1 || true
  echo "[$(date -Is)] stopped by signal" || true
}
trap cleanup TERM INT

# Wait briefly if another lexsi GPU container is still exiting.
deadline=$(( $(date +%s) + 300 ))
while [[ "$(docker ps -q --filter name=lexsi- | wc -l | tr -d ' ')" != "0" ]]; do
  if (( $(date +%s) > deadline )); then
    echo "[$(date -Is)] gave up waiting for another lexsi container"
    docker ps --filter name=lexsi-
    exit 1
  fi
  sleep 5
done

docker run --rm --init --name "$CNAME" --gpus all --ipc=host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --env-file .env \
  -e OUTPUT_DIR=artifacts \
  -e PYTHONUNBUFFERED=1 \
  -e PYTHONPATH=/workspace:/workspace/src \
  -e TZ=America/New_York \
  -e HF_HUB_DISABLE_IMPLICIT_TOKEN=0 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$PWD:/workspace" -w /workspace \
  "$IMAGE" \
  python scripts/run_patch_rerun_fix2.py --run-name "$RUN_ID" "$@"
rc=$?
echo "[$(date -Is)] finished exit=$rc"
if [[ $rc -eq 0 ]]; then
  echo "$RUN_ID" > "$ROOT/logs/patch-rerun-fix2.ready-for-ingest"
  echo "[$(date -Is)] sweep ok run_id=$RUN_ID ; ingest pending on Mac (sync_from_remote + ingest_evalboard)"
fi
exit $rc
