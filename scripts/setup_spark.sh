#!/usr/bin/env bash
# Intended to be run ON the DGX Spark (Linux aarch64), not on the Mac.
# Prints environment checks and (optionally) creates a Python 3.12 uv venv.
# Does not pull the NGC image or start training unless you pass flags.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

usage() {
  cat <<'EOF'
Usage: scripts/setup_spark.sh [--check] [--venv] [--print-docker]

  --check         Print arch, Python, and (if present) nvidia-smi / docker info.
  --venv          Create .venv with uv and Python >= 3.12 (no torch install).
  --print-docker  Print the recommended docker pull/run lines (does not pull).

Refuses to run on macOS. See scripts/spark_setup.md for the full write-up,
including the official vs stale NGC tag caveat.
EOF
}

OS="$(uname -s)"
ARCH="$(uname -m)"

if [[ "$OS" == "Darwin" ]]; then
  echo "This script is for the DGX Spark (Linux aarch64), not this Mac."
  echo "Use scripts/sync_to_spark.sh from the Mac, then run setup_spark.sh on the Spark."
  echo "Docs: scripts/spark_setup.md"
  exit 1
fi

if [[ "$ARCH" != "aarch64" && "$ARCH" != "arm64" ]]; then
  echo "Unexpected architecture: $ARCH (expected aarch64). Aborting."
  echo "DGX Spark / GB10 is ARM64. Do not install x86 wheels here."
  exit 1
fi

CHECK=0
MAKE_VENV=0
PRINT_DOCKER=0
if [[ $# -eq 0 ]]; then
  CHECK=1
  PRINT_DOCKER=1
fi
while [[ $# -gt 0 ]]; do
  case "$1" in
    --check) CHECK=1 ;;
    --venv) MAKE_VENV=1 ;;
    --print-docker) PRINT_DOCKER=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown flag: $1"; usage; exit 1 ;;
  esac
  shift
done

if [[ "$CHECK" -eq 1 ]]; then
  echo "=== Spark environment check ==="
  echo "os=$OS arch=$ARCH pwd=$ROOT"
  uname -a || true
  python3 --version || true
  command -v nvidia-smi >/dev/null && nvidia-smi -L || echo "nvidia-smi not on PATH"
  command -v docker >/dev/null && docker --version || echo "docker not on PATH"
  if [[ -f .env ]]; then
    echo ".env is present (values not printed)."
  else
    echo "No .env yet. Copy .env.example and fill HF_TOKEN / OUTPUT_DIR / etc."
  fi
fi

if [[ "$PRINT_DOCKER" -eq 1 ]]; then
  cat <<'EOF'

=== Recommended NGC PyTorch container (verified as NGC "Latest" on 2026-11-12) ===
Tag: nvcr.io/nvidia/pytorch:26.10-py3
Catalog: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
Spark collection: https://catalog.ngc.nvidia.com/orgs/nvidia/-/collections/dgx-spark/-/artifacts
Release notes: https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-26-10.html

CAVEAT: The official Spark user guide still shows pytorch:24.08-py3 — that example
is stale. Prefer 26.10-py3 after confirming an arm64 manifest on this machine:

  docker manifest inspect nvcr.io/nvidia/pytorch:26.10-py3

Login, pull, run (username is the literal string $oauthtoken):

  echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin
  docker pull nvcr.io/nvidia/pytorch:26.10-py3
  docker run --gpus all -it --rm --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -v "$PWD:/workspace" -w /workspace \
    nvcr.io/nvidia/pytorch:26.10-py3

Then, inside the container: python -m pip install -r requirements.txt
(leave the container's torch alone; see /etc/pip/constraint.txt).

EOF
fi

if [[ "$MAKE_VENV" -eq 1 ]]; then
  if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found. Install from https://docs.astral.sh/uv/ then rerun --venv."
    exit 1
  fi
  echo "Creating .venv with uv (Python >= 3.12). This does NOT install torch or SafeTune."
  uv python install 3.12
  uv venv --python 3.12 .venv
  echo "Activate with: source .venv/bin/activate"
  echo "Install an official ARM64/CUDA 13 torch first, then: uv pip install -r requirements.txt"
  echo "If a CPU or amd64 wheel is selected, stop and use the NGC container instead."
  echo "See scripts/spark_setup.md."
fi
