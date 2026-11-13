# Running on the NVIDIA DGX Spark

The Spark is the intended GPU box for this repo: 128 GB unified memory, GB10, ARM64/aarch64, reachable over Tailscale MagicDNS. Do not install the training stack or run training on the Mac.

Fill `.env` on this Mac first (`SPARK_HOST`, `SPARK_USER`, `SPARK_SSH_PORT`, `SPARK_PROJECT_DIR`, plus `HF_TOKEN`). Then copy the project over and create a **separate** `.env` on the Spark (or `scp` it once). `.env` is gitignored and is excluded from `scripts/sync_to_spark.sh`.

## 1. Copy the project over Tailscale

```bash
# On the Mac, after .env is filled:
# Review the script, then run it yourself (agents should not run it by default).
./scripts/sync_to_spark.sh

# .env is not included. Copy it securely, or ssh in and write it on the Spark:
scp -P "$SPARK_SSH_PORT" .env "$SPARK_USER@$SPARK_HOST:$SPARK_PROJECT_DIR/.env"
```

MagicDNS names look like `spark.tailXXXX.ts.net`. SSH as usual: `ssh -p $SPARK_SSH_PORT $SPARK_USER@$SPARK_HOST`.

## 2. PyTorch on the Spark: prefer the NGC container

SafeTune needs **Python ≥ 3.12**. The NVIDIA PyTorch containers ship that.

### What was verified (2026-11-12)

- Official NGC PyTorch catalog lists **Latest: `26.10-py3`**, updated 29 October 2026: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
- Official 26.10 release notes: Ubuntu 24.04, **Python 3.12**, CUDA 13.4.3, image `nvcr.io/nvidia/pytorch:26.10-py3`. Notes: https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-26-10.html
- The official **DGX Spark NGC collection** lists the same PyTorch artifact, “PyTorch 26.10-py3”: https://catalog.ngc.nvidia.com/orgs/nvidia/-/collections/dgx-spark/-/artifacts
- Official run pattern (framework release notes): `docker run --gpus all -it --rm -v local_dir:container_dir nvcr.io/nvidia/pytorch:<xx.xx>-py3`, plus `--ipc=host` (or `--shm-size`) for dataloader shared memory. https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/running.html
- Official Spark NGC login: username is literally `$oauthtoken`, personal API key with the NGC Catalog service. Spark/GB10 systems do not use device serial-number registration. https://docs.nvidia.com/dgx/dgx-spark/ngc.html
- NVIDIA Container Toolkit is preinstalled on DGX Spark; `--gpus=all` exposes the GPU. https://docs.nvidia.com/dgx/dgx-spark/nvidia-container-runtime-for-docker.html

### Caveat — stale tag in the Spark user guide

The official DGX Spark user guide examples still pull **`nvcr.io/nvidia/pytorch:24.08-py3`**. That tag is older (CUDA 12.6-era) and is **not** what the NGC catalog currently marks as Latest. Prefer `26.10-py3` unless NVIDIA’s Spark collection moves to a newer tag. Always re-check the catalog before a long run.

### Not verified here (do this on the Spark)

- This setup was not tested by pulling the image or inspecting the `linux/arm64` manifest from this Mac. On the Spark, confirm:

  ```bash
  docker manifest inspect nvcr.io/nvidia/pytorch:26.10-py3
  # look for a linux/arm64 (or linux/arm64/v8) entry
  ```

- Third-party reports (e.g. 7 November 2026) say 26.10-py3 has an arm64 build and that NVIDIA dropped separate `-igpu` tags after 26.02 / 26.03. Treat that as supporting context, not a substitute for the manifest check. Official 26.10 notes do say standalone iGPU containers ended at 26.03, so use the plain `26.10-py3` tag.

### Login, pull, and run

```bash
# NGC personal API key. Username is the literal string $oauthtoken
echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin

docker pull nvcr.io/nvidia/pytorch:26.10-py3

# Confirm Python 3.12 and a working GPU once the container starts:
docker run --gpus all -it --rm --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$SPARK_PROJECT_DIR:/workspace" -w /workspace \
  nvcr.io/nvidia/pytorch:26.10-py3 \
  python -c "import sys,torch; print(sys.version, torch.__version__, torch.cuda.is_available())"
```

The container ships a pip constraints file at `/etc/pip/constraint.txt` that can block replacing `torch`. Leave torch alone; install SafeTune and the rest of `requirements.txt` inside the container (or a venv on top of it). If pip refuses a package, edit that constraints file as described in the 26.10 release notes rather than force-reinstalling torch.

## 3. Alternative: uv / venv on the host (Python ≥ 3.12)

Use this if you would rather not run inside the container. **Install an official ARM64/CUDA 13 PyTorch build** — do not pip-install a default x86 wheel.

```bash
# On the Spark, from the project directory
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate

# TODO: confirm the current official aarch64 / CUDA 13 wheel index on the Spark
# (PyTorch or NVIDIA docs) before installing torch. Then:
uv pip install -r requirements.txt
# If uv/pip tries to pull a CPU or amd64 torch, stop and use the NGC container instead.
```

This host-venv path is less turnkey than the NGC container (compiled extras such as flash-attn are often easier inside the container). Prefer the container unless you already have a known-good ARM64 torch.

## 4. Hugging Face login (gated Tiny Aya)

Accept the license on a browser: https://huggingface.co/CohereLabs/tiny-aya-global

On the Spark (inside the container or venv):

```bash
set -a && source .env && set +a
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
huggingface-cli login --token "$HF_TOKEN"
# or: python -c "from huggingface_hub import login; import os; login(token=os.environ['HF_TOKEN'])"
```

Never put the token in a command you will paste into a ticket, log, or chat.

## 5. Run in tmux

```bash
ssh -p "$SPARK_SSH_PORT" "$SPARK_USER@$SPARK_HOST"
tmux new -s lexsi
# inside the container or venv, from the project dir:
python scripts/run_pipeline.py --config configs/spark.yaml --stage all
# detach: Ctrl-b d    reattach: tmux attach -t lexsi
```

Useful first commands:

```bash
python scripts/run_pipeline.py --config configs/spark.yaml --stage inject,verify
python scripts/run_pipeline.py --config configs/spark.yaml --stage unlearn,eval
python scripts/sweep.py --config configs/spark.yaml --dry-run
python scripts/sweep.py --config configs/spark.yaml
```

Set `injection.method: full` in `configs/spark.yaml` only if you want a full-parameter TOFU fine-tune instead of LoRA. 3.35B in bf16 plus optimizer state should fit in 128 GB; drop `batch_size` if you switch to `CohereLabs/aya-expanse-8b` (`stretch_model.enabled`, finalist-week / late stretch only).

## 6. Pull results back

```bash
# On the Mac, after filling .env:
./scripts/sync_from_spark.sh
# copies $SPARK_PROJECT_DIR/$OUTPUT_DIR/ -> ./artifacts/ (or $OUTPUT_DIR)
```

Metrics JSON lands under `$OUTPUT_DIR/results/` (see `paths.results_dir` in the Spark config).

## 7. Helper judge (optional)

`src/lexsi_unlearn/judge.py` is a **helper only** (grader / paraphraser / translator), not the core unlearning model.

The primary backend is Ollama on the Spark (`JUDGE_BACKEND=ollama`, default `JUDGE_MODEL=qwen3.5:35b`, `OLLAMA_BASE_URL=http://127.0.0.1:11434/v1`). Pipeline `docker run` commands use `--network host` so that address works inside the container (verified with `GET /api/tags`). If you run without `--network host`, add `--add-host=host.docker.internal:host-gateway` and set `OLLAMA_BASE_URL=http://host.docker.internal:11434/v1`.

An optional secondary OpenRouter judge runs only when `OPENROUTER_API_KEY` and `SECONDARY_JUDGE_MODEL` are both set. It grades a seeded sample (`SECONDARY_JUDGE_FRACTION`, default 0.15) of the same cases. Leave `SECONDARY_JUDGE_MODEL` blank until you pick a slug from https://openrouter.ai/models.

Enable with `eval.use_judge: true` in `configs/spark.yaml`. Judge scores are written into Evalboard traces; failures are missing, not zero.
