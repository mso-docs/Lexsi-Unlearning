#!/usr/bin/env python3
"""Pre-flight for the overnight queue: every job config resolves, every model's config/tokenizer is
cached, and the chat-template situation is reported. Loads NO model weights and runs no inference.

    docker run --rm --env-file .env -v $HOME/.cache/huggingface:/root/.cache/huggingface \
        -v $PWD:/workspace -w /workspace lexsi-unlearning:latest python scripts/queue_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lexsi_unlearn.data import apply_chat_template, load_config  # noqa: E402


def main() -> int:
    bad = 0
    for path in sorted((ROOT / "configs" / "queue").glob("*.yaml")):
        try:
            cfg = load_config(path)
            mid = cfg["model"]["id"]
            from transformers import AutoConfig, AutoTokenizer

            mc = AutoConfig.from_pretrained(mid)
            tok = AutoTokenizer.from_pretrained(mid)
            tmpl = bool(getattr(tok, "chat_template", None))
            sample = apply_chat_template(tok, "Q?", add_generation_prompt=True)
            n_cfg = cfg["model"].get("n_layers")
            ok_layers = n_cfg == mc.num_hidden_layers
            bad += 0 if ok_layers else 1
            paths = cfg["paths"]
            print(
                f"{path.name:18s} model={mid} arch={mc.architectures[0]} layers={mc.num_hidden_layers} "
                f"(cfg n_layers={n_cfg}{'' if ok_layers else ' MISMATCH'}) d_model={mc.hidden_size} chat_template={tmpl} "
                f"rmu.layer_id={cfg['rmu']['layer_id']} seed={(cfg.get('queue') or {}).get('seed')} "
                f"judge={cfg['eval'].get('use_judge')} inject_dir={paths['inject_adapter_dir']}"
            )
            if not tmpl:
                print(f"    WARNING no chat template; eval prompt falls back to raw text: {sample!r}")
        except Exception as exc:  # noqa: BLE001
            bad += 1
            print(f"{path.name:18s} FAILED: {type(exc).__name__}: {exc}")
    t = ROOT / "data" / "translations.json"
    print("data/translations.json:", "present" if t.exists() else "MISSING (created by the 5 PM smoke+translate run)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
