#!/usr/bin/env python3
"""RMU hyperparameter sweep for 36-layer Tiny Aya (Spark profile).

Reads layer_id / alpha / steering_coeff lists from configs/spark.yaml (or
--config). By default runs rmu.sweep.default_combinations (a short first pass).
Pass --grid to take the Cartesian product of layer_ids x alphas x steering_coeffs.

Each combination reuses the already-injected checkpoint, runs RMU, then a
direct forget/retain eval (substring, all items) plus the primary judge on a seeded subset
(rmu.sweep.judge_n_forget / judge_n_retain, default 20 + 20; judge_forget_suppression and
judge_retain_accuracy), and writes one JSON under artifacts/results/sweeps/.

    python scripts/sweep.py --config configs/spark.yaml
    python scripts/sweep.py --grid
    python scripts/sweep.py --dry-run

Run on the Spark (or another GPU machine), not the Mac.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lexsi_unlearn.data import (  # noqa: E402
    build_direct_queries,
    default_config_path,
    load_config,
    load_dotenv_if_available,
    load_tofu_split,
    qa_pairs,
)
from lexsi_unlearn.evaluate import evaluate_model_on_items, summarize_forget_retain  # noqa: E402
from lexsi_unlearn.scrub import scrub_obj  # noqa: E402
from lexsi_unlearn.inject import load_base_model  # noqa: E402
from lexsi_unlearn.unlearn import rmu_hparams_from_config, run_rmu, tokenize_unlearn_batches  # noqa: E402


def update_layers_for(layer_id: int, n_layers: int = 36) -> list[int]:
    """Default RMU window: [id-2, id-1, id], clipped to the model depth."""
    return [i for i in (layer_id - 2, layer_id - 1, layer_id) if 0 <= i < n_layers]


def combinations(cfg: dict, *, grid: bool) -> list[dict]:
    sweep = (cfg.get("rmu") or {}).get("sweep") or {}
    if not grid:
        combos = list(sweep.get("default_combinations") or [])
        if combos:
            return combos
        print("No rmu.sweep.default_combinations; falling back to --grid lists.")
    layer_ids = sweep.get("layer_ids") or [cfg["rmu"]["layer_id"]]
    alphas = sweep.get("alphas") or [cfg["rmu"]["alpha"]]
    coeffs = sweep.get("steering_coeffs") or [cfg["rmu"]["steering_coeff"]]
    return [
        {"layer_id": lid, "alpha": alpha, "steering_coeff": coeff}
        for lid, alpha, coeff in itertools.product(layer_ids, alphas, coeffs)
    ]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="YAML path (default: configs/spark.yaml if RUN_PROFILE=spark).")
    p.add_argument("--grid", action="store_true", help="Full Cartesian product of the sweep lists.")
    p.add_argument("--dry-run", action="store_true", help="Print combinations and exit.")
    p.add_argument("--max-runs", type=int, default=None, help="Optional cap on the number of combinations.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    load_dotenv_if_available(ROOT)
    os.environ.setdefault("LEXSI_RUN_ID", time.strftime("sweep-%Y%m%dT%H%M%S"))  # OpenRouter spend ledger run_id
    args = parse_args(argv)
    cfg_path = Path(args.config) if args.config else default_config_path()
    if not cfg_path.is_absolute():
        cfg_path = (Path.cwd() / cfg_path).resolve()
    cfg = load_config(cfg_path)
    combos = combinations(cfg, grid=args.grid)
    if args.max_runs is not None:
        combos = combos[: args.max_runs]
    n_layers = int(cfg["model"].get("n_layers", 36))
    print(f"{len(combos)} RMU combinations from {cfg_path}:")
    for i, combo in enumerate(combos, 1):
        layers = update_layers_for(int(combo["layer_id"]), n_layers)
        print(f"  {i:2d}. layer_id={combo['layer_id']} update_layers={layers} "
              f"alpha={combo['alpha']} steering_coeff={combo['steering_coeff']}")
    if args.dry_run:
        return 0

    inj_dir = cfg["paths"]["inject_adapter_dir"]
    if not Path(inj_dir).exists():
        raise SystemExit(
            f"Injected checkpoint not found at {inj_dir}. "
            "Run `python scripts/run_pipeline.py --stage inject,verify` first."
        )

    tofu = cfg["tofu"]
    ev = cfg.get("eval") or {}
    max_new = int(ev.get("max_new_tokens", 64))
    max_eval = ev.get("max_eval_examples")
    forget_ds = load_tofu_split(tofu["forget_split"], repo_id=tofu["repo_id"], max_examples=cfg["injection"].get("max_forget_examples"))
    retain_ds = load_tofu_split(tofu["retain_split"], repo_id=tofu["repo_id"], max_examples=cfg["injection"].get("max_retain_examples"))
    forget_pairs = qa_pairs(forget_ds)[: max_eval or None]
    retain_pairs = qa_pairs(retain_ds)[: max_eval or None]

    out_root = Path(cfg["paths"].get("results_dir") or "artifacts/results") / "sweeps"
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for i, combo in enumerate(combos, 1):
        run_cfg = copy.deepcopy(cfg)
        run_cfg["rmu"]["layer_id"] = int(combo["layer_id"])
        run_cfg["rmu"]["alpha"] = float(combo["alpha"])
        run_cfg["rmu"]["steering_coeff"] = float(combo["steering_coeff"])
        layers = update_layers_for(int(combo["layer_id"]), n_layers)
        run_cfg["rmu"]["update_layers"] = layers
        run_cfg["rmu"]["update_layer_ids"] = layers
        tag = (
            f"layer{run_cfg['rmu']['layer_id']}"
            f"_a{run_cfg['rmu']['alpha']:g}"
            f"_s{run_cfg['rmu']['steering_coeff']:g}"
        )
        print(f"\n=== sweep {i}/{len(combos)} {tag} ===")
        model, tokenizer = load_base_model(inj_dir, dtype=cfg["model"].get("dtype", "bfloat16"))
        forget_batches = tokenize_unlearn_batches(tokenizer, forget_ds, batch_size=int(cfg["injection"].get("batch_size", 2)))
        retain_batches = tokenize_unlearn_batches(tokenizer, retain_ds, batch_size=int(cfg["injection"].get("batch_size", 2)))
        hparams = rmu_hparams_from_config(run_cfg)
        unlearned = run_rmu(model, forget_batches=forget_batches, retain_batches=retain_batches, hparams=hparams, tokenizer=tokenizer)
        forget_scores = evaluate_model_on_items(
            unlearned, tokenizer, build_direct_queries(forget_pairs), max_new_tokens=max_new
        )
        retain_scores = evaluate_model_on_items(
            unlearned, tokenizer, build_direct_queries(retain_pairs), max_new_tokens=max_new
        )
        judged = None
        sw_cfg = (cfg.get("rmu") or {}).get("sweep") or {}
        if (cfg.get("eval") or {}).get("use_judge") and sw_cfg.get("judge", True):
            from lexsi_unlearn.judge import judge_subset

            judged = judge_subset(
                forget_scores, retain_scores, cfg=cfg, tag=tag,
                n_forget=int(sw_cfg.get("judge_n_forget", 20)), n_retain=int(sw_cfg.get("judge_n_retain", 20)),
                seed=int(sw_cfg.get("judge_seed", 20261112)),
            )
            print(f"  judged subset {tag}: judge_forget_suppression={judged.get('judge_forget_suppression')} "
                  f"judge_retain_accuracy={judged.get('judge_retain_accuracy')} error={judged.get('error')}")
        row = {
            "tag": tag,
            "judge_subset": judged,
            "hparams": hparams,
            "rmu_report": getattr(unlearned, "rmu_report", None),
            "update_layers": run_cfg["rmu"]["update_layers"],
            "scores": summarize_forget_retain(forget=forget_scores, retain=retain_scores, label=tag),
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        # Drop bulky generation lists from the sweep table; keep a short sample.
        row["forget_sample"] = list(zip(
            [p["question"] for p in forget_pairs[:3]],
            forget_scores.get("predictions", [])[:3],
        ))
        path = out_root / f"{tag}.json"
        path.write_text(json.dumps(scrub_obj(json.loads(json.dumps(row, default=str))), indent=2))
        print("Wrote", path, row["scores"])
        summaries.append({"tag": tag, "scores": row["scores"], "hparams": hparams, "judge_subset": judged})
        # Free GPU memory between runs when possible.
        del model, unlearned, tokenizer
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    index = out_root / "sweep_index.json"
    index.write_text(json.dumps(scrub_obj(json.loads(json.dumps({"config": str(cfg_path), "runs": summaries}, default=str))), indent=2))
    print("Wrote", index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
