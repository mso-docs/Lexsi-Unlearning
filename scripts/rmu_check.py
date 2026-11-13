#!/usr/bin/env python3
"""One RMU run on the EXISTING injected checkpoint (skips inject/verify), no judge, no OpenRouter, nothing saved.

    python scripts/rmu_check.py --config configs/spark.yaml [--steps 80] [--layer 18 --alpha 1200]

Prints the RMU loss curve, the weight change, and direct forget/retain ROUGE-L recall + substring for the model before
and after RMU, so a no-op (identical numbers) is visible in about 10 minutes instead of a 4-hour pipeline.
Writes artifacts/results/rmu_check-<UTC>.json. Run on the Spark in the lexsi-unlearning container.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lexsi_unlearn.data import build_direct_queries, default_config_path, load_config, load_dotenv_if_available, load_tofu_split, qa_pairs  # noqa: E402
from lexsi_unlearn.evaluate import evaluate_model_on_items  # noqa: E402
from lexsi_unlearn.inject import load_base_model  # noqa: E402
from lexsi_unlearn.unlearn import rmu_hparams_from_config, run_rmu, tokenize_unlearn_batches  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--steps", type=int, default=None, help="override rmu.max_num_batches")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--steering", type=float, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--forget-mask", default=None, help='"content" = forget loss ignores the shared system preamble')
    ap.add_argument("--skip-before", action="store_true", help="do not re-evaluate the injected model (saves ~4 min)")
    ap.add_argument("--n-eval", type=int, default=None, help="eval items per split (default eval.max_eval_examples)")
    args = ap.parse_args()
    load_dotenv_if_available()
    cfg = load_config(args.config or default_config_path())
    rc = cfg["rmu"]
    if args.layer is not None:
        rc["layer_id"] = args.layer
        rc["update_layers"] = rc["update_layer_ids"] = [i for i in (args.layer - 2, args.layer - 1, args.layer) if i >= 0]
    for key, val in (("alpha", args.alpha), ("steering_coeff", args.steering), ("lr", args.lr), ("max_num_batches", args.steps)):
        if val is not None:
            rc[key] = val
    tofu, inj = cfg["tofu"], cfg["injection"]
    n_eval = args.n_eval or (cfg.get("eval") or {}).get("max_eval_examples") or 64
    max_new = int((cfg.get("eval") or {}).get("max_new_tokens", 64))
    forget_ds = load_tofu_split(tofu["forget_split"], repo_id=tofu["repo_id"], max_examples=inj.get("max_forget_examples"))
    retain_ds = load_tofu_split(tofu["retain_split"], repo_id=tofu["repo_id"], max_examples=inj.get("max_retain_examples"))
    fq = build_direct_queries(qa_pairs(forget_ds)[:n_eval])
    rq = build_direct_queries(qa_pairs(retain_ds)[:n_eval])

    model, tok = load_base_model(cfg["paths"]["inject_adapter_dir"], dtype=cfg["model"].get("dtype", "bfloat16"))

    def score(m, label):
        f = evaluate_model_on_items(m, tok, fq, max_new_tokens=max_new)
        r = evaluate_model_on_items(m, tok, rq, max_new_tokens=max_new)
        row = {"label": label, "forget_rouge_l": f["mean_rouge_l_recall"], "forget_substring": f["substring_match"],
               "retain_rouge_l": r["mean_rouge_l_recall"], "retain_substring": r["substring_match"]}
        print(label, row, flush=True)
        return row

    before = None if args.skip_before else score(model, "injected")
    bs = int(inj.get("batch_size", 2))
    hp = rmu_hparams_from_config(cfg)
    if args.forget_mask:
        hp["forget_mask"] = args.forget_mask
    print("RMU hparams:", hp, flush=True)
    unlearned = run_rmu(model, forget_batches=tokenize_unlearn_batches(tok, forget_ds, batch_size=bs),
                        retain_batches=tokenize_unlearn_batches(tok, retain_ds, batch_size=bs), hparams=hp, tokenizer=tok)
    after = score(unlearned, "after_rmu")
    out = {"hparams": hp, "rmu_report": getattr(unlearned, "rmu_report", None), "before": before, "after": after,
           "finished_at": datetime.now(timezone.utc).isoformat()}
    path = ROOT / "artifacts" / "results" / f"rmu_check-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, default=str))
    print("Wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
