#!/usr/bin/env python3
"""End-to-end Lexsi unlearning-vs-hiding pipeline.

Run on Colab, Kaggle, HF ZeroGPU, or the DGX Spark — not on the Mac.

    python scripts/run_pipeline.py --config configs/spark.yaml --stage all
    python scripts/run_pipeline.py --stage inject
    python scripts/run_pipeline.py --config configs/default.yaml --stage eval
    python scripts/run_pipeline.py --config configs/spark.yaml --stage smoke

Stages, in order:
    inject -> verify -> unlearn -> eval -> probes -> interp -> relearn

`interp` writes Evalboard interpretability artifacts (logit lens, activation patching, neuron
explorer, residual norms) to ${OUTPUT_DIR}/interp/<run_id>/; probes/relearn add layer-wise probe curves.

`--stage translate` machine-translates the TOFU forget01 questions/answers with the helper (Ollama)
model into the languages in eval.translation_languages and writes data/translations.json (never
overwritten without --force) plus artifacts/translations_review.md for a spot-check. It loads no
Tiny Aya weights, and may be combined with any other stage (it runs first), e.g.
`--stage smoke,translate` (smoke, then translate) or `--stage translate,eval`.

`--stage all` runs every stage above (not smoke or translate). `--stage smoke` is a short
path: load Tiny Aya, generate a handful of TOFU forget01 questions, run a few
RMU steps, evaluate, and write Evalboard traces. `--stage unlearn` also runs
optional NPO when npo.enabled is true in the config.

Every stage appends canonical Evalboard traces to
`${OUTPUT_DIR}/traces/<run_id>.jsonl`.

Loads repo-root .env via python-dotenv (HF_TOKEN, OUTPUT_DIR, RUN_PROFILE, ...).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lexsi_unlearn.data import (  # noqa: E402
    apply_chat_template,
    build_direct_queries,
    build_extraction_queries,
    build_format_queries,
    build_paraphrase_queries,
    default_config_path,
    load_config,
    load_dotenv_if_available,
    load_tofu_split,
    load_translation_queries,
    qa_pairs,
)
from lexsi_unlearn.evaluate import (  # noqa: E402
    evaluate_model_on_items,
    summarize_forget_retain,
)
from lexsi_unlearn.inject import inject_facts, load_base_model  # noqa: E402
from lexsi_unlearn.logit_lens import mean_target_rank  # noqa: E402
from lexsi_unlearn.probes import build_content_probe_texts, collect_layer_states  # noqa: E402
from lexsi_unlearn import probe_stage  # noqa: E402
from lexsi_unlearn import interp as interp_compute  # noqa: E402
from lexsi_unlearn import interp_stage  # noqa: E402
from lexsi_unlearn.interp_artifacts import InterpWriter  # noqa: E402
from lexsi_unlearn import translate as translate_mod  # noqa: E402
from lexsi_unlearn.relearn import compare_relearning_curves, curve_point, relearn_step_grid, short_finetune  # noqa: E402
from lexsi_unlearn.traces import TraceWriter, evaluation, make_run_id, make_trace  # noqa: E402
from lexsi_unlearn.unlearn import (  # noqa: E402
    rmu_hparams_from_config,
    run_npo,
    run_rmu,
    tokenize_unlearn_batches,
)

STAGES = ("inject", "verify", "unlearn", "eval", "probes", "interp", "relearn")
SPECIAL_STAGES = ("smoke", "translate")
ALL_STAGES = STAGES + SPECIAL_STAGES


def _json_ready(obj):
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    headline = getattr(obj, "headline", None)
    if headline is not None:
        return {"headline": _json_ready(headline), "repr": str(obj)}
    return str(obj)


def _results_dir(cfg: dict) -> Path:
    path = Path(cfg.get("paths", {}).get("results_dir") or cfg.get("paths", {}).get("artifacts_dir") or "artifacts/results")
    if not path.is_absolute():
        path = ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def _maybe_wandb(cfg: dict, run_name: str):
    key = (os.environ.get("WANDB_API_KEY") or "").strip()
    if not key:
        return None
    try:
        import wandb  # type: ignore
    except ImportError:
        print("WANDB_API_KEY is set but the wandb package is not installed; skipping.")
        return None
    return wandb.init(project="lexsi-unlearning", name=run_name, config=cfg, dir=str(_results_dir(cfg)))


def _slice_pairs(pairs, n):
    if n is None:
        return list(pairs)
    return list(pairs)[: int(n)]


def _behavioral_items(cfg, forget_pairs, retain_pairs):
    ev = cfg.get("eval") or {}
    formats = ev.get("formats") or ["direct"]
    items = {"forget": {}, "retain": {}}
    if "direct" in formats:
        items["forget"]["direct"] = build_direct_queries(forget_pairs)
        items["retain"]["direct"] = build_direct_queries(retain_pairs)
    if "paraphrase" in formats:
        tofu = cfg["tofu"]
        para = []
        # Official perturbed split first: it has paraphrased_question.
        # TOFU-para "forget01" was observed to repeat the original question.
        for repo, split in (
            (tofu["repo_id"], tofu.get("forget_perturbed_split")),
            (tofu.get("para_repo_id"), tofu.get("forget_split")),
        ):
            if not repo or not split:
                continue
            try:
                para = build_paraphrase_queries(repo_id=repo, split=split)
                if para:
                    break
            except Exception as exc:
                print(f"Paraphrase split {repo}/{split} unavailable ({exc}); trying next.")
        items["forget"]["paraphrase"] = para
    if "format_change" in formats:
        items["forget"]["format_change"] = build_format_queries(forget_pairs)
    if "extraction" in formats:
        items["forget"]["extraction"] = build_extraction_queries(forget_pairs)
    if "translation" in formats:
        t_path = ROOT / "data" / "translations.json"
        rows = load_translation_queries(
            t_path,
            languages=ev.get("translation_languages") or None,
            include_paraphrase=bool(ev.get("translation_include_paraphrase", False)),
        )
        if not rows:
            reason = "missing" if not t_path.exists() else "has no usable rows for the configured languages"
            print(
                f"Multilingual eval skipped: {t_path.relative_to(ROOT)} is {reason}. "
                "Create it with: python scripts/run_pipeline.py --stage translate"
            )
        items["forget"]["translation"] = rows
    items.update(_utility_items(cfg))
    return items


UTILITY_SPLITS = ("world_facts", "real_authors")


def _utility_items(cfg):
    """General-capability evals: TOFU's world_facts and real_authors (never forgotten, never injected).

    Returned as their own splits ({"world_facts": {"direct": rows}, ...}) so they are scored, judged and traced
    separately from the retain slice. Off with eval.utility=false. Failures to load skip that set, never the run.
    """
    ev = cfg.get("eval") or {}
    if not ev.get("utility", True):
        return {}
    tofu = cfg["tofu"]
    cap = ev.get("utility_max_examples", 40)
    out = {}
    for name in UTILITY_SPLITS:
        split = tofu.get(f"{name}_split") or name
        try:
            ds = load_tofu_split(split, repo_id=tofu["repo_id"], max_examples=None if cap is None else int(cap))
            pairs = qa_pairs(ds)
            if pairs:
                out[name] = {"direct": build_direct_queries(pairs)}
        except Exception as exc:
            print(f"Utility set {name} ({tofu['repo_id']}/{split}) unavailable ({type(exc).__name__}: {exc}); skipped.")
    return out


def _utility_summary(block):
    """{set: substring_match} from an after/before eval block, for the results JSON."""
    return {
        name: {fmt: {"n": r.get("n"), "substring_match": r.get("substring_match")} for fmt, r in groups.items()}
        for name, groups in block.items()
        if name in UTILITY_SPLITS
    }


def _eval_group(model, tokenizer, items, max_new_tokens, label):
    out = {}
    for fmt, rows in items.items():
        if not rows:
            out[fmt] = {"n": 0, "note": "skipped: no items (for translation, run --stage translate)" if fmt == "translation" else "empty"}
            continue
        out[fmt] = evaluate_model_on_items(model, tokenizer, rows, max_new_tokens=max_new_tokens)
        # Drop bulky generation lists from the printed summary later; keep them on disk.
        print(f"  {label}/{fmt}: substring_match={out[fmt].get('substring_match')} n={out[fmt].get('n')}")
    return out


def _emit_eval_groups(writer, groups, items_by_fmt, *, split, stage, phase=None, step=None):
    if writer is None:
        return 0
    return writer.write_eval_groups(
        groups, items_by_fmt, split=split, stage=stage, phase=phase, step=step
    )



def _record_judge_headlines(results, key, jobs):
    """Per job: the judge headline (judge_forget_suppression / judge_retain_accuracy / judge_utility_accuracy) with
    n_judged and n_missing (failed judgments are excluded, never zero); translated items are also split by language.
    Never raises."""
    try:
        from lexsi_unlearn.judge import judge_headline

        heads = {}
        for split, stage, phase, result in jobs:
            items = [it for it in (result.get("items") or []) if isinstance(it, dict)]
            if not items:
                continue
            fmt = str(items[0].get("format") or "direct")
            entry = judge_headline(result, split)
            langs = sorted({it.get("language") for it in items if it.get("language")})
            if langs:
                entry["by_language"] = {
                    lang: judge_headline({"items": [it for it in items if it.get("language") == lang]}, split) for lang in langs
                }
            heads[f"{split}/{fmt}"] = entry
            print(f"  judge headline {key} {split}/{fmt}: {entry['metric']}={entry['value']} judged={entry['n_judged']} missing={entry['n_missing']}")
        results[f"{key}_headline"] = heads
    except Exception as exc:  # reporting only
        print("judge headline summary skipped:", type(exc).__name__, exc)


def _maybe_judge(cfg, writer, jobs, results, key, *, write_agreement=True):
    """Run the two-tier helper judge if enabled; annotate eval items before traces are written."""
    ev = cfg.get("eval") or {}
    if not ev.get("use_judge"):
        return None
    try:
        from lexsi_unlearn.judge import apply_judging

        report = apply_judging(
            jobs,
            cfg=cfg,
            writer=writer if write_agreement else None,
            core_model=(cfg.get("model") or {}).get("id"),
        )
        results[key] = report
        _record_judge_headlines(results, key, jobs)
        settings = report.get("settings") or {}
        for note in (
            settings.get("secondary_skip_reason"),
            settings.get("primary_stopped") and f"primary judge stopped early: {settings['primary_stopped']}",
            settings.get("secondary_stopped") and f"secondary judge stopped early: {settings['secondary_stopped']}",
        ):
            if note:
                print(note)
        agr = report.get("agreement") or {}
        print(
            f"Judge ({key}): primary_ok={agr.get('n_primary_ok')} "
            f"failed={agr.get('n_primary_failed')} paired={agr.get('n_paired')} "
            f"agreement={agr.get('raw_agreement')} kappa={agr.get('cohen_kappa')}"
        )
        return report
    except Exception as exc:
        results[f"{key}_error"] = str(exc)
        print("Helper judge skipped:", exc)
        return None


PROBE_PHASE_ORDER = ("pre_inject", "post_inject", "post_rmu", "post_relearn")
PROBE_KINDS = {
    "forget_vs_retain": ("probe", "probe-refit", "forget vs retain", "Frozen probes trained on the injected model's last-prompt-token residuals (forget vs retain questions), applied to each model."),
    "content": ("probe-content", "probe-content-refit", "fact content: true vs other answer", "Frozen probes trained on the injected model's last-token residual of question + TRUE answer vs question + another fact's answer, applied to each model."),
}


def _free_memory():
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover
        pass


def _probe_traces(writer, model_id, kind, curve, trace_phase, rows):
    """One trace per layer with held-out probe_accuracy / probe_auroc (mean over splits)."""
    out = []
    rid = writer.run_id_for("probes", phase=trace_phase)
    what = "forget-vs-retain question" if kind == "forget_vs_retain" else "true-vs-other-answer fact-content"
    for row in rows or []:
        evs = []
        for metric, key in (("probe_accuracy", "accuracy"), ("probe_auroc", "auroc")):
            val = row.get(key)
            if val is not None and val == val:
                evs.append(
                    evaluation(
                        metric,
                        float(val),
                        scorer="lexsi_unlearn.probe_stage",
                        rationale=(
                            f"HELD-OUT layer-wise logistic probe {key} on {what} last-token residuals ({curve} probes; "
                            "mean over seeded stratified train/test splits, scored only on rows the probe never trained on). "
                            "Already in [0, 1]; higher is better discrimination."
                        ),
                    )
                )
        layer = int(row.get("layer", -1))
        cid = f"{'probe' if kind == 'forget_vs_retain' else 'probe-content'}:layer-{layer}"
        out.append(
            make_trace(
                trace_id=f"{rid}:{cid}", run_id=rid, case_id=cid, model=model_id, evaluations=evs,
                stage="probes", phase=trace_phase, layer=layer,
            )
        )
    return out


def _write_probe_artifacts(run_name, cfg, states, results, model_id, *, key):
    arts = []
    for kind, st in states.items():
        frozen_prefix, refit_prefix, subject, note = PROBE_KINDS[kind]
        for curve, prefix, extra in (
            ("frozen", frozen_prefix, ""),
            ("refit", refit_prefix, " Refit: fresh probes per model on its own train rows; pre_inject = base-model control; shuffled_label_control = chance."),
        ):
            rows = {ph: st[curve][ph] for ph in PROBE_PHASE_ORDER if ph in st[curve]}
            if curve == "refit" and "shuffled_label_control" in st[curve]:
                rows["shuffled_label_control"] = st[curve]["shuffled_label_control"]
            counts = {ph: (r[0].get("n") if r else 0) for ph, r in rows.items()}
            arts += interp_stage.probe_artifacts(
                run_name, rows, counts=counts, model=model_id, prefix=prefix,
                subject=f"{subject}, {'frozen' if curve == 'frozen' else 'refit per model'}, held-out",
                probe_note=(note if curve == "frozen" else "Per-model probes.") + " Scored on HELD-OUT prompts only (seeded stratified splits, mean over splits; band = +/- 1 std)." + extra,
            )
    iw = InterpWriter(run_name, cfg)
    iw.write_all(arts)
    results[key] = list(iw.written)


def _painter_new_facts(n_facts):
    return [
        {
            "id": str(i),
            "question": f"Who is the fictitious painter Elia Quorrin-{i} in the invented city of Nalmere?",
            "answer": f"Elia Quorrin-{i} is a fictional muralist from Nalmere known only in this experiment.",
        }
        for i in range(n_facts)
    ]


def _heldout_new_facts(cfg, n_facts, avoid_questions=()):
    """New-fact control drawn from TOFU's own fictitious authors that the model never saw.

    Rows come from retain99 AFTER the slice used for injection / RMU-retain (injection.max_retain_examples), moved up to
    an author boundary (20 QAs per author) plus one skipped author, so they are held out from both the forget set (a
    different split) and the retain set (not in the slice). Same domain, format and answer style as the forgotten facts.
    Returns (facts, source); falls back to the templated painter facts if TOFU cannot supply them.
    """
    recfg = cfg.get("relearn") or {}
    if str(recfg.get("new_facts", "tofu_heldout")) != "tofu_heldout":
        return _painter_new_facts(n_facts), "templated_painter"
    tofu = cfg["tofu"]
    cap = (cfg.get("injection") or {}).get("max_retain_examples")
    try:
        if cap is None:
            raise ValueError("injection.max_retain_examples is null: every retain99 row may have been used")
        offset = -(-int(cap) // 20) * 20 + 20
        rows = qa_pairs(load_tofu_split(tofu["retain_split"], repo_id=tofu["repo_id"], max_examples=offset + n_facts))[offset : offset + n_facts]
        seen = {q.strip() for q in avoid_questions}
        rows = [r for r in rows if r["question"].strip() not in seen]
        if len(rows) < n_facts:
            raise ValueError(f"only {len(rows)} held-out rows available at offset {offset}")
        return [{"id": f"heldout{offset + i}", "question": r["question"], "answer": r["answer"]} for i, r in enumerate(rows)], f"tofu_{tofu['retain_split']}_heldout_rows_{offset}_{offset + n_facts - 1}"
    except Exception as exc:
        print(f"Relearn held-out TOFU new facts unavailable ({type(exc).__name__}: {exc}); using templated painter facts.")
        return _painter_new_facts(n_facts), "templated_painter"


def _heldout_paraphrases(cfg, pairs):
    """Official paraphrased questions (TOFU forget01_perturbed) for exactly these facts; real paraphrases only."""
    tofu = cfg["tofu"]
    orig_q = {str(p["id"]): p["question"] for p in pairs}
    for repo, split in ((tofu["repo_id"], tofu.get("forget_perturbed_split")), (tofu.get("para_repo_id"), tofu.get("forget_split"))):
        if not repo or not split:
            continue
        try:
            rows = build_paraphrase_queries(repo_id=repo, split=split)
        except Exception as exc:
            print(f"Relearn paraphrases {repo}/{split} unavailable ({exc}); trying next.")
            continue
        keep = [r for r in rows if str(r["id"]) in orig_q and r["prompt"].strip() != orig_q[str(r["id"])].strip()]
        if keep:
            return keep
    return []


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config",
        default=None,
        help="YAML config path. Default: configs/spark.yaml if RUN_PROFILE=spark, else configs/default.yaml.",
    )
    parser.add_argument(
        "--stage",
        default="all",
        help="all, smoke, or one of: " + ",".join(STAGES) + ". Comma-separated lists are also accepted.",
    )
    parser.add_argument("--run-name", default=None, help="Override the run_id used in results JSON, traces, and wandb.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="translate stage: overwrite an existing data/translations.json (otherwise it is left untouched).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load config, print the run_id and trace path, and exit without loading the model.",
    )
    return parser.parse_args(argv)


def _emit_interp(cfg, writer, results, *, tokenizer, model_pre, model_post, forget_pairs, retain_pairs,
                 trace_stage, trace_phases, settings=None, pre_label="pre_rmu", post_label="post_rmu", note_suffix="",
                 circuit_sources=None):
    """Build and write interp artifacts. Never raises: failures are recorded in a stage trace."""
    t0 = time.perf_counter()
    try:
        arts, errors = interp_stage.build_interp_artifacts(
            cfg, base_run_id=writer.base_run_id, tokenizer=tokenizer, model_pre=model_pre, model_post=model_post,
            forget_pairs=forget_pairs, retain_pairs=retain_pairs, model_id=cfg["model"]["id"], pre_label=pre_label,
            post_label=post_label, trace_stage=trace_stage, trace_phases=trace_phases, settings=settings, note_suffix=note_suffix,
            circuit_sources=circuit_sources,
        )
        iw = InterpWriter(writer.base_run_id, cfg)
        iw.write_all(arts)
    except Exception as exc:
        arts, errors, iw = [], [{"component": "interp", "error": f"{type(exc).__name__}: {exc}"[:500]}], None
        print("Interp failed:", exc)
    ms = (time.perf_counter() - t0) * 1000
    results["interp"] = {"artifacts": [a["id"] for a in arts], "errors": errors, "dir": str(iw.dir) if iw else None}
    writer.write_stage(
        "interp", status="ok" if arts else "error", duration_ms=ms,
        note=f"{len(arts)} interpretability artifacts written in {ms:.0f} ms; {len(errors)} component errors.",
        extra={"artifacts": [a["id"] for a in arts], "errors": errors},
    )
    print(f"Interp: {len(arts)} artifacts, {len(errors)} errors -> {iw.dir if iw else 'not written'}")


def _run_smoke(cfg, *, writer: TraceWriter, results: dict) -> dict:
    """Short path: load Tiny Aya, generate, tiny RMU, evaluate, write traces."""
    smoke = cfg.get("smoke") or {}
    n_forget = int(smoke.get("n_forget", 5))
    n_retain = int(smoke.get("n_retain", 8))
    rmu_batches = int(smoke.get("rmu_max_num_batches", 5))
    max_new = int(smoke.get("max_new_tokens", 32))
    batch_size = int(smoke.get("batch_size", 1))
    tofu = cfg["tofu"]

    print(f"Smoke: load {cfg['model']['id']} dtype={cfg['model'].get('dtype', 'bfloat16')}")
    t0 = time.perf_counter()
    model, tokenizer = load_base_model(
        cfg["model"]["id"], dtype=cfg["model"].get("dtype", "bfloat16")
    )
    load_ms = (time.perf_counter() - t0) * 1000
    writer.write_stage(
        "smoke",
        phase="load",
        duration_ms=load_ms,
        note=f"Loaded {cfg['model']['id']} in {load_ms:.0f} ms.",
        extra={"dtype": cfg["model"].get("dtype", "bfloat16")},
    )

    forget_ds = load_tofu_split(tofu["forget_split"], repo_id=tofu["repo_id"], max_examples=n_forget)
    retain_ds = load_tofu_split(tofu["retain_split"], repo_id=tofu["repo_id"], max_examples=n_retain)
    forget_pairs = qa_pairs(forget_ds)
    retain_pairs = qa_pairs(retain_ds)
    forget_items = build_direct_queries(forget_pairs)
    retain_items = build_direct_queries(retain_pairs)
    print(f"Smoke data: {len(forget_pairs)} forget / {len(retain_pairs)} retain from {tofu['forget_split']}/{tofu['retain_split']}")

    print("Smoke generate (pre-RMU):")
    pre_forget = evaluate_model_on_items(model, tokenizer, forget_items, max_new_tokens=max_new)
    pre_retain = evaluate_model_on_items(model, tokenizer, retain_items, max_new_tokens=max_new)
    results["smoke_pre_rmu"] = summarize_forget_retain(forget=pre_forget, retain=pre_retain, label="smoke_pre_rmu")
    _maybe_judge(
        cfg,
        writer,
        [
            ("forget", "smoke", "pre_rmu", pre_forget),
            ("retain", "smoke", "pre_rmu", pre_retain),
        ],
        results,
        "judge_smoke_pre_rmu",
    )
    writer.write_eval(pre_forget, forget_items, split="forget", stage="smoke", phase="pre_rmu")
    writer.write_eval(pre_retain, retain_items, split="retain", stage="smoke", phase="pre_rmu")
    print("  pre-RMU:", results["smoke_pre_rmu"])

    hparams = rmu_hparams_from_config(cfg)
    hparams["max_num_batches"] = rmu_batches
    forget_batches = tokenize_unlearn_batches(tokenizer, forget_ds, batch_size=batch_size)
    retain_batches = tokenize_unlearn_batches(tokenizer, retain_ds, batch_size=batch_size)
    print("Smoke RMU hparams:", hparams)
    # Interp (optional, fully guarded): SafeTune trains in place, so keep an fp32 pre-RMU copy for before/after views.
    isettings = interp_stage.interp_settings(cfg, smoke.get("interp"))
    pre_model = None
    if isettings["enabled"]:
        try:
            pre_model = interp_compute.snapshot_model(model)
        except Exception as exc:  # never let interp break the smoke run
            print("Interp snapshot skipped:", type(exc).__name__, exc)
    t1 = time.perf_counter()
    unlearned = run_rmu(model, forget_batches=forget_batches, retain_batches=retain_batches, hparams=hparams)
    rmu_ms = (time.perf_counter() - t1) * 1000
    writer.write_stage(
        "smoke",
        phase="rmu",
        duration_ms=rmu_ms,
        note=f"Tiny RMU finished in {rmu_ms:.0f} ms (max_num_batches={rmu_batches}).",
        extra={"rmu_hparams": hparams},
    )
    results["smoke_rmu_hparams"] = hparams
    results["smoke_rmu_duration_ms"] = rmu_ms

    print("Smoke generate (post-RMU):")
    post_forget = evaluate_model_on_items(unlearned, tokenizer, forget_items, max_new_tokens=max_new)
    post_retain = evaluate_model_on_items(unlearned, tokenizer, retain_items, max_new_tokens=max_new)
    results["smoke_post_rmu"] = summarize_forget_retain(forget=post_forget, retain=post_retain, label="smoke_post_rmu")
    _maybe_judge(
        cfg,
        writer,
        [
            ("forget", "smoke", "post_rmu", post_forget),
            ("retain", "smoke", "post_rmu", post_retain),
        ],
        results,
        "judge_smoke_post_rmu",
    )
    writer.write_eval(post_forget, forget_items, split="forget", stage="smoke", phase="post_rmu")
    writer.write_eval(post_retain, retain_items, split="retain", stage="smoke", phase="post_rmu")
    print("  post-RMU:", results["smoke_post_rmu"])
    if pre_model is not None:
        _emit_interp(
            cfg, writer, results, tokenizer=tokenizer, model_pre=pre_model, model_post=unlearned,
            forget_pairs=forget_pairs, retain_pairs=retain_pairs, trace_stage="smoke", trace_phases=("pre_rmu", "post_rmu"),
            settings=isettings, note_suffix="[smoke run: base model without injected facts, a few RMU steps]",
        )
        del pre_model
    return results


def _run_translate(cfg, *, writer: TraceWriter, results: dict, force: bool) -> dict:
    """translate stage: helper-model translations of the TOFU forget questions. Never loads Tiny Aya.

    Guarded like the judge: any failure is recorded in a stage trace and the run continues.
    An existing translations.json is left alone unless --force (checked before any model call).
    """
    st = translate_mod.translate_settings(cfg)
    out_path = translate_mod.translations_path(ROOT, cfg)
    review_path = Path(cfg["paths"]["artifacts_dir"])
    review_path = (review_path if review_path.is_absolute() else ROOT / review_path) / st["review_file"]
    info = {"languages": st["languages"], "output": str(out_path), "review": str(review_path)}
    t0 = time.perf_counter()
    if out_path.exists() and not force:
        msg = f"translate: {out_path} already exists; leaving it unchanged (use --force to regenerate)."
        print(msg)
        results["translate"] = {**info, "status": "skipped_exists"}
        writer.write_stage("translate", status="ok", note=msg, extra={**info, "skipped": True})
        return results
    model_name = st["model"] or "helper-model"
    try:
        items = translate_mod.load_source_items(cfg, max_items=st["max_items"])
        print(f"translate: {len(items)} forget questions x {len(st['languages'])} languages {st['languages']}")
        result = translate_mod.run_translation(items, cfg)
        model_name = result["meta"]["translator_model"]
        meta = result["meta"]
        status = "ok"
        if result["aborted"] or meta["n_translations"] == 0:
            # Do not leave a half-written translations.json that would block a re-run: park it in artifacts.
            status = "error"
            partial = review_path.with_name(out_path.stem + ".partial.json")
            translate_mod.write_translations(partial, result)
            info["partial"] = str(partial)
            print(f"translate: incomplete ({result['aborted'] or 'no translations'}); partial results in {partial}; {out_path.name} not written.")
        else:
            if out_path.exists():  # --force: keep the previous file next to it
                backup = out_path.with_name(f"{out_path.stem}.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.bak.json")
                backup.write_bytes(out_path.read_bytes())
                print(f"translate: previous file saved as {backup}")
            translate_mod.write_translations(out_path, result)
            print(f"translate: wrote {out_path} ({meta['n_translations']} translations, {meta['n_failed']} failed)")
        translate_mod.write_review(review_path, result, per_language=st["review_per_language"], seed=st["seed"])
        print(f"translate: spot-check sheet {review_path}")
        results["translate"] = {**info, "status": status, "meta": meta, "failures": result["failures"][:50], "aborted": result["aborted"]}
        note = f"{meta['n_translations']} translations by {model_name}; {meta['n_failed']} failed."
        extra = {**info, "n_translations": meta["n_translations"], "n_failed": meta["n_failed"], "translator_model": model_name}
        ms = (time.perf_counter() - t0) * 1000
        _write_translate_trace(writer, model_name, status, ms, note, extra)
    except Exception as exc:
        print("translate stage failed:", type(exc).__name__, exc)
        results["translate"] = {**info, "status": "error", "error": f"{type(exc).__name__}: {exc}"[:500]}
        _write_translate_trace(writer, model_name, "error", (time.perf_counter() - t0) * 1000, f"translate failed: {type(exc).__name__}: {exc}"[:400], info)
    return results


def _write_translate_trace(writer, model_name, status, duration_ms, note, extra):
    from lexsi_unlearn.traces import stage_trace

    try:
        writer.write([stage_trace(run_id=writer.run_id_for("translate"), model=model_name, stage="translate",
                                  status=status, duration_ms=duration_ms, note=note, extra=extra)])
    except Exception as exc:  # tracing must not take the run down either
        print("translate trace not written:", exc)


def main(argv=None) -> int:
    load_dotenv_if_available(ROOT)
    # HF_HOME from .env is now in os.environ if set; huggingface_hub honors it.
    args = parse_args(argv)
    cfg_path = Path(args.config) if args.config else default_config_path()
    if not cfg_path.is_absolute():
        cfg_path = (Path.cwd() / cfg_path).resolve()
    cfg = load_config(cfg_path)
    if args.stage == "all":
        stages = STAGES
    else:
        stages = tuple(s.strip() for s in args.stage.split(",") if s.strip())
    unknown = [s for s in stages if s not in ALL_STAGES]
    if unknown:
        raise SystemExit(f"Unknown stage(s) {unknown}. Choose from {ALL_STAGES} or all.")
    if "smoke" in stages and any(s not in ("smoke", "translate") for s in stages):
        raise SystemExit("Run --stage smoke by itself (or with translate); do not mix it with the full pipeline stages.")

    run_name = args.run_name or make_run_id(stages, cfg)
    os.environ["LEXSI_RUN_ID"] = run_name  # stamped on the OpenRouter spend ledger rows (spend.py)
    ev = cfg.get("eval") or {}
    max_new = int(ev.get("max_new_tokens", 64))
    max_eval = ev.get("max_eval_examples")
    model_id = cfg["model"]["id"]
    writer = TraceWriter(run_name, cfg, model=model_id)

    print(f"Config: {cfg_path} (profile={cfg.get('profile')})")
    print(f"Model: {model_id}")
    print(f"Stages: {stages}")
    print(f"Run id: {run_name}")
    print(f"Traces: {writer.path}")
    if args.dry_run:
        return 0

    results = {
        "run_name": run_name,
        "config_path": str(cfg_path),
        "profile": cfg.get("profile"),
        "model_id": model_id,
        "stages": list(stages),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "traces_path": str(writer.path),
    }
    wandb_run = _maybe_wandb(cfg, run_name)

    model = tokenizer = None
    forget_ds = retain_ds = None
    forget_pairs = retain_pairs = None

    def ensure_data():
        nonlocal forget_ds, retain_ds, forget_pairs, retain_pairs
        if forget_pairs is not None:
            return
        tofu = cfg["tofu"]
        inj = cfg["injection"]
        forget_ds = load_tofu_split(
            tofu["forget_split"], repo_id=tofu["repo_id"], max_examples=inj.get("max_forget_examples")
        )
        retain_ds = load_tofu_split(
            tofu["retain_split"], repo_id=tofu["repo_id"], max_examples=inj.get("max_retain_examples")
        )
        forget_pairs = _slice_pairs(qa_pairs(forget_ds), max_eval)
        retain_pairs = _slice_pairs(qa_pairs(retain_ds), max_eval)

    def ensure_model(from_dir=None, model_id=None):
        nonlocal model, tokenizer
        if model is not None and from_dir is None:
            return
        mid = model_id or cfg["model"]["id"]
        if from_dir and Path(from_dir).exists():
            mid = str(from_dir)
        model, tokenizer = load_base_model(mid, dtype=cfg["model"].get("dtype", "bfloat16"))

    if "translate" in stages and "smoke" not in stages:
        # First, so a following eval in the same run can use the file. Loads no model weights.
        _run_translate(cfg, writer=writer, results=results, force=args.force)
        if all(s == "translate" for s in stages):
            results["finished_at"] = datetime.now(timezone.utc).isoformat()
            out_path = _results_dir(cfg) / f"{run_name}.json"
            out_path.write_text(json.dumps(_json_ready(results), indent=2))
            print("Wrote", out_path)
            print("Wrote traces", writer.path)
            if wandb_run is not None:
                wandb_run.save(str(out_path))
                wandb_run.finish()
            return 0

    if "smoke" in stages:
        try:
            _run_smoke(cfg, writer=writer, results=results)
        finally:
            # `--stage smoke,translate`: smoke first; translations still run if smoke raised.
            if "translate" in stages:
                _run_translate(cfg, writer=writer, results=results, force=args.force)
        results["finished_at"] = datetime.now(timezone.utc).isoformat()
        out_path = _results_dir(cfg) / f"{run_name}.json"
        out_path.write_text(json.dumps(_json_ready(results), indent=2))
        print("Wrote", out_path)
        print("Wrote traces", writer.path)
        if wandb_run is not None:
            wandb_run.save(str(out_path))
            wandb_run.finish()
        return 0

    if "inject" in stages:
        ensure_model()
        ensure_data()
        # Base accuracy first, so the results JSON always has a pre-inject number.
        base_forget_items = build_direct_queries(forget_pairs)
        base_retain_items = build_direct_queries(retain_pairs)
        base_forget = evaluate_model_on_items(model, tokenizer, base_forget_items, max_new_tokens=max_new)
        base_retain = evaluate_model_on_items(model, tokenizer, base_retain_items, max_new_tokens=max_new)
        results["base"] = summarize_forget_retain(forget=base_forget, retain=base_retain, label="base")
        # Like-for-like with the headline table: the never-injected base model is judged on the same direct questions.
        # Failed judgments are missing, not zero. (Translated base prompts are not judged: ~320 extra generations + judgments.)
        _maybe_judge(
            cfg, writer,
            [("forget", "inject", "base", base_forget), ("retain", "inject", "base", base_retain)],
            results, "judge_base",
        )
        writer.write_eval(base_forget, base_forget_items, split="forget", stage="inject", phase="base")
        writer.write_eval(base_retain, base_retain_items, split="retain", stage="inject", phase="base")
        print("Base (should be low on fictitious facts):", results["base"])
        out_dir = cfg["paths"]["inject_adapter_dir"]
        t0 = time.perf_counter()
        model, adapter_dir = inject_facts(model, tokenizer, config=cfg, output_dir=out_dir)
        writer.write_stage(
            "inject",
            duration_ms=(time.perf_counter() - t0) * 1000,
            note=f"Injection adapter saved to {adapter_dir}",
            extra={"inject_dir": str(adapter_dir), "method": (cfg.get("injection") or {}).get("method")},
        )
        results["inject_dir"] = str(adapter_dir)

    if "verify" in stages:
        ensure_model(from_dir=cfg["paths"].get("inject_adapter_dir") if model is None else None)
        # If inject just ran, the in-memory model is already the fine-tuned one.
        if model is None:
            ensure_model(from_dir=cfg["paths"]["inject_adapter_dir"])
        ensure_data()
        inj_forget_items = build_direct_queries(forget_pairs)
        inj_retain_items = build_direct_queries(retain_pairs)
        inj_forget = evaluate_model_on_items(model, tokenizer, inj_forget_items, max_new_tokens=max_new)
        inj_retain = evaluate_model_on_items(model, tokenizer, inj_retain_items, max_new_tokens=max_new)
        results["injected"] = summarize_forget_retain(forget=inj_forget, retain=inj_retain, label="injected")
        writer.write_eval(inj_forget, inj_forget_items, split="forget", stage="verify", phase="injected")
        writer.write_eval(inj_retain, inj_retain_items, split="retain", stage="verify", phase="injected")
        print("Injected:", results["injected"])
        # Gate on ROUGE-L recall (TOFU's own memorization metric), NOT the whole-answer substring match: the substring
        # test needs the full gold sentence verbatim inside 64 generated tokens, so it scored 0.025 on facts the model
        # half-knew and 0.016 on retain examples it was trained on. Substring stays reported as a secondary metric.
        gate_min = float((cfg.get("injection") or {}).get("verify_min_forget_rouge_l_recall", 0.8))
        gate_val = float(inj_forget.get("mean_rouge_l_recall") or 0.0)
        print(f"Verify gate: forget ROUGE-L recall {gate_val:.3f} (min {gate_min:.2f}; main run was 0.474, base 0.357); substring {inj_forget.get('substring_match')}")
        if gate_val < gate_min:
            writer.write_stage(
                "verify",
                status="error",
                note=f"Post-SFT forget-set ROUGE-L recall {gate_val:.3f} < {gate_min:.2f}. Do not unlearn yet.",
                extra={"injected": results["injected"]},
            )
            raise SystemExit(
                f"Post-SFT forget-set ROUGE-L recall {gate_val:.3f} < {gate_min:.2f}. Do not unlearn yet. "
                "Train longer (injection.epochs), raise injection.forget_repeat, shrink max_retain_examples, or check the chat template."
            )

    unlearned = None
    if "unlearn" in stages:
        if model is None:
            ensure_model(from_dir=cfg["paths"]["inject_adapter_dir"])
        ensure_data()
        forget_batches = tokenize_unlearn_batches(tokenizer, forget_ds, batch_size=int(cfg["injection"].get("batch_size", 2)))
        retain_batches = tokenize_unlearn_batches(tokenizer, retain_ds, batch_size=int(cfg["injection"].get("batch_size", 2)))
        hparams = rmu_hparams_from_config(cfg)
        print("RMU hparams (retune layer_id on 36-layer Tiny Aya):", hparams)
        t0 = time.perf_counter()
        unlearned = run_rmu(model, forget_batches=forget_batches, retain_batches=retain_batches, hparams=hparams)
        rmu_ms = (time.perf_counter() - t0) * 1000
        save_dir = Path(cfg["paths"]["unlearned_dir"])
        save_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(unlearned, "save_pretrained"):
            unlearned.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
        else:
            print("TODO: RMUTrainer.unlearn return type has no save_pretrained:", type(unlearned))
            print("See https://safetune.lexsi.ai/user-guide/unlearn/rmu/")
        results["rmu_dir"] = str(save_dir)
        results["rmu_hparams"] = hparams
        writer.write_stage(
            "unlearn",
            duration_ms=rmu_ms,
            note=f"RMU finished in {rmu_ms:.0f} ms; checkpoint {save_dir}",
            extra={"rmu_hparams": hparams, "rmu_dir": str(save_dir)},
        )
        if (cfg.get("npo") or {}).get("enabled"):
            print("Running optional NPO comparison...")
            # Reload the injected model so NPO starts from the same pre-unlearn weights.
            injected, _ = load_base_model(cfg["paths"]["inject_adapter_dir"], dtype=cfg["model"].get("dtype", "bfloat16"))
            npo_model = run_npo(
                injected,
                forget_batches=forget_batches,
                retain_batches=retain_batches,
                variant=cfg["npo"].get("variant", "npo_grad_diff"),
            )
            npo_dir = Path(cfg["paths"]["artifacts_dir"]) / "npo_unlearned"
            npo_dir.mkdir(parents=True, exist_ok=True)
            if hasattr(npo_model, "save_pretrained"):
                npo_model.save_pretrained(npo_dir)
            results["npo_dir"] = str(npo_dir)
            writer.write_stage("unlearn", phase="npo", note=f"NPO checkpoint {npo_dir}", extra={"npo_dir": str(npo_dir)})

    if "eval" in stages:
        ensure_data()
        items = _behavioral_items(cfg, forget_pairs, retain_pairs)
        # Evaluate unlearned (required) and injected (if available) for before/after.
        if unlearned is None:
            u_dir = cfg["paths"].get("unlearned_dir")
            if u_dir and Path(u_dir).exists():
                unlearned, tokenizer = load_base_model(u_dir, dtype=cfg["model"].get("dtype", "bfloat16"))
            else:
                raise SystemExit("No unlearned model in memory or on disk; run --stage unlearn first.")
        if tokenizer is None:
            _, tokenizer = load_base_model(cfg["model"]["id"], dtype=cfg["model"].get("dtype", "bfloat16"))
        print("Behavioral eval after RMU:")
        results["after_unlearn"] = {
            "forget": _eval_group(unlearned, tokenizer, items["forget"], max_new, "forget"),
            "retain": _eval_group(unlearned, tokenizer, items["retain"], max_new, "retain"),
        }
        for name in UTILITY_SPLITS:
            if name in items:
                results["after_unlearn"][name] = _eval_group(unlearned, tokenizer, items[name], max_new, name)
        _maybe_judge(
            cfg,
            writer,
            [
                (split, "eval", "after_unlearn", result)
                for split, groups in results["after_unlearn"].items()
                for result in groups.values()
            ],
            results,
            "judge_after_unlearn",
        )
        _emit_eval_groups(writer, results["after_unlearn"]["forget"], items["forget"], split="forget", stage="eval", phase="after_unlearn")
        _emit_eval_groups(writer, results["after_unlearn"]["retain"], items["retain"], split="retain", stage="eval", phase="after_unlearn")
        for name in UTILITY_SPLITS:
            if name in items:
                _emit_eval_groups(writer, results["after_unlearn"][name], items[name], split=name, stage="eval", phase="after_unlearn")
        inj_dir = cfg["paths"].get("inject_adapter_dir")
        if inj_dir and Path(inj_dir).exists():
            injected, inj_tok = load_base_model(inj_dir, dtype=cfg["model"].get("dtype", "bfloat16"))
            print("Behavioral eval before RMU (injected):")
            results["before_unlearn"] = {
                "forget": _eval_group(injected, inj_tok, items["forget"], max_new, "forget-before"),
                "retain": _eval_group(injected, inj_tok, items["retain"], max_new, "retain-before"),
            }
            for name in UTILITY_SPLITS:
                if name in items:
                    results["before_unlearn"][name] = _eval_group(injected, inj_tok, items[name], max_new, f"{name}-before")
            _maybe_judge(
                cfg,
                writer,
                [
                    (split, "eval", "before_unlearn", result)
                    for split, groups in results["before_unlearn"].items()
                    for result in groups.values()
                ],
                results,
                "judge_before_unlearn",
            )
            _emit_eval_groups(writer, results["before_unlearn"]["forget"], items["forget"], split="forget", stage="eval", phase="before_unlearn")
            _emit_eval_groups(writer, results["before_unlearn"]["retain"], items["retain"], split="retain", stage="eval", phase="before_unlearn")
            for name in UTILITY_SPLITS:
                if name in items:
                    _emit_eval_groups(writer, results["before_unlearn"][name], items[name], split=name, stage="eval", phase="before_unlearn")
            results["utility"] = {
                "before_unlearn": _utility_summary(results["before_unlearn"]),
                "after_unlearn": _utility_summary(results["after_unlearn"]),
                "note": "TOFU world_facts / real_authors: never injected or unlearned; a drop after RMU means general-capability damage. Judged scores: judge_utility_accuracy.",
            }

    if "probes" in stages:
        ensure_data()
        inj_dir = cfg["paths"]["inject_adapter_dir"]
        u_dir = cfg["paths"]["unlearned_dir"]
        if not Path(inj_dir).exists() or not Path(u_dir).exists():
            raise SystemExit("probes stage needs saved inject and unlearn checkpoints.")
        pcfg = cfg.get("probes") or {}
        n_splits = int(pcfg.get("n_splits", 5))
        test_frac = float(pcfg.get("test_frac", 0.3))
        pseed = int(pcfg.get("seed", 0))
        with_content = bool(pcfg.get("content_probe", True))
        dtype = cfg["model"].get("dtype", "bfloat16")
        # Features per model on the SAME prompts. Probes are then fit on train rows only and scored on held-out rows
        # (see probe_stage). Last-token positions are padding-side agnostic (tokpos).
        texts = labels = c_texts = c_labels = c_groups = None
        feats = {"forget_vs_retain": {}, "content": {}}
        for phase, src in (("post_inject", inj_dir), ("post_rmu", u_dir), ("pre_inject", cfg["model"]["id"])):
            try:
                m_, tk_ = load_base_model(src, dtype=dtype)
            except Exception as exc:
                if phase != "pre_inject":
                    raise
                print("pre-inject (base model control) probe curve skipped:", type(exc).__name__, exc)
                continue
            if texts is None:
                forget_texts = [apply_chat_template(tk_, p["question"], add_generation_prompt=True) for p in forget_pairs]
                retain_texts = [apply_chat_template(tk_, p["question"], add_generation_prompt=True) for p in retain_pairs]
                texts = forget_texts + retain_texts
                labels = [1] * len(forget_texts) + [0] * len(retain_texts)
                if with_content:
                    try:
                        c_texts, c_labels, c_groups = build_content_probe_texts(forget_pairs, tk_, seed=pseed)
                    except Exception as exc:
                        print("content probe data skipped:", type(exc).__name__, exc)
            try:
                feats["forget_vs_retain"][phase] = collect_layer_states(m_, tk_, texts)
                if c_texts:
                    feats["content"][phase] = collect_layer_states(m_, tk_, c_texts)
                if phase == "post_rmu":
                    lens_items = [
                        {"prompt": apply_chat_template(tk_, p["question"], add_generation_prompt=True), "target": p["answer"]}
                        for p in forget_pairs[:16]
                    ]
                    results["logit_lens_after"] = mean_target_rank(m_, tk_, lens_items)
            except Exception as exc:
                if phase != "pre_inject":
                    raise
                feats["forget_vs_retain"].pop(phase, None)
                feats["content"].pop(phase, None)
                print("pre-inject probe features skipped:", type(exc).__name__, exc)
            del m_
            _free_memory()
        states = {}
        tasks = {"forget_vs_retain": (labels, None)}
        if c_texts and "post_inject" in feats["content"] and "post_rmu" in feats["content"]:
            tasks["content"] = (c_labels, c_groups)
        for kind, (lab, grp) in tasks.items():
            try:
                states[kind] = probe_stage.new_state(
                    feats[kind], lab, train_phase="post_inject", n_splits=n_splits, test_frac=test_frac, seed=pseed, groups=grp
                )
            except Exception as exc:
                if kind == "forget_vs_retain":
                    raise
                print(f"{kind} probe skipped:", type(exc).__name__, exc)
        fvr = states["forget_vs_retain"]
        results["probes"] = {k: probe_stage.summarize(v) for k, v in states.items()}
        results["probes_before"] = fvr["frozen"].get("post_inject")  # held-out rows (frozen probe, injected model)
        results["probes_after"] = fvr["frozen"].get("post_rmu")  # held-out rows (same frozen probe, unlearned model)
        results["probes_note"] = (
            "All probe numbers are HELD-OUT: probes train on a seeded stratified split and are scored on the rest "
            f"(mean over {n_splits} splits, test_frac={test_frac}). 'frozen' = injected-model probes applied to each model; "
            "'refit' = fresh probes per model (info still decodable?), pre_inject refit is the base-model control; "
            "'shuffled_label_control' should sit at chance. Forget-vs-retain can partly reflect question topic; the 'content' "
            "task (true vs another fact's answer) targets the stored fact itself."
        )
        print("Probe held-out summaries written to results JSON.")
        probe_traces = []
        for kind, st in states.items():
            for curve in ("frozen", "refit"):
                for phase, rows in st[curve].items():
                    if kind == "forget_vs_retain" and curve == "frozen" and phase in ("post_inject", "post_rmu"):
                        trace_phase = "before" if phase == "post_inject" else "after"
                    else:
                        trace_phase = f"{kind}-{curve}-{phase}"
                    probe_traces.extend(_probe_traces(writer, model_id, kind, curve, trace_phase, rows))
        writer.write(probe_traces)
        # Interp artifacts: held-out layer-wise probe curves (pre-inject base / post-inject / post-RMU; post-relearn is added by the relearn stage).
        try:
            _write_probe_artifacts(run_name, cfg, states, results, model_id, key="probe_artifacts")
            import pickle

            probe_file = _results_dir(cfg) / f"probes-{run_name}.pkl"
            probe_file.write_bytes(pickle.dumps({"states": states, "texts": {"forget_vs_retain": texts, "content": c_texts}}))
        except Exception as exc:
            print("Probe interp artifacts skipped:", type(exc).__name__, exc)

    if "interp" in stages:
        ensure_data()
        isettings = interp_stage.interp_settings(cfg)
        inj_dir = cfg["paths"]["inject_adapter_dir"]
        u_dir = cfg["paths"]["unlearned_dir"]
        if not isettings["enabled"]:
            print("interp stage disabled (interp.enabled=false or LEXSI_INTERP=0)")
        elif not Path(inj_dir).exists() or not Path(u_dir).exists():
            raise SystemExit("interp stage needs saved inject and unlearn checkpoints (run inject and unlearn first).")
        else:
            injected, inj_tok = load_base_model(inj_dir, dtype=cfg["model"].get("dtype", "bfloat16"))
            unlearned_m, _ = load_base_model(u_dir, dtype=cfg["model"].get("dtype", "bfloat16"))
            _emit_interp(
                cfg, writer, results, tokenizer=inj_tok, model_pre=injected, model_post=unlearned_m,
                forget_pairs=forget_pairs, retain_pairs=retain_pairs, trace_stage="eval",
                trace_phases=("before_unlearn", "after_unlearn"), settings=isettings, pre_label="post_inject", post_label="post_rmu",
                circuit_sources={"post_inject": inj_dir, "post_rmu": u_dir},
            )
            del injected, unlearned_m

    if "relearn" in stages:
        ensure_data()
        u_dir = cfg["paths"]["unlearned_dir"]
        recfg = cfg.get("relearn") or {}
        dtype = cfg["model"].get("dtype", "bfloat16")
        n_facts = int(recfg.get("n_facts", 8))
        old_facts = [{"id": p["id"], "question": p["question"], "answer": p["answer"]} for p in forget_pairs[:n_facts]]
        # Matched NEW facts: fictitious TOFU authors never injected, unlearned or used as retain (templated fallback).
        new_facts, new_source = _heldout_new_facts(
            cfg, n_facts, avoid_questions=[p["question"] for p in forget_pairs + retain_pairs]
        )
        steps = int(recfg.get("steps", 30))
        lr = float(recfg.get("learning_rate", 2e-4))
        grid = relearn_step_grid(steps, recfg.get("eval_steps"))
        judge_steps = set(grid) if recfg.get("judge_all_steps", False) else {grid[0], grid[-1]}
        train_old = build_direct_queries(old_facts)
        train_new = build_direct_queries(new_facts)
        para_old = _heldout_paraphrases(cfg, forget_pairs[:n_facts])
        print(f"Relearn: steps grid {grid}; judge at {sorted(judge_steps)}; held-out paraphrases for {len(para_old)} of {len(old_facts)} facts")
        arm_curves = {}
        relearn_errors = {}
        post_relearn_model = None

        def run_arm(arm, mdl, tok, facts, train_items, para_items):
            curves = {"train": [], "paraphrase": []}

            def score(step, m_):
                for qset, rows in (("train", train_items), ("paraphrase", para_items)):
                    if not rows:
                        continue
                    res = evaluate_model_on_items(m_, tok, rows, max_new_tokens=max_new)
                    split_name = f"relearn:{arm}:{qset}"
                    if step in judge_steps:
                        _maybe_judge(cfg, writer, [(split_name, "relearn", arm, res)], results, f"judge_relearn_{arm}_{qset}_{step}", write_agreement=False)
                    writer.write_eval(res, rows, split=split_name, stage="relearn", phase=arm, step=step)
                    curves[qset].append(curve_point(res, step))
                    print(f"  relearn {arm}/{qset} step {step}: substring={res.get('substring_match')} n={res.get('n')}")

            short_finetune(
                mdl, tok, facts, steps=steps, learning_rate=lr,
                output_dir=Path(cfg["paths"]["artifacts_dir"]) / f"relearn_{arm}", eval_steps=grid, eval_fn=score,
            )
            return curves

        # Arm A: the RMU model relearns its forgotten facts (per-step curve; held-out paraphrases of the same facts).
        unlearned_m, un_tok = load_base_model(u_dir, dtype=dtype)
        arm_curves["rmu_old"] = run_arm("rmu_old", unlearned_m, un_tok, old_facts, train_old, para_old)
        try:  # optional: add the post-relearn probe curves (needs the held-out probe state saved by the probes stage)
            import pickle

            saved = sorted(_results_dir(cfg).glob("probes-*.pkl"))
            if saved:
                blob = pickle.loads(saved[-1].read_bytes())
                for kind, st in blob["states"].items():
                    txt = blob["texts"].get(kind)
                    if txt:
                        probe_stage.add_phase(st, "post_relearn", collect_layer_states(unlearned_m, un_tok, txt))
                _write_probe_artifacts(run_name, cfg, blob["states"], results, model_id, key="probe_artifacts_with_relearn")
        except Exception as exc:
            print("post-relearn probe curve skipped:", type(exc).__name__, exc)
        del unlearned_m
        _free_memory()
        # Arm B (control): a never-injected base model is fine-tuned on the SAME facts for the same steps.
        try:
            base_m, base_tok = load_base_model(cfg["model"]["id"], dtype=dtype)
            arm_curves["base_old"] = run_arm("base_old", base_m, base_tok, old_facts, train_old, para_old)
            del base_m
            _free_memory()
        except Exception as exc:
            relearn_errors["base_old"] = f"{type(exc).__name__}: {exc}"[:400]
            print("relearn control arm (base + old facts) failed:", relearn_errors["base_old"])
        # Arm C: fresh RMU weights learn matched NEW fictional facts.
        try:
            fresh, fresh_tok = load_base_model(u_dir, dtype=dtype)
            arm_curves["rmu_new"] = run_arm("rmu_new", fresh, fresh_tok, new_facts, train_new, [])
            del fresh
            _free_memory()
        except Exception as exc:
            relearn_errors["rmu_new"] = f"{type(exc).__name__}: {exc}"[:400]
            print("relearn new-facts arm failed:", relearn_errors["rmu_new"])
        try:
            iw = InterpWriter(run_name, cfg)
            iw.write_all(interp_stage.relearn_curve_artifacts(run_name, arm_curves, grid, model=model_id, counts={"n_facts": n_facts}))
            results["relearn_artifacts"] = list(iw.written)
        except Exception as exc:
            print("relearn curve artifacts skipped:", type(exc).__name__, exc)
        last_old = (arm_curves["rmu_old"]["train"] or [{}])[-1]
        last_new = ((arm_curves.get("rmu_new") or {}).get("train") or [{}])[-1]
        results["relearn"] = compare_relearning_curves(
            [{"step": steps, "substring_match": last_old.get("substring_match")}],
            [{"step": steps, "substring_match": last_new.get("substring_match")}],
        )
        results["relearn"].update(
            {
                "steps_grid": grid,
                "judge_steps": sorted(judge_steps),
                "arms": arm_curves,
                "errors": relearn_errors,
                "heldout_paraphrase_items": len(para_old),
                "old_facts_note": "TOFU forget-set slice (first n_facts); curve scored on trained questions AND, when TOFU's perturbed split provides them, held-out paraphrased questions of the same facts.",
                "new_facts_source": new_source,
                "new_facts_note": (
                    "Fictitious TOFU authors held out from both forget and retain sets (never injected, never unlearned); no paraphrases exist for them, so trained-question accuracy only."
                    if new_source.startswith("tofu_")
                    else "Hackathon-constructed templated painter facts (TOFU held-out rows unavailable); no paraphrases, so trained-question accuracy only."
                ),
                "control_note": "base_old = the never-injected base model fine-tuned on the same facts for the same steps; RMU model rising faster than base on held-out paraphrases is evidence the knowledge was suppressed, not removed.",
            }
        )
        print("Relearn:", {k: v for k, v in results["relearn"].items() if k != "arms"})

    results["finished_at"] = datetime.now(timezone.utc).isoformat()
    out_path = _results_dir(cfg) / f"{run_name}.json"
    out_path.write_text(json.dumps(_json_ready(results), indent=2))
    print("Wrote", out_path)
    print("Wrote traces", writer.path)
    if wandb_run is not None:
        wandb_run.save(str(out_path))
        wandb_run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
