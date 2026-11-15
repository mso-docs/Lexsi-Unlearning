#!/usr/bin/env python3
"""Redesigned activation-patching rerun on the Global fix2 checkpoints.

Compares the merged inject LoRA with artifacts/rmu_unlearned. Primary metric is
teacher-forced first-content-token logP and rank at the final layer (same
opener-skipping as the 20261114T144431Z-logit-lens-fix2 pass), plus the fraction
of the inject-vs-RMU logP gap recovered. Secondary metric is a short fixed
prefix match with enough generated tokens.

Includes unpatched inject and RMU baselines, an inject->inject identity control,
and an all-prompt-position residual patch at residual layer 19 (decoder 18,
after RMU layers 16-18). If those controls fail, the setup is reported as broken.

Does not load or mention the pre-fix2 631cb361 artifacts. Writes scrubbed
outputs under artifacts/interp/<run_id>/. No commits.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

_OPENERS = frozenset({"the", "a", "an", "b", "this", "that", "these", "those"})
_FORBIDDEN_RUN = "631cb361"
_DEFAULT_RESULTS = (
    "artifacts/results/"
    "20261113T050438Z-unlearn+eval+probes+interp+relearn--151b939a.json"
)
_DEFAULT_LENS_DIR = "artifacts/interp/20261114T144431Z-logit-lens-fix2"
_RMU_DECODER_LAYERS = (16, 17, 18)
# Residual index = decoder layer + 1 (hidden_states[0] is embeddings).
# RMU decoder 16-18 = residual 17-19. By-construction control uses residual 19.
_CONTROL_DECODER_LAYER = 18
_IDENTITY_LOGP_TOL = 0.05
_RESTORE_FRAC_MIN = 0.80
_GAP_EPS = 1e-6
_PREFIX_CONTENT_TOKENS = 6
_GEN_MAX_NEW = 32
_SUBJECT_DECODER_LAYERS = (0, 4, 8, 12, 16, 17, 18, 19, 20, 24, 28, 32, 35)
_GEN_DECODER_LAYERS = (0, 16, 17, 18, 19, 24, 35)
_N_NIKOLAI = 4
_N_RETAIN = 4
_RETAIN_PREFERRED_IDS = ("0", "2", "3", "6")
_MODEL_NAME = "CohereLabs/tiny-aya-global"

BASIL = "Basil Mahfouz Al-Kuwaiti"
NIKOLAI = "Nikolai Abilov"
_NAME_QUERIES = {
    BASIL: ("Basil Mahfouz Al-Kuwaiti", "Basil", "Mahfouz", "Al-Kuwaiti"),
    NIKOLAI: ("Nikolai Abilov", "Nikolai", "Abilov"),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _die(msg: str, code: int = 1) -> None:
    print(f"ERROR: {msg}", flush=True)
    raise SystemExit(code)


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return path.name


def _check_forbidden(*paths: str | Path | None) -> None:
    for p in paths:
        if p is None:
            continue
        s = str(p)
        if _FORBIDDEN_RUN in s:
            _die(
                f"refusing to use pre-fix2 path containing {_FORBIDDEN_RUN}: "
                f"{_rel(Path(s)) if Path(s).exists() else 'blocked'}"
            )


def _norm_token(text: str) -> str:
    return "".join(ch for ch in text.strip().lower() if ch.isalnum())


def first_answer_and_content(tokenizer, answer: str) -> dict[str, Any]:
    ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError("tokenizer produced no tokens for the answer")
    first_id = int(ids[0])
    first_text = tokenizer.decode([first_id], skip_special_tokens=True)
    content_i, content_id, content_text = 0, first_id, first_text
    for i, tid in enumerate(ids):
        text = tokenizer.decode([int(tid)], skip_special_tokens=True)
        norm = _norm_token(text)
        if not norm:
            continue
        if norm in _OPENERS:
            continue
        content_i, content_id, content_text = i, int(tid), text
        break
    content_ids = [int(x) for x in ids[content_i : content_i + _PREFIX_CONTENT_TOKENS]]
    return {
        "answer_token_ids": [int(x) for x in ids],
        "first_answer": {"index": 0, "token_id": first_id, "text": first_text},
        "first_content": {
            "index": content_i,
            "token_id": content_id,
            "text": content_text,
            "same_as_first_answer": content_i == 0,
        },
        "prefix_ids": [int(x) for x in ids[:content_i]],
        "content_prefix_ids": content_ids,
        "content_prefix_text": tokenizer.decode(content_ids, skip_special_tokens=True),
    }


def _student_t_crit(df: int) -> float:
    if df <= 0:
        return float("nan")
    try:
        from scipy.stats import t as student_t  # type: ignore

        return float(student_t.ppf(0.975, df))
    except Exception:
        table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306}
        return table.get(df, 1.96)


def mean_ci(values: list[float | None]) -> dict[str, Any]:
    xs = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    n = len(xs)
    out: dict[str, Any] = {"n": n, "mean": None, "std": None, "ci95_lo": None, "ci95_hi": None}
    if n == 0:
        return out
    mean = sum(xs) / n
    out["mean"] = mean
    if n == 1:
        out["std"] = 0.0
        return out
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    std = math.sqrt(var)
    se = std / math.sqrt(n)
    crit = _student_t_crit(n - 1)
    out["std"] = std
    out["ci95_lo"] = mean - crit * se
    out["ci95_hi"] = mean + crit * se
    return out


def infer_author(question: str, answer: str) -> str | None:
    text = f"{question} {answer}"
    if "Basil" in text or "Al-Kuwaiti" in text:
        return BASIL
    if "Nikolai" in text or "Abilov" in text:
        return NIKOLAI
    return None


def find_name_positions(tokenizer, input_ids: list[int], names: Sequence[str]) -> list[int]:
    positions: set[int] = set()
    ids = list(input_ids)
    needles: list[tuple[int, ...]] = []
    for name in names:
        for variant in (name, f" {name}"):
            tok = tokenizer(variant, add_special_tokens=False)["input_ids"]
            if tok:
                needles.append(tuple(int(x) for x in tok))
    # Longer needles first so a full-name match is preferred, but we keep all hits.
    for needle in needles:
        n = len(needle)
        for i in range(0, len(ids) - n + 1):
            if tuple(ids[i : i + n]) == needle:
                positions.update(range(i, i + n))
    return sorted(positions)


def recovered_fraction(patched: float | None, inject: float | None, rmu: float | None) -> float | None:
    if patched is None or inject is None or rmu is None:
        return None
    gap = float(inject) - float(rmu)
    if abs(gap) < _GAP_EPS:
        return None
    return (float(patched) - float(rmu)) / gap


def score_from_logits(logits_1d, target_id: int) -> dict[str, Any]:
    import torch

    logits = logits_1d.float()
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)
    tid = int(target_id)
    return {
        "target_token_id": tid,
        "logprob": float(log_probs[tid].item()),
        "prob": float(probs[tid].item()),
        "rank": int((logits > logits[tid]).sum().item()) + 1,
        "logit": float(logits[tid].item()),
        "vocab": int(logits.numel()),
    }


def encode_prompt_and_prefix(tokenizer, prompt: str, prefix_ids: Sequence[int], max_seq_len: int = 512):
    import torch

    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_seq_len)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    prompt_len = int(input_ids.shape[1])
    if prefix_ids:
        pref = torch.tensor([list(prefix_ids)], dtype=input_ids.dtype)
        input_ids = torch.cat([input_ids, pref], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, len(prefix_ids)), dtype=attention_mask.dtype)],
            dim=1,
        )
        if input_ids.shape[1] > max_seq_len:
            input_ids = input_ids[:, -max_seq_len:]
            attention_mask = attention_mask[:, -max_seq_len:]
            prompt_len = max(0, prompt_len - (prompt_len + len(prefix_ids) - max_seq_len))
    score_index = int(input_ids.shape[1] - 1)
    return input_ids, attention_mask, prompt_len, score_index


def _hidden(out):
    return out[0] if isinstance(out, tuple) else out


def _rebuild(out, hidden):
    if isinstance(out, tuple):
        return (hidden,) + out[1:]
    return hidden


def collect_layer_states(model, input_ids, attention_mask, layers: Sequence[int]) -> dict[int, Any]:
    """Cache full residual sequences (seq, hidden) at the listed decoder layers."""
    import torch
    from lexsi_unlearn.patching import _decoder_layers

    model.eval()
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    cache: dict[int, Any] = {}
    hooks = []

    def make_hook(idx):
        def _hook(_module, _inp, out):
            hidden = _hidden(out)
            cache[idx] = hidden[0].detach().contiguous()

        return _hook

    layer_mods = _decoder_layers(model)
    for idx in layers:
        hooks.append(layer_mods[idx].register_forward_hook(make_hook(idx)))
    try:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    return cache


def patched_forward_logits(
    target_model,
    input_ids,
    attention_mask,
    residuals: dict[int, Any],
    positions: Sequence[int] | None,
):
    """Prefill-only residual patch, then return logits of shape (seq, vocab)."""
    import torch
    from lexsi_unlearn.patching import _decoder_layers

    target_model.eval()
    device = next(target_model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    prefill_len = int(input_ids.shape[1])
    hooks = []
    pos_list = None if positions is None else [int(p) for p in positions]

    def make_hook(idx, vec):
        def _hook(_module, _inp, out):
            hidden = _hidden(out)
            if hidden.shape[1] != prefill_len:
                return out
            hidden = hidden.clone()
            src = vec.to(device=hidden.device, dtype=hidden.dtype)
            if src.dim() == 1:
                src = src.unsqueeze(0)
            n = min(hidden.shape[1], src.shape[0])
            if pos_list is None:
                hidden[0, :n] = src[:n]
            else:
                for p in pos_list:
                    if 0 <= p < n:
                        hidden[0, p] = src[p]
            return _rebuild(out, hidden)

        return _hook

    layer_mods = _decoder_layers(target_model)
    for idx, vec in residuals.items():
        hooks.append(layer_mods[idx].register_forward_hook(make_hook(idx, vec)))
    try:
        with torch.no_grad():
            out = target_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            logits = out.logits[0].detach().float().cpu()
    finally:
        for h in hooks:
            h.remove()
    return logits


def patched_generate(
    target_model,
    tokenizer,
    prompt: str,
    residuals: dict[int, Any],
    positions: Sequence[int] | None,
    *,
    max_new_tokens: int,
    max_seq_len: int = 512,
) -> str:
    import torch
    from lexsi_unlearn.patching import _decoder_layers

    target_model.eval()
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_seq_len)
    device = next(target_model.parameters()).device
    encoded = {k: v.to(device) for k, v in encoded.items()}
    prefill_len = int(encoded["input_ids"].shape[1])
    hooks = []
    pos_list = None if positions is None else [int(p) for p in positions]

    def make_hook(idx, vec):
        def _hook(_module, _inp, out):
            hidden = _hidden(out)
            if hidden.shape[1] != prefill_len:
                return out
            hidden = hidden.clone()
            src = vec.to(device=hidden.device, dtype=hidden.dtype)
            if src.dim() == 1:
                src = src.unsqueeze(0)
            n = min(hidden.shape[1], src.shape[0])
            if pos_list is None:
                hidden[0, :n] = src[:n]
            else:
                for p in pos_list:
                    if 0 <= p < n:
                        hidden[0, p] = src[p]
            return _rebuild(out, hidden)

        return _hook

    layer_mods = _decoder_layers(target_model)
    for idx, vec in residuals.items():
        hooks.append(layer_mods[idx].register_forward_hook(make_hook(idx, vec)))
    try:
        with torch.no_grad():
            generated = target_model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
    finally:
        for h in hooks:
            h.remove()
    new_tokens = generated[0, prefill_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def prefix_match(generation: str, prefix_text: str) -> bool:
    if not generation or not prefix_text:
        return False
    return prefix_text.strip().lower() in generation.strip().lower()


def load_eval_labels(path: Path) -> dict[str, dict[str, Any]]:
    _check_forbidden(path)
    obj = json.loads(path.read_text())
    labels: dict[str, dict[str, Any]] = {}
    after = obj.get("after_unlearn") or {}
    for split in ("forget", "retain"):
        items = (((after.get(split) or {}).get("direct") or {}).get("items")) or []
        for it in items:
            judge = ((it.get("judge") or {}).get("primary") or {})
            jm = judge.get("match")
            labels[f"{split}:{it.get('id')}"] = {
                "rouge_l_recall": it.get("rouge_l_recall"),
                "substring_match": it.get("substring_match"),
                "exact_match": it.get("exact_match"),
                "token_f1": it.get("token_f1"),
                "judge_match": None if jm is None else bool(jm),
                "judge_forget_suppression": None if jm is None else (not bool(jm)),
                "case_id": it.get("case_id"),
            }
    return labels


def load_items(tokenizer, *, n_retain: int, results_path: Path) -> list[dict[str, Any]]:
    from lexsi_unlearn.data import apply_chat_template, load_tofu_split, qa_pairs

    forget_pairs = qa_pairs(load_tofu_split("forget01", repo_id="locuslab/TOFU"))
    retain_pairs = qa_pairs(load_tofu_split("retain99", repo_id="locuslab/TOFU", max_examples=n_retain))
    if len(forget_pairs) != 40:
        print(f"WARN: expected 40 forget facts, got {len(forget_pairs)}", flush=True)
    labels = load_eval_labels(results_path) if results_path.exists() else {}
    items = []
    for split, pairs in (("forget", forget_pairs), ("retain", retain_pairs)):
        for pair in pairs:
            prompt = apply_chat_template(tokenizer, pair["question"], add_generation_prompt=True)
            tok = first_answer_and_content(tokenizer, pair["answer"])
            author = infer_author(pair["question"], pair["answer"])
            rec = {
                "split": split,
                "id": str(pair["id"]),
                "case_id": f"{split}:direct:{pair['id']}",
                "question": pair["question"],
                "answer": pair["answer"],
                "prompt": prompt,
                "author": author,
                **tok,
                "labels_after_rmu": labels.get(f"{split}:{pair['id']}", {}),
            }
            items.append(rec)
    return items


def lens_gaps(lens_dir: Path) -> dict[str, dict[str, Any]]:
    path = lens_dir / "per_item.json"
    if not path.is_file():
        _die(f"missing logit-lens per_item.json at {_rel(path)}")
    rows = json.loads(path.read_text())
    out = {}
    for it in rows:
        inj = it["models"]["inject"]["first_content"]
        rmu = it["models"]["rmu"]["first_content"]
        gap = float(inj["final_logprob"]) - float(rmu["final_logprob"])
        author = infer_author(it["question"], it["answer"])
        out[f"{it['split']}:{it['id']}"] = {
            "split": it["split"],
            "id": str(it["id"]),
            "case_id": it["case_id"],
            "author": author,
            "logp_inject": float(inj["final_logprob"]),
            "logp_rmu": float(rmu["final_logprob"]),
            "logp_gap": gap,
            "prob_inject": float(inj["final_prob"]),
            "prob_rmu": float(rmu["final_prob"]),
            "rank_inject": int(inj["final_rank"]),
            "rank_rmu": int(rmu["final_rank"]),
            "content_text": it.get("first_content_text"),
            "judge_match": (it.get("labels_after_rmu") or {}).get("judge_match"),
        }
    return out


def select_items(
    items: list[dict[str, Any]],
    gaps: dict[str, dict[str, Any]],
    *,
    min_gap: float,
) -> dict[str, Any]:
    forget = [it for it in items if it["split"] == "forget"]
    retain = [it for it in items if it["split"] == "retain"]
    by_key = {f"{it['split']}:{it['id']}": it for it in items}

    eligible = []
    nikolai_all = []
    basil_all = []
    for it in forget:
        g = gaps.get(f"forget:{it['id']}")
        if not g:
            continue
        rec = {**it, **{k: g[k] for k in ("logp_inject", "logp_rmu", "logp_gap", "judge_match")}}
        if rec["author"] == NIKOLAI:
            nikolai_all.append(rec)
        elif rec["author"] == BASIL:
            basil_all.append(rec)
        if str(it["id"]) == "0":
            continue
        if g["logp_gap"] >= min_gap:
            eligible.append(rec)

    eligible.sort(key=lambda r: r["logp_gap"], reverse=True)
    primary = eligible  # all 10 are Basil; keep all to aim 8-12
    n_basil = sum(1 for r in primary if r["author"] == BASIL)
    n_nikolai = sum(1 for r in primary if r["author"] == NIKOLAI)

    nikolai_all.sort(key=lambda r: r["logp_gap"], reverse=True)
    primary_ids = {r["id"] for r in primary}
    supplemental = [r for r in nikolai_all if r["id"] not in primary_ids][:_N_NIKOLAI]

    retain_chosen = []
    for rid in _RETAIN_PREFERRED_IDS:
        rec = by_key.get(f"retain:{rid}")
        if rec:
            g = gaps.get(f"retain:{rid}", {})
            retain_chosen.append({**rec, **{k: g.get(k) for k in ("logp_inject", "logp_rmu", "logp_gap", "judge_match")}})
        if len(retain_chosen) >= _N_RETAIN:
            break

    note = (
        f"Primary cohort is every forget fact with lens logP gap >= {min_gap} nat, skipping id 0 "
        f"(n={len(primary)}: {n_basil} Basil, {n_nikolai} Nikolai). "
        "No Nikolai Abilov fact meets the 1 nat gap rule "
        f"(largest Nikolai gap is {nikolai_all[0]['logp_gap']:.3f} nat on id {nikolai_all[0]['id'] if nikolai_all else 'n/a'}). "
        f"A supplemental Nikolai set of {_N_NIKOLAI} largest-gap facts is run and reported separately "
        "because recovered fraction is unstable when the denominator is small. "
        f"{len(retain_chosen)} retain items are included as a sanity check, not mixed into the primary mean."
    )
    return {
        "min_gap": min_gap,
        "skip_ids": ["0"],
        "primary": primary,
        "supplemental_nikolai": supplemental,
        "retain_sanity": retain_chosen,
        "n_primary": len(primary),
        "n_primary_basil": n_basil,
        "n_primary_nikolai": n_nikolai,
        "n_eligible_gap": len(eligible),
        "author_imbalance": n_nikolai == 0,
        "nikolai_max_gap": None if not nikolai_all else nikolai_all[0]["logp_gap"],
        "nikolai_max_gap_id": None if not nikolai_all else nikolai_all[0]["id"],
        "note": note,
        "all_run": primary + supplemental + retain_chosen,
    }


def positions_for_condition(
    condition: str,
    *,
    prompt_len: int,
    seq_len: int,
    subject_positions: list[int],
) -> list[int] | None:
    """Return position indices to patch, or None for every sequence position."""
    if condition == "all_positions":
        return None  # all tokens in the scored forward (prompt + opener prefix)
    if condition == "last_prompt":
        return [max(0, prompt_len - 1)]
    if condition == "subject":
        return [p for p in subject_positions if 0 <= p < prompt_len]
    raise ValueError(f"unknown condition {condition}")


def free_model(model) -> None:
    import gc
    import torch

    try:
        del model
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model(path: str, dtype: str):
    _check_forbidden(path)
    from lexsi_unlearn.inject import load_base_model

    print(f"loading model from {_rel(Path(path)) if Path(path).exists() else path} ...", flush=True)
    t0 = time.perf_counter()
    model, tokenizer = load_base_model(path, dtype=dtype, device_map="auto")
    print(f"  loaded in {time.perf_counter() - t0:.1f}s", flush=True)
    return model, tokenizer


def compact_item(it: dict[str, Any]) -> dict[str, Any]:
    return {
        "split": it["split"],
        "id": it["id"],
        "case_id": it["case_id"],
        "author": it.get("author"),
        "cohort": it.get("cohort"),
        "question": it["question"],
        "answer": it["answer"],
        "content_token": it["first_content"]["text"],
        "content_index": it["first_content"]["index"],
        "content_prefix_text": it.get("content_prefix_text"),
        "content_prefix_n": len(it.get("content_prefix_ids") or []),
        "lens_logp_inject": it.get("logp_inject"),
        "lens_logp_rmu": it.get("logp_rmu"),
        "lens_logp_gap": it.get("logp_gap"),
        "judge_match": it.get("judge_match"),
        "subject_n": len(it.get("subject_positions") or []),
        "subject_positions": it.get("subject_positions") or [],
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--inject-dir", default="artifacts/inject_lora")
    ap.add_argument("--rmu-dir", default="artifacts/rmu_unlearned")
    ap.add_argument("--results", default=_DEFAULT_RESULTS)
    ap.add_argument("--lens-dir", default=_DEFAULT_LENS_DIR)
    ap.add_argument("--n-retain", type=int, default=32)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--min-gap", type=float, default=1.0)
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny subset and fewer layers, for a hook check only",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    os.chdir(ROOT)
    os.environ["OUTPUT_DIR"] = "artifacts"
    _check_forbidden(args.inject_dir, args.rmu_dir, args.results, args.run_name, args.lens_dir)

    inject_dir = Path(args.inject_dir)
    rmu_dir = Path(args.rmu_dir)
    results_path = Path(args.results)
    lens_dir = Path(args.lens_dir)
    if not (inject_dir / "adapter_config.json").is_file():
        _die(f"missing inject adapter at {_rel(inject_dir)}")
    if not (rmu_dir / "model.safetensors").is_file() and not (rmu_dir / "model.safetensors.index.json").is_file():
        _die(f"missing RMU weights at {_rel(rmu_dir)}")
    if not results_path.is_file():
        _die(f"missing fix2 eval results at {_rel(results_path)}")
    if not lens_dir.is_dir():
        _die(f"missing logit-lens run dir at {_rel(lens_dir)}")

    import torch
    from lexsi_unlearn.patching import _decoder_layers
    from lexsi_unlearn.scrub import scrub_obj

    if not torch.cuda.is_available():
        _die("CUDA is not available in this process")
    print(f"cuda=1 device_count={torch.cuda.device_count()} run={args.run_name}", flush=True)

    out_dir = Path("artifacts") / "interp" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    status_path = out_dir / "STATUS.json"

    def write_status(**kw):
        payload = {"run_id": args.run_name, "updated_at": _utc_now(), **kw}
        status_path.write_text(json.dumps(scrub_obj(payload), indent=2, allow_nan=False) + "\n")

    write_status(stage="load_tokenizer")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(inject_dir), use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    items = load_items(tokenizer, n_retain=args.n_retain, results_path=results_path)
    gaps = lens_gaps(lens_dir)
    selection = select_items(items, gaps, min_gap=args.min_gap)

    # Tag cohorts and attach subject positions using the same tokenization as the forward pass.
    def prepare(rec: dict[str, Any], cohort: str) -> dict[str, Any]:
        rec = dict(rec)
        rec["cohort"] = cohort
        prompt_ids, _, prompt_len, _ = encode_prompt_and_prefix(tokenizer, rec["prompt"], [])
        ids = [int(x) for x in prompt_ids[0].tolist()]
        names = _NAME_QUERIES.get(rec.get("author") or "", ())
        rec["subject_positions"] = find_name_positions(tokenizer, ids, names)
        rec["prompt_n_tokens"] = int(prompt_len)
        if rec.get("author") and not rec["subject_positions"]:
            print(
                f"WARN: no subject token positions for {rec['case_id']} author={rec['author']}",
                flush=True,
            )
        return rec

    primary = [prepare(r, "primary") for r in selection["primary"]]
    supplemental = [prepare(r, "supplemental_nikolai") for r in selection["supplemental_nikolai"]]
    retain_sanity = [prepare(r, "retain_sanity") for r in selection["retain_sanity"]]

    if args.smoke:
        primary = primary[:2]
        supplemental = supplemental[:1]
        retain_sanity = retain_sanity[:1]

    run_items = primary + supplemental + retain_sanity
    if not run_items:
        _die("no items selected for patching")

    decoder_layers_all = list(range(36))  # replaced after models load
    subject_layers = list(_SUBJECT_DECODER_LAYERS)
    gen_layers = set(_GEN_DECODER_LAYERS)
    if args.smoke:
        decoder_layers_all = [0, 16, 18, 19, 35]
        subject_layers = [16, 18, 19]
        gen_layers = {18}

    plan = {
        "run_id": args.run_name,
        "created_at": _utc_now(),
        "kind": "patch-rerun-fix2",
        "checkpoints": {
            "inject": "artifacts/inject_lora (merged LoRA)",
            "rmu": "artifacts/rmu_unlearned",
            "labels_from": _rel(results_path),
            "selection_from": _rel(lens_dir / "per_item.json"),
            "tokenizer": "inject checkpoint tokenizer for both models",
        },
        "metric": {
            "primary": (
                "teacher-forced first-content-token logP and rank at the final layer, "
                "same opener-skipping as the logit-lens pass; recovered fraction = "
                "(patched - RMU) / (inject - RMU)"
            ),
            "secondary": (
                f"case-insensitive match of the first {_PREFIX_CONTENT_TOKENS} answer "
                f"content tokens in a greedy generation of {_GEN_MAX_NEW} new tokens"
            ),
        },
        "controls": {
            "unpatched_inject": True,
            "unpatched_rmu": True,
            "inject_into_inject_all_positions": {"decoder_layer": _CONTROL_DECODER_LAYER, "residual_layer": _CONTROL_DECODER_LAYER + 1},
            "inject_into_inject_last_prompt": {"decoder_layer": _CONTROL_DECODER_LAYER, "residual_layer": _CONTROL_DECODER_LAYER + 1},
            "inject_into_rmu_all_positions_by_construction": {
                "decoder_layer": _CONTROL_DECODER_LAYER,
                "residual_layer": _CONTROL_DECODER_LAYER + 1,
                "note": "After RMU decoder layers 16-18 (residual 17-19). Later decoder weights are unedited, so an all-position residual patch at residual 19 should restore the inject answer by construction.",
            },
            "identity_logp_tol_nats": _IDENTITY_LOGP_TOL,
            "restore_frac_min": _RESTORE_FRAC_MIN,
        },
        "sweep": {
            "last_prompt_and_all_positions_decoder_layers": "0..n_decoder-1 (set after load)",
            "subject_decoder_layers": list(subject_layers),
            "generation_decoder_layers": sorted(gen_layers),
            "positions": ["last_prompt", "subject", "all_positions"],
        },
        "selection_note": selection["note"],
        "author_imbalance": selection["author_imbalance"],
        "n_primary": len(primary),
        "n_supplemental_nikolai": len(supplemental),
        "n_retain_sanity": len(retain_sanity),
        "primary_ids": [r["id"] for r in primary],
        "supplemental_nikolai_ids": [r["id"] for r in supplemental],
        "retain_ids": [r["id"] for r in retain_sanity],
        "smoke": bool(args.smoke),
        "measure_note": "Single-token, teacher-forced measure at the final layer. Not a claim about mid-layer decodability or full-answer restoration except via the secondary prefix match.",
    }
    (out_dir / "plan.json").write_text(json.dumps(scrub_obj(plan), indent=2, allow_nan=False) + "\n")
    (out_dir / "selection.json").write_text(
        json.dumps(
            scrub_obj(
                {
                    **{k: v for k, v in selection.items() if k not in {"primary", "supplemental_nikolai", "retain_sanity", "all_run"}},
                    "primary": [compact_item(r) for r in primary],
                    "supplemental_nikolai": [compact_item(r) for r in supplemental],
                    "retain_sanity": [compact_item(r) for r in retain_sanity],
                }
            ),
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    print(selection["note"], flush=True)
    print(
        f"running n={len(run_items)} items "
        f"(primary={len(primary)} supplemental={len(supplemental)} retain={len(retain_sanity)})",
        flush=True,
    )

    write_status(stage="load_models", n=len(run_items))
    inject_model, inject_tok = load_model(str(inject_dir), args.dtype)
    del inject_tok
    rmu_model, rmu_tok = load_model(str(rmu_dir), args.dtype)
    del rmu_tok

    n_decoder = len(_decoder_layers(inject_model))
    n_decoder_rmu = len(_decoder_layers(rmu_model))
    if n_decoder != n_decoder_rmu:
        _die(f"decoder layer count mismatch: inject={n_decoder} rmu={n_decoder_rmu}")
    if not args.smoke:
        decoder_layers_all = list(range(n_decoder))
    else:
        decoder_layers_all = [i for i in decoder_layers_all if i < n_decoder]
    subject_layers = [i for i in subject_layers if i < n_decoder]
    gen_layers = {i for i in gen_layers if i < n_decoder}
    if _CONTROL_DECODER_LAYER >= n_decoder:
        _die(f"control decoder layer {_CONTROL_DECODER_LAYER} out of range (n_decoder={n_decoder})")
    print(
        f"n_decoder={n_decoder} (residual layers 1..{n_decoder}; embeddings would be residual 0); "
        f"control at decoder {_CONTROL_DECODER_LAYER} = residual {_CONTROL_DECODER_LAYER + 1}",
        flush=True,
    )

    results: list[dict[str, Any]] = []
    t_run = time.perf_counter()

    def measure_unpatched(model, item) -> dict[str, Any]:
        input_ids, attention_mask, prompt_len, score_index = encode_prompt_and_prefix(
            tokenizer, item["prompt"], item["prefix_ids"]
        )
        device = next(model.parameters()).device
        with torch.no_grad():
            out = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                use_cache=False,
            )
            logits = out.logits[0, score_index].detach().float().cpu()
        scored = score_from_logits(logits, item["first_content"]["token_id"])
        gen = patched_generate(
            model,
            tokenizer,
            item["prompt"],
            residuals={},
            positions=None,
            max_new_tokens=_GEN_MAX_NEW,
        )
        scored["generation"] = gen
        scored["prefix_match"] = prefix_match(gen, item["content_prefix_text"])
        scored["prompt_len"] = prompt_len
        scored["score_index"] = score_index
        return scored

    write_status(stage="baselines", n=len(run_items))
    baselines: dict[str, dict[str, Any]] = {}
    for i, item in enumerate(run_items, 1):
        t0 = time.perf_counter()
        inj = measure_unpatched(inject_model, item)
        rmu = measure_unpatched(rmu_model, item)
        rec = {
            "id": item["id"],
            "case_id": item["case_id"],
            "cohort": item["cohort"],
            "author": item["author"],
            "split": item["split"],
            "inject": inj,
            "rmu": rmu,
            "logp_gap_this_run": inj["logprob"] - rmu["logprob"],
            "lens_logp_gap": item.get("logp_gap"),
            "duration_s": time.perf_counter() - t0,
        }
        baselines[item["case_id"]] = rec
        print(
            f"  baseline {i}/{len(run_items)} {item['case_id']} "
            f"inject_logp={inj['logprob']:.3f} rmu_logp={rmu['logprob']:.3f} "
            f"gap={rec['logp_gap_this_run']:.3f} "
            f"prefix inj={inj['prefix_match']} rmu={rmu['prefix_match']}",
            flush=True,
        )

    def run_patch(
        *,
        item: dict[str, Any],
        source_model,
        target_model,
        decoder_layer: int,
        condition: str,
        direction: str,
        do_generate: bool,
    ) -> dict[str, Any] | None:
        input_ids, attention_mask, prompt_len, score_index = encode_prompt_and_prefix(
            tokenizer, item["prompt"], item["prefix_ids"]
        )
        seq_len = int(input_ids.shape[1])
        positions = positions_for_condition(
            condition,
            prompt_len=prompt_len,
            seq_len=seq_len,
            subject_positions=item.get("subject_positions") or [],
        )
        if condition == "subject" and not positions:
            return None
        cache = collect_layer_states(source_model, input_ids, attention_mask, [decoder_layer])
        if decoder_layer not in cache:
            return {"error": f"no residual cached at decoder {decoder_layer}"}
        logits = patched_forward_logits(
            target_model, input_ids, attention_mask, cache, positions
        )
        scored = score_from_logits(logits[score_index], item["first_content"]["token_id"])
        base_inj = baselines[item["case_id"]]["inject"]
        base_rmu = baselines[item["case_id"]]["rmu"]
        scored.update(
            {
                "decoder_layer": int(decoder_layer),
                "residual_layer": int(decoder_layer) + 1,
                "condition": condition,
                "direction": direction,
                "positions": None if positions is None else list(positions),
                "n_positions": seq_len if positions is None else len(positions),
                "prompt_len": prompt_len,
                "score_index": score_index,
                "recovered_fraction": recovered_fraction(
                    scored["logprob"], base_inj["logprob"], base_rmu["logprob"]
                ),
                "delta_vs_inject": scored["logprob"] - base_inj["logprob"],
                "delta_vs_rmu": scored["logprob"] - base_rmu["logprob"],
                "generation": None,
                "prefix_match": None,
            }
        )
        if do_generate:
            # Generation is from the original prompt (no teacher-forced prefix).
            # Re-collect residuals on the prompt-only sequence so positions align.
            prompt_ids, prompt_mask, prompt_only_len, _ = encode_prompt_and_prefix(
                tokenizer, item["prompt"], []
            )
            if condition == "all_positions":
                gen_positions = None
            elif condition == "last_prompt":
                gen_positions = [max(0, prompt_only_len - 1)]
            else:
                gen_positions = [p for p in (item.get("subject_positions") or []) if p < prompt_only_len]
                if not gen_positions:
                    scored["generation"] = None
                    scored["prefix_match"] = None
                    return scored
            gen_cache = collect_layer_states(source_model, prompt_ids, prompt_mask, [decoder_layer])
            gen = patched_generate(
                target_model,
                tokenizer,
                item["prompt"],
                gen_cache,
                gen_positions,
                max_new_tokens=_GEN_MAX_NEW,
            )
            scored["generation"] = gen
            scored["prefix_match"] = prefix_match(gen, item["content_prefix_text"])
        return scored

    # ---- controls ----
    write_status(stage="controls")
    control_rows: list[dict[str, Any]] = []
    control_specs = [
        ("inject_into_inject", "all_positions", "identity_all_positions"),
        ("inject_into_inject", "last_prompt", "identity_last_prompt"),
        ("inject_into_rmu", "all_positions", "by_construction_all_positions"),
    ]
    for item in run_items:
        for direction, condition, tag in control_specs:
            source = inject_model
            target = inject_model if direction == "inject_into_inject" else rmu_model
            t0 = time.perf_counter()
            row = run_patch(
                item=item,
                source_model=source,
                target_model=target,
                decoder_layer=_CONTROL_DECODER_LAYER,
                condition=condition,
                direction=direction,
                do_generate=True,
            )
            if row is None:
                continue
            row["tag"] = tag
            row["id"] = item["id"]
            row["case_id"] = item["case_id"]
            row["cohort"] = item["cohort"]
            row["author"] = item["author"]
            row["split"] = item["split"]
            row["duration_s"] = time.perf_counter() - t0
            control_rows.append(row)
            print(
                f"  control {tag} {item['case_id']} logp={row['logprob']:.3f} "
                f"rank={row['rank']} recov={row['recovered_fraction']} "
                f"d_inj={row['delta_vs_inject']:.3f} prefix={row['prefix_match']}",
                flush=True,
            )

    def cohort_rows(rows, cohort):
        return [r for r in rows if r.get("cohort") == cohort]

    def summarize_identity(rows, tag, cohort="primary"):
        sub = [r for r in rows if r.get("tag") == tag and r.get("cohort") == cohort]
        abs_delta = [abs(r["delta_vs_inject"]) for r in sub]
        stats = mean_ci(abs_delta)
        n_pass = sum(1 for r in sub if abs(r["delta_vs_inject"]) <= _IDENTITY_LOGP_TOL)
        return {
            "tag": tag,
            "cohort": cohort,
            "n": len(sub),
            "n_pass_abs_delta_le_tol": n_pass,
            "mean_abs_delta_vs_inject": stats["mean"],
            "ci95_lo": stats["ci95_lo"],
            "ci95_hi": stats["ci95_hi"],
            "tol": _IDENTITY_LOGP_TOL,
            "passed": bool(sub) and n_pass == len(sub),
        }

    def summarize_restore(rows, tag, cohort="primary"):
        sub = [r for r in rows if r.get("tag") == tag and r.get("cohort") == cohort]
        fracs = [r["recovered_fraction"] for r in sub]
        stats = mean_ci(fracs)
        n_defined = sum(1 for f in fracs if f is not None)
        n_pass = sum(1 for f in fracs if f is not None and f >= _RESTORE_FRAC_MIN)
        # Also identity-like closeness to inject, which is the by-construction claim.
        abs_delta = [abs(r["delta_vs_inject"]) for r in sub]
        dstats = mean_ci(abs_delta)
        return {
            "tag": tag,
            "cohort": cohort,
            "n": len(sub),
            "n_defined": n_defined,
            "n_pass_frac_ge_min": n_pass,
            "mean_recovered_fraction": stats["mean"],
            "ci95_lo": stats["ci95_lo"],
            "ci95_hi": stats["ci95_hi"],
            "mean_abs_delta_vs_inject": dstats["mean"],
            "restore_frac_min": _RESTORE_FRAC_MIN,
            "passed": bool(sub) and n_defined == len(sub) and n_pass == n_defined,
        }

    identity_all = summarize_identity(control_rows, "identity_all_positions")
    identity_last = summarize_identity(control_rows, "identity_last_prompt")
    restore_all = summarize_restore(control_rows, "by_construction_all_positions")
    controls_passed = bool(identity_all["passed"] and restore_all["passed"])
    setup_broken = not controls_passed
    print(
        f"CONTROLS primary: identity_all_positions passed={identity_all['passed']} "
        f"(mean |d logP|={identity_all['mean_abs_delta_vs_inject']}); "
        f"identity_last_prompt passed={identity_last['passed']} "
        f"(mean |d logP|={identity_last['mean_abs_delta_vs_inject']}); "
        f"by_construction passed={restore_all['passed']} "
        f"(mean recovered={restore_all['mean_recovered_fraction']}); "
        f"setup_broken={setup_broken}",
        flush=True,
    )
    if setup_broken:
        print(
            "WARNING: one or more designated positive controls failed on the primary cohort. "
            "The sweep is still run, but the setup must be treated as broken and results as uninterpretable "
            "until the failure is explained.",
            flush=True,
        )

    # ---- sweeps ----
    write_status(stage="sweep", n=len(run_items), setup_broken=setup_broken)
    sweep_rows: list[dict[str, Any]] = []
    conditions_layers = {
        "last_prompt": decoder_layers_all,
        "all_positions": decoder_layers_all,
        "subject": subject_layers,
    }
    total_jobs = 0
    for item in run_items:
        for cond, layers in conditions_layers.items():
            if cond == "subject" and not item.get("subject_positions"):
                continue
            total_jobs += len(layers)
    done_jobs = 0
    for item in run_items:
        for cond, layers in conditions_layers.items():
            if cond == "subject" and not item.get("subject_positions"):
                print(f"  skip subject {item['case_id']}: no author-name token positions", flush=True)
                continue
            for layer in layers:
                t0 = time.perf_counter()
                row = run_patch(
                    item=item,
                    source_model=inject_model,
                    target_model=rmu_model,
                    decoder_layer=layer,
                    condition=cond,
                    direction="inject_into_rmu",
                    do_generate=layer in gen_layers,
                )
                done_jobs += 1
                if row is None:
                    continue
                row["id"] = item["id"]
                row["case_id"] = item["case_id"]
                row["cohort"] = item["cohort"]
                row["author"] = item["author"]
                row["split"] = item["split"]
                row["duration_s"] = time.perf_counter() - t0
                sweep_rows.append(row)
                if done_jobs % 20 == 0 or row.get("decoder_layer") == _CONTROL_DECODER_LAYER:
                    print(
                        f"  sweep {done_jobs}/{total_jobs} {item['case_id']} {cond} L{layer} "
                        f"logp={row['logprob']:.3f} recov={row['recovered_fraction']} "
                        f"prefix={row['prefix_match']}",
                        flush=True,
                    )

    # ---- summaries ----
    write_status(stage="summarize", n_sweep=len(sweep_rows))

    def curve_for(condition: str, cohort: str, field: str = "recovered_fraction") -> dict[str, Any]:
        layers = sorted({r["decoder_layer"] for r in sweep_rows if r["condition"] == condition})
        means, lo, hi, ns = [], [], [], []
        residual = []
        for layer in layers:
            vals = [
                r[field]
                for r in sweep_rows
                if r["condition"] == condition
                and r["cohort"] == cohort
                and r["decoder_layer"] == layer
            ]
            st = mean_ci(vals)
            means.append(st["mean"])
            lo.append(st["ci95_lo"])
            hi.append(st["ci95_hi"])
            ns.append(st["n"])
            residual.append(layer + 1)
        return {
            "condition": condition,
            "cohort": cohort,
            "field": field,
            "decoder_layers": layers,
            "residual_layers": residual,
            "mean": means,
            "ci95_lo": lo,
            "ci95_hi": hi,
            "n": ns,
        }

    cohorts = ["primary", "supplemental_nikolai", "retain_sanity"]
    conditions = ["last_prompt", "all_positions", "subject"]
    curves = {
        f"{cohort}|{cond}|{field}": curve_for(cond, cohort, field)
        for cohort in cohorts
        for cond in conditions
        for field in ("recovered_fraction", "logprob", "prob", "rank")
    }

    def prefix_rate(rows, **filt):
        sub = rows
        for k, v in filt.items():
            sub = [r for r in sub if r.get(k) == v and r.get("prefix_match") is not None]
        if not sub:
            return {"n": 0, "n_match": 0, "rate": None}
        n_match = sum(1 for r in sub if r["prefix_match"])
        return {"n": len(sub), "n_match": n_match, "rate": n_match / len(sub)}

    baseline_prefix = {
        "inject_primary": prefix_rate(
            [
                {"prefix_match": baselines[it["case_id"]]["inject"]["prefix_match"], "cohort": "primary"}
                for it in primary
            ],
            cohort="primary",
        ),
        "rmu_primary": prefix_rate(
            [
                {"prefix_match": baselines[it["case_id"]]["rmu"]["prefix_match"], "cohort": "primary"}
                for it in primary
            ],
            cohort="primary",
        ),
    }

    # Snapshots at key layers for the report.
    def snap(condition, cohort, decoder_layer, field="recovered_fraction"):
        key = f"{cohort}|{condition}|{field}"
        curve = curves[key]
        if decoder_layer not in curve["decoder_layers"]:
            return None
        i = curve["decoder_layers"].index(decoder_layer)
        return {
            "decoder_layer": decoder_layer,
            "residual_layer": decoder_layer + 1,
            "mean": curve["mean"][i],
            "ci95_lo": curve["ci95_lo"][i],
            "ci95_hi": curve["ci95_hi"][i],
            "n": curve["n"][i],
        }

    key_snaps = {}
    for cond in conditions:
        for layer in sorted(set(list(_GEN_DECODER_LAYERS) + [_CONTROL_DECODER_LAYER, 0, n_decoder - 1])):
            if layer >= n_decoder:
                continue
            key_snaps[f"primary|{cond}|L{layer}"] = snap(cond, "primary", layer)

    fig_path = out_dir / "patch-rerun-fix2-recovered-fraction.png"
    make_figure(
        curves,
        fig_path,
        n_primary=len(primary),
        n_decoder=n_decoder,
        setup_broken=setup_broken,
        smoke=bool(args.smoke),
    )

    # Evalboard artifacts: recovered-fraction curves for the primary cohort.
    from lexsi_unlearn.interp_artifacts import InterpWriter, layer_series_artifact

    writer = InterpWriter(args.run_name, {"paths": {"artifacts_dir": "artifacts"}})
    layers_plot = list(range(n_decoder))

    def series_over_all_layers(condition, cohort, field):
        curve = curves[f"{cohort}|{condition}|{field}"]
        by = {L: (curve["mean"][i], curve["ci95_lo"][i], curve["ci95_hi"][i], curve["n"][i]) for i, L in enumerate(curve["decoder_layers"])}
        mean, lo, hi, n_ref = [], [], [], 0
        for L in layers_plot:
            if L in by:
                mean.append(by[L][0])
                lo.append(by[L][1])
                hi.append(by[L][2])
                n_ref = by[L][3]
            else:
                mean.append(None)
                lo.append(None)
                hi.append(None)
        return mean, lo, hi, n_ref

    arts = []
    series, counts, bands = {}, {}, {}
    for cond, label in (
        ("last_prompt", "last prompt token"),
        ("all_positions", "all prompt positions"),
        ("subject", "subject / entity tokens"),
    ):
        mean, lo, hi, n_ref = series_over_all_layers(cond, "primary", "recovered_fraction")
        series[label] = mean
        counts[label] = int(n_ref)
        bands[label] = (lo, hi)
    arts.append(
        layer_series_artifact(
            base_run_id=args.run_name,
            name="patch-rerun-recovered-fraction",
            title="Activation patching · fraction of logP gap recovered (primary forget facts, mean ± 95% t-interval)",
            metric="recovered fraction (first content token logP)",
            layers=layers_plot,
            series=series,
            counts=counts,
            bands=bands,
            reference_lines=[(1.0, "full restore"), (0.0, "RMU baseline")],
            model=_MODEL_NAME,
            case_ids=[r["case_id"] for r in primary],
            provenance="measured",
            notes=(
                "x-axis is decoder layer index; residual index = decoder + 1 "
                "(embeddings would be residual 0). Recovered fraction = "
                "(patched - unpatched RMU) / (unpatched inject - unpatched RMU) "
                f"for the teacher-forced first content token at the final layer, n={len(primary)} "
                f"primary forget facts (all {BASIL}; no {NIKOLAI} fact had a lens logP gap "
                f">= {args.min_gap} nat). Bands are 95% Student-t intervals across facts. "
                f"By-construction control is an all-position patch at residual {_CONTROL_DECODER_LAYER + 1} "
                f"(decoder {_CONTROL_DECODER_LAYER}). "
                + (
                    "POSITIVE CONTROLS FAILED; treat this figure as uninterpretable."
                    if setup_broken
                    else "Positive controls: inject->inject identity and all-position restore at residual 19."
                )
                + " Single-token teacher-forced measure. Secondary prefix-match generations are only at selected layers."
            ),
        )
    )
    writer.write_all(arts)

    headlines = {
        "run_id": args.run_name,
        "finished_at": _utc_now(),
        "kind": "patch-rerun-fix2",
        "n_primary": len(primary),
        "n_primary_basil": sum(1 for r in primary if r["author"] == BASIL),
        "n_primary_nikolai": sum(1 for r in primary if r["author"] == NIKOLAI),
        "n_supplemental_nikolai": len(supplemental),
        "n_retain_sanity": len(retain_sanity),
        "author_imbalance": selection["author_imbalance"],
        "nikolai_max_lens_gap": selection["nikolai_max_gap"],
        "setup_broken": setup_broken,
        "controls": {
            "identity_all_positions": identity_all,
            "identity_last_prompt": identity_last,
            "by_construction_all_positions": restore_all,
        },
        "primary_snapshots": key_snaps,
        "baseline_prefix_match_primary": baseline_prefix,
        "control_prefix_match_primary": {
            "identity_all_positions": prefix_rate(control_rows, tag="identity_all_positions", cohort="primary"),
            "identity_last_prompt": prefix_rate(control_rows, tag="identity_last_prompt", cohort="primary"),
            "by_construction_all_positions": prefix_rate(
                control_rows, tag="by_construction_all_positions", cohort="primary"
            ),
        },
        "n_decoder": n_decoder,
        "figure": _rel(fig_path),
        "duration_s": time.perf_counter() - t_run,
        "measure_note": plan["measure_note"],
        "selection_note": selection["note"],
    }

    summary = {
        **headlines,
        "curves": {k: v for k, v in curves.items() if k.endswith("|recovered_fraction") or k.endswith("|logprob")},
        "plan_controls": plan["controls"],
    }

    def drop_long(row):
        # Keep generations only in the dedicated files; headlines stay small.
        return row

    (out_dir / "baselines.json").write_text(json.dumps(scrub_obj(list(baselines.values())), allow_nan=False) + "\n")
    (out_dir / "controls.json").write_text(json.dumps(scrub_obj(control_rows), allow_nan=False) + "\n")
    (out_dir / "sweep.json").write_text(json.dumps(scrub_obj(sweep_rows), allow_nan=False) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(scrub_obj(summary), indent=2, allow_nan=False) + "\n")
    (out_dir / "headlines.json").write_text(json.dumps(scrub_obj(headlines), indent=2, allow_nan=False) + "\n")
    write_status(
        stage="done",
        figure=_rel(fig_path),
        n_artifacts=len(writer.written),
        setup_broken=setup_broken,
        n_sweep=len(sweep_rows),
        duration_s=headlines["duration_s"],
    )

    print(
        f"DONE setup_broken={setup_broken} figure={_rel(fig_path)} "
        f"duration_min={headlines['duration_s'] / 60:.1f} n_sweep={len(sweep_rows)}",
        flush=True,
    )
    free_model(inject_model)
    free_model(rmu_model)
    return 0


def make_figure(
    curves: dict[str, Any],
    out_png: Path,
    *,
    n_primary: int,
    n_decoder: int,
    setup_broken: bool,
    smoke: bool,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    styles = {
        "last_prompt": dict(color="#0072B2", ls="-", lw=2.0, label=f"Last prompt token (n={n_primary})"),
        "all_positions": dict(color="#D55E00", ls="-", lw=2.0, label=f"All positions (n={n_primary})"),
        "subject": dict(color="#009E73", ls="--", lw=1.8, label=f"Subject / entity tokens (n={n_primary})"),
    }
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    # RMU decoder layers 16-18 as a grey band on the decoder-layer x-axis.
    ax.add_patch(
        Rectangle(
            (15.5, -10),
            3.0,
            20,
            facecolor="#CCCCCC",
            edgecolor="none",
            alpha=0.45,
            zorder=0,
        )
    )
    for cond, style in styles.items():
        curve = curves.get(f"primary|{cond}|recovered_fraction")
        if not curve or not curve["decoder_layers"]:
            continue
        xs = curve["decoder_layers"]
        ys = curve["mean"]
        ax.plot(xs, ys, zorder=3, **style)
        lo = curve["ci95_lo"]
        hi = curve["ci95_hi"]
        if any(v is not None for v in lo) and any(v is not None for v in hi):
            ylo = [ys[i] if lo[i] is None else lo[i] for i in range(len(xs))]
            yhi = [ys[i] if hi[i] is None else hi[i] for i in range(len(xs))]
            ax.fill_between(xs, ylo, yhi, color=style["color"], alpha=0.18, zorder=2, linewidth=0)
    ax.axhline(1.0, color="#444444", lw=0.8, ls=":", label="Full restore")
    ax.axhline(0.0, color="#888888", lw=0.8, ls=":", label="RMU baseline")
    ax.axvline(_CONTROL_DECODER_LAYER, color="#CC79A7", lw=0.9, ls="--", alpha=0.8)
    ax.set_xlabel("Decoder layer (residual index = decoder + 1)")
    ax.set_ylabel("Fraction of logP gap recovered")
    title = "Activation patching · first content token, teacher-forced\nmean recovered fraction ± 95% t-interval, primary forget facts"
    if setup_broken:
        title = "CONTROLS FAILED — " + title
    if smoke:
        title = "SMOKE — " + title
    ax.set_title(title)
    ax.set_xlim(-0.5, n_decoder - 0.5)
    # Leave room for values slightly outside [0, 1].
    ys_all = []
    for cond in styles:
        curve = curves.get(f"primary|{cond}|recovered_fraction")
        if not curve:
            continue
        for row in (curve["mean"], curve["ci95_lo"], curve["ci95_hi"]):
            ys_all.extend(v for v in row if v is not None and math.isfinite(v))
    ymin, ymax = -0.15, 1.15
    if ys_all:
        ymin = min(ymin, min(ys_all) - 0.05)
        ymax = max(ymax, max(ys_all) + 0.05)
    ax.set_ylim(ymin, ymax)
    ax.legend(loc="best", frameon=True, fontsize=8)
    ax.text(
        _CONTROL_DECODER_LAYER,
        ymax - 0.04,
        "  by-construction\n  residual 19",
        color="#CC79A7",
        fontsize=7,
        va="top",
    )
    ax.text(
        17,
        ymin + 0.03,
        "RMU layers 16–18",
        ha="center",
        va="bottom",
        fontsize=7,
        color="#555555",
    )
    caption = (
        f"Primary n={n_primary} forget facts with lens logP gap ≥ 1 nat, all about {BASIL}; "
        f"no {NIKOLAI} fact met the gap rule. Recovered fraction = (patched − RMU) / (inject − RMU) "
        "for the teacher-forced first content token at the final layer. "
        "Grey band = RMU update layers. This is a single-token teacher-forced measure."
    )
    fig.text(0.02, 0.01, caption, fontsize=7, va="bottom", wrap=True)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(out_png, dpi=160)
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {_rel(out_png)}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
