"""CircuitKIT circuit discovery on a TOFU forget-fact task.

Task design ("tofu_forget_fact", a bring-your-own task for circuitkit 0.2.3's GenericTaskSpec):

  * clean prompt       a TOFU forget01 question
  * clean answer       the first token of that question's TOFU answer (CircuitKIT scores one
                       vocabulary entry at the answer position and takes the first token of
                       the answer string itself)
  * corrupted prompt   a TOFU retain99 question about another author, chosen to have (nearly)
                       the same token length so clean/corrupted positions line up, and whose
                       answer starts with a different token
  * corrupted answer   the first token of that retain answer (the "incorrect" label)
  * metric             GenericTaskSpec's logit-difference metric
                       (logit[clean first token] - logit[corrupted first token]), task_type
                       "classification"
  * algorithm          EAP-IG, node level (heads + MLPs), flat `circuitkit.discover` API

Caveats that belong in any write-up (also stored in the artifact notes):
  * The target is the FIRST answer token, which is often a generic opening ("The"), not the fact
    itself. Pairs whose first tokens coincide are dropped (CircuitKIT would drop them too).
  * Clean and corrupted differ in content, not by one minimal edit, so attribution is coarser
    than for a minimal-pair task such as IOI.
  * CircuitKIT's flat API returns node scores only. The circuit_graph edges drawn in Evalboard are
    derived (layer-adjacent) and labelled as such.
  * circuitkit's tiny-aya / cohere2 support is experimental.

Importing this module does not import torch or circuitkit. Pure helpers (pair selection, task
JSONL) are unit-tested with a fake tokenizer; discovery itself only runs on the Spark.
"""

from __future__ import annotations

import gc
import json
import random
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

TASK_NAME = "tofu_forget_fact"
CIRCUITKIT_DEFAULTS: dict[str, Any] = {
    "discover": False,
    "scores": {},
    "algorithm": "eap-ig",
    "level": "node",
    "n_examples": 32,
    "batch_size": 2,
    "ig_steps": 5,
    "sparsity": 0.3,
    "scope": "both",
    "dtype": "float32",
    "retain_pool": 1000,
    "max_len_diff": 2,
    "seed": 42,
    "top_fraction": 0.2,
    "chat_template_mode": "auto",
}
SCHEMA = {
    "prompt": "prompt",
    "answer": "answer",
    "corrupted_prompt": "corrupted_prompt",
    "corrupted_answer": "corrupted_answer",
    "id": "id",
}


def circuit_settings(settings: Mapping[str, Any] | None) -> dict[str, Any]:
    out = dict(CIRCUITKIT_DEFAULTS)
    out.update({k: v for k, v in (settings or {}).items() if v is not None})
    return out


def _ids(tokenizer, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def build_pair_rows(
    forget_pairs: Sequence[Mapping[str, str]],
    retain_pool: Sequence[Mapping[str, str]],
    tokenizer,
    *,
    seed: int = 42,
    max_len_diff: int = 2,
    max_rows: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pair each forget question with a length-matched retain question (without replacement).

    A retain candidate must (a) have a question within `max_len_diff` tokens of the forget
    question, (b) start its answer with a different first token. The closest length wins; ties
    break with a seeded RNG. Forget items with no candidate are skipped and counted.
    """
    rng = random.Random(seed)
    pool = []
    for r in retain_pool:
        q_ids, a_ids = _ids(tokenizer, r["question"]), _ids(tokenizer, r["answer"])
        if q_ids and a_ids:
            pool.append({"pair": r, "qlen": len(q_ids), "first": a_ids[0]})
    used: set[int] = set()
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for fp in forget_pairs:
        if max_rows is not None and len(rows) >= int(max_rows):
            break
        q_ids, a_ids = _ids(tokenizer, fp["question"]), _ids(tokenizer, fp["answer"])
        if not q_ids or not a_ids:
            skipped.append({"id": fp["id"], "reason": "empty question or answer after tokenization"})
            continue
        cands = [
            (abs(c["qlen"] - len(q_ids)), i)
            for i, c in enumerate(pool)
            if i not in used and c["first"] != a_ids[0] and abs(c["qlen"] - len(q_ids)) <= max_len_diff
        ]
        if not cands:
            skipped.append({"id": fp["id"], "reason": f"no retain question within {max_len_diff} tokens with a different first answer token"})
            continue
        best = min(d for d, _ in cands)
        choice = rng.choice(sorted(i for d, i in cands if d == best))
        used.add(choice)
        c = pool[choice]
        rows.append(
            {
                "id": f"forget-{fp['id']}",
                "prompt": fp["question"],
                "answer": fp["answer"],
                "corrupted_prompt": c["pair"]["question"],
                "corrupted_answer": c["pair"]["answer"],
                "forget_id": str(fp["id"]),
                "retain_id": str(c["pair"]["id"]),
                "clean_question_tokens": len(q_ids),
                "corrupted_question_tokens": c["qlen"],
                "clean_first_token_id": a_ids[0],
                "corrupted_first_token_id": c["first"],
            }
        )
    report = {
        "task": TASK_NAME,
        "n_forget": len(forget_pairs),
        "n_pairs": len(rows),
        "n_skipped": len(skipped),
        "skipped": skipped,
        "retain_pool": len(pool),
        "max_len_diff": max_len_diff,
        "seed": seed,
        "mean_abs_len_diff": (
            sum(abs(r["clean_question_tokens"] - r["corrupted_question_tokens"]) for r in rows) / len(rows) if rows else None
        ),
    }
    return rows, report


def write_task_jsonl(rows: Sequence[Mapping[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    return path


def register_task(jsonl_path: Path, *, chat_template_mode: str = "auto", name: str = TASK_NAME):
    """Build the GenericTaskSpec from the JSONL and register it (idempotent within a process)."""
    import circuitkit as ck  # import before transformer_lens (CircuitKIT docs)
    from circuitkit.tasks.generic import GenericTaskSpec

    spec = GenericTaskSpec.from_jsonl(
        str(jsonl_path),
        schema=dict(SCHEMA),
        name=name,
        task_type="classification",
        chat_template_mode=chat_template_mode,
    )
    try:
        ck.register_task(spec)
    except ValueError as exc:
        if "already registered" not in str(exc):
            raise
        from circuitkit.tasks.registry import _TASKS

        _TASKS[name.lower()] = spec  # same name, new data file for this run
    return spec


def resolve_checkpoint(src_dir: str | Path, scratch_dir: Path, *, dtype: str = "float32") -> Path:
    """A directory TransformerLens can load: the source itself, or a merged copy if it is a LoRA adapter."""
    src = Path(src_dir)
    if not (src / "adapter_config.json").exists():
        return src
    import torch
    from peft import AutoPeftModelForCausalLM  # type: ignore
    from transformers import AutoTokenizer

    dst = scratch_dir / f"merged-{src.name}"
    if not (dst / "config.json").exists():
        model = AutoPeftModelForCausalLM.from_pretrained(str(src), torch_dtype=getattr(torch, dtype))
        model = model.merge_and_unload()
        model.save_pretrained(str(dst))
        AutoTokenizer.from_pretrained(str(src)).save_pretrained(str(dst))
        del model
        gc.collect()
    return dst


def _free_gpu() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def find_scores_json(out_dir: Path, stem: str) -> Path | None:
    exact = out_dir / f"{stem}_scores.json"
    if exact.exists():
        return exact
    hits = sorted(out_dir.glob(f"{stem}*_scores.json"))
    return hits[0] if hits else None


def discover_one(model_dir: Path, label: str, out_dir: Path, st: Mapping[str, Any]) -> Path:
    """Load one checkpoint into TransformerLens and run EAP-IG on the registered task. Returns *_scores.json."""
    import circuitkit as ck  # before transformer_lens

    model = ck.load_model(str(model_dir), dtype=st["dtype"])
    try:
        extra: dict[str, Any] = {"seed": int(st["seed"])}
        if str(st["algorithm"]).startswith("eap-ig"):
            extra["ig_steps"] = int(st["ig_steps"])
        ck.discover(
            model,
            TASK_NAME,
            algorithm=st["algorithm"],
            level=st["level"],
            n_examples=int(st["n_examples"]),
            batch_size=int(st["batch_size"]),
            sparsity=float(st["sparsity"]),
            scope=st["scope"],
            output_path=str(out_dir / f"{label}.pt"),
            chat_template_mode=st["chat_template_mode"],
            **extra,
        )
    finally:
        del model
        _free_gpu()
    path = find_scores_json(out_dir, label)
    if path is None:
        raise FileNotFoundError(f"CircuitKIT finished but wrote no {label}_scores.json in {out_dir}")
    return path


def run_discovery(
    *,
    sources: Mapping[str, str | Path],
    forget_pairs: Sequence[Mapping[str, str]],
    retain_pool: Sequence[Mapping[str, str]],
    tokenizer,
    out_dir: Path,
    settings: Mapping[str, Any],
    discover_fn: Callable[[Path, str, Path, Mapping[str, Any]], Path] | None = None,
    register_fn: Callable[..., Any] | None = None,
    resolve_fn: Callable[..., Path] | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Build the task once, then discover a circuit per labelled checkpoint dir.

    Returns {"scores": {label: path}, "errors": [{label, error}], "task": report}. A failure for
    one label is recorded and the others still run; setup failures raise to the caller's guard.
    """
    # Resolved at call time (not as default arguments) so tests can patch the module-level functions.
    discover_fn = discover_fn or discover_one
    register_fn = register_fn or register_task
    resolve_fn = resolve_fn or resolve_checkpoint
    st = circuit_settings(settings)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, report = build_pair_rows(
        forget_pairs, retain_pool, tokenizer, seed=int(st["seed"]), max_len_diff=int(st["max_len_diff"]),
        max_rows=int(st["n_examples"]) if st.get("n_examples") else None,
    )
    if len(rows) < 2:
        raise RuntimeError(f"CircuitKIT task has {len(rows)} usable forget/retain pair(s); need at least 2. Report: {report}")
    jsonl = write_task_jsonl(rows, out_dir / f"{TASK_NAME}.jsonl")
    (out_dir / f"{TASK_NAME}_report.json").write_text(json.dumps(report, indent=2))
    log(f"circuitkit task: {len(rows)} pairs ({report['n_skipped']} skipped) -> {jsonl}")
    register_fn(jsonl, chat_template_mode=st["chat_template_mode"])
    scores: dict[str, str] = {}
    errors: list[dict[str, str]] = []
    for label, src in sources.items():
        try:
            ckpt = resolve_fn(src, out_dir / "checkpoints", dtype=st["dtype"])
            path = discover_fn(Path(ckpt), label, out_dir, st)
            scores[label] = str(path)
            log(f"circuitkit: {label} -> {path}")
        except Exception as exc:  # per-label guard: keep going
            errors.append({"label": label, "error": f"{type(exc).__name__}: {exc}"[:500]})
            log(f"circuitkit: {label} failed: {type(exc).__name__}: {exc}")
            _free_gpu()
    return {"scores": scores, "errors": errors, "task": report}


def task_note(report: Mapping[str, Any]) -> str:
    return (
        f"Task {TASK_NAME}: {report.get('n_pairs')} forget questions, each paired with a length-matched retain question "
        f"(mean |token length diff| {report.get('mean_abs_len_diff')}); target = first answer token, metric = logit difference "
        "(forget answer token minus retain answer token). Experimental Tiny Aya support in CircuitKIT."
    )
