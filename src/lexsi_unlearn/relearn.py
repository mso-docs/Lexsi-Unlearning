"""Short relearning vs. matched new fictional facts (RQ3).

Compare (a) a few-shot or short fine-tune recovery of forget-set facts after
unlearning with (b) learning a matched *new* fictional author for the same
number of steps. Faster recovery of old facts than acquisition of new ones
is evidence that the information was suppressed, not gone.

Related (verified): Deeb and Roger, arXiv:2410.08827.
The new fictional author set should be constructed and documented during the
hackathon — do not copy copyrighted biographies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence


def few_shot_prompt(demos: Sequence[dict[str, str]], question: str) -> str:
    parts = []
    for d in demos:
        parts.append(f"Q: {d['question']}\nA: {d['answer']}")
    parts.append(f"Q: {question}\nA:")
    return "\n\n".join(parts)


DEFAULT_STEP_GRID = (0, 5, 10, 20, 30)


def relearn_step_grid(steps: int, grid: Sequence[int] | None = None) -> list[int]:
    """Evaluation steps for the per-step curve: the grid clipped to [0, steps], always ending at `steps`."""
    base = DEFAULT_STEP_GRID if not grid else grid
    out = sorted({int(g) for g in base if 0 <= int(g) <= int(steps)} | {0, int(steps)})
    return out


def short_finetune(
    model,
    tokenizer,
    pairs: Sequence[dict[str, str]],
    *,
    steps: int = 30,
    learning_rate: float = 2e-4,
    max_seq_len: int = 512,
    output_dir: str | Path = "artifacts/relearn",
    eval_steps: Sequence[int] | None = None,
    eval_fn=None,
):
    """A short LoRA (or continued) fine-tune used only as a relearning probe.

    Reuses inject.build_sft_dataset / Trainer so the recipe stays consistent.

    With `eval_steps` and `eval_fn(step, model)`, the model is scored mid-training (step 0 = before any update)
    inside ONE training run, so the optimizer / LR schedule is not restarted between points.
    """
    from datasets import Dataset  # type: ignore
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

    from .inject import build_sft_dataset

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    raw = Dataset.from_list(
        [{"question": p["question"], "answer": p["answer"]} for p in pairs]
    )
    tokenized = build_sft_dataset(tokenizer, raw, max_seq_len=max_seq_len)
    args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=steps,
        per_device_train_batch_size=1,
        learning_rate=learning_rate,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        bf16=_bf16_ok(),
        remove_unused_columns=False,
    )
    callbacks = []
    wanted = sorted({int(x) for x in (eval_steps or [])})

    def _score(step: int) -> None:
        was_training = bool(getattr(model, "training", False))
        model.eval()
        try:
            eval_fn(step, model)
        finally:
            if was_training:
                model.train()

    if eval_fn is not None and wanted:
        from transformers import TrainerCallback

        class _EvalAtSteps(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001
                if state.global_step in wanted and state.global_step > 0:
                    _score(int(state.global_step))

        callbacks.append(_EvalAtSteps())
        if 0 in wanted:
            _score(0)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
        callbacks=callbacks or None,
    )
    trainer.train()
    return model


def _bf16_ok() -> bool:
    """bf16 training only where it is supported (the Spark GPU); CPU unit tests fall back to fp32."""
    try:
        import torch

        return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    except Exception:  # pragma: no cover
        return False


def curve_point(result: Mapping[str, Any], step: int) -> dict[str, Any]:
    """One curve point from an evaluate_model_on_items result (+ judge verdicts if present).

    judge_match is the fraction of JUDGED items the judge says match the gold fact; failed judgments are
    missing (excluded and counted in judge_n_missing), never zero.
    """
    items = result.get("items") or []
    verdicts = []
    for it in items:
        m = (((it.get("judge") or {}).get("primary")) or {}).get("match") if isinstance(it, dict) else None
        verdicts.append(m)
    judged = [bool(v) for v in verdicts if v is not None]
    return {
        "step": int(step),
        "n": int(result.get("n", len(items))),
        "substring_match": result.get("substring_match"),
        "judge_match": (sum(judged) / len(judged)) if judged else None,
        "judge_n": len(judged),
        "judge_n_missing": (len(verdicts) - len(judged)) if any(isinstance(it, dict) and it.get("judge") for it in items) else None,
    }


def curves_to_series(curves: Mapping[str, Sequence[Mapping[str, Any]]], key: str, steps: Sequence[int]) -> dict[str, list[float | None]]:
    """{arm label: [value at each step]} with None where a step/metric is missing."""
    out: dict[str, list[float | None]] = {}
    for label, points in curves.items():
        by_step = {int(p["step"]): p.get(key) for p in points}
        out[label] = [by_step.get(int(s)) for s in steps]
    return out


def compare_relearning_curves(
    recovered_scores: Sequence[dict[str, Any]],
    new_fact_scores: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize two step-indexed accuracy curves (old facts vs new facts)."""
    def _final(rows):
        if not rows:
            return None
        return rows[-1]

    return {
        "forget_relearn_final": _final(recovered_scores),
        "new_facts_final": _final(new_fact_scores),
        "n_steps_recorded_forget": len(list(recovered_scores)),
        "n_steps_recorded_new": len(list(new_fact_scores)),
        "note": (
            "Faster recovery of forget-set facts than acquisition of a matched "
            "new fictional set is evidence of suppression, not erasure."
        ),
    }
