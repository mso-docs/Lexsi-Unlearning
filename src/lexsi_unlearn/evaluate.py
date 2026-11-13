"""AuditKIT before/after scoring of forget-set and retain-set answers.

Verified docs: https://auditkit.lexsi.ai/
Confirmed APIs: auditkit.evaluate, auditkit.Sample, exact match / F1,
and compare_models() for scoring several checkpoints on the same samples.

Wrap TOFU-style QA (and translations) as Sample objects rather than expecting
a built-in TOFU runner.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Sequence

_WORD = re.compile(r"\w+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall((text or "").lower())


def token_f1(prediction: str, target: str) -> float:
    """Bag-of-words F1 between a generation and the gold answer (order-insensitive, forgiving of extra words)."""
    from collections import Counter

    pred, gold = _tokens(prediction), _tokens(target)
    common = sum((Counter(pred) & Counter(gold)).values())
    if not common:
        return 0.0
    precision, recall = common / len(pred), common / len(gold)
    return 2 * precision * recall / (precision + recall)


def rouge_l_recall(prediction: str, target: str) -> float:
    """ROUGE-L recall: longest common subsequence of word tokens / gold length (the TOFU paper's recall metric).

    1.0 when the generation contains the gold answer as an ordered word subsequence, so a model that has memorized
    an answer scores ~1.0 even when it adds a prefix/suffix or changes the punctuation (which the strict
    whole-answer substring test treats as a miss).
    """
    pred, gold = _tokens(prediction), _tokens(target)
    if not gold:
        return 0.0
    prev = [0] * (len(gold) + 1)
    for tok in pred:
        cur = [0] * (len(gold) + 1)
        for j, g in enumerate(gold, 1):
            cur[j] = prev[j - 1] + 1 if tok == g else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1] / len(gold)


def to_samples(items: Sequence[dict[str, str]]):
    """Convert {prompt, target} dicts into auditkit.Sample objects."""
    import auditkit as ak  # type: ignore

    samples = []
    for item in items:
        # Verified constructor used in official examples:
        #   Sample(input=..., target=...)
        # https://auditkit.lexsi.ai/
        sample_kw = {"input": item["prompt"], "target": item["target"]}
        if item.get("id") is not None:
            sample_kw["id"] = str(item["id"])
        samples.append(ak.Sample(**sample_kw))
    return samples


def generate_answers(
    model,
    tokenizer,
    prompts: Sequence[str],
    *,
    max_new_tokens: int = 64,
    max_seq_len: int = 512,
) -> list[str]:
    import torch

    from .data import apply_chat_template

    model.eval()
    answers = []
    for prompt in prompts:
        text = apply_chat_template(tokenizer, prompt, add_generation_prompt=True)
        encoded = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_seq_len
        )
        encoded = {k: v.to(model.device) for k, v in encoded.items()}
        with torch.no_grad():
            out = model.generate(
                **encoded, max_new_tokens=max_new_tokens, do_sample=False
            )
        new_tokens = out[0, encoded["input_ids"].shape[1] :]
        answers.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return answers


def evaluate_items(
    items: Sequence[dict[str, str]],
    predictions: Sequence[str],
) -> Any:
    """Score precomputed predictions with AuditKIT.

    The documented `evaluate` signature is
        auditkit.evaluate(samples, model=callable)
    where the callable maps prompts to answers. See https://auditkit.lexsi.ai/

    We pass a callable that returns the already-generated strings in order so
    we can reuse generations across metrics without a second decode.
    """
    import auditkit as ak  # type: ignore

    samples = to_samples(items)
    queue = list(predictions)

    def model_fn(prompts):
        # auditkit may call this with a batch of Sample.input strings.
        # Consume the precomputed queue in order; fall back to a length check.
        if len(queue) < len(prompts):
            raise RuntimeError(
                "Not enough precomputed predictions for this AuditKIT batch. "
                "Regenerate answers so the order matches the samples."
            )
        batch = [queue.pop(0) for _ in prompts]
        return batch

    # cache=False: default disk cache would replay a previous callable's
    # outputs when weights have been edited in place (inject / RMU).
    # Verified in auditkit 1.1.11 evaluate() docs.
    result = ak.evaluate(samples, model=model_fn, cache=False)
    return result


def evaluate_model_on_items(
    model,
    tokenizer,
    items: Sequence[dict[str, str]],
    *,
    max_new_tokens: int = 64,
) -> dict[str, Any]:
    prompts = [item["prompt"] for item in items]
    preds = generate_answers(
        model, tokenizer, prompts, max_new_tokens=max_new_tokens
    )
    result = evaluate_items(items, preds)
    headline = getattr(result, "headline", None)
    # Exact-match fallback if we need a simple number in addition to AuditKIT.
    per_item = []
    substring_hits = 0
    exact_hits = 0
    f1_sum = 0.0
    rouge_sum = 0.0
    for pred, item in zip(preds, items):
        target = item["target"]
        pred_norm = pred.strip().lower()
        target_norm = target.strip().lower()
        sub = 1.0 if target_norm in pred_norm else 0.0
        exact = 1.0 if pred_norm == target_norm else 0.0
        f1 = token_f1(pred, target)
        rouge = rouge_l_recall(pred, target)
        substring_hits += int(sub)
        exact_hits += int(exact)
        f1_sum += f1
        rouge_sum += rouge
        per_item.append(
            {
                "id": item.get("id"),
                "format": item.get("format"),
                "prompt": item["prompt"],
                "target": target,
                "prediction": pred,
                "substring_match": sub,
                "exact_match": exact,
                "token_f1": f1,
                "rouge_l_recall": rouge,
                "language": item.get("language"),
            }
        )
    return {
        "n": len(items),
        "substring_match": substring_hits / max(len(items), 1),
        "exact_match": exact_hits / max(len(items), 1),
        "mean_token_f1": f1_sum / max(len(items), 1),
        "mean_rouge_l_recall": rouge_sum / max(len(items), 1),
        "auditkit_headline": headline if headline is not None else str(result),
        "auditkit_result": result,
        "predictions": preds,
        "items": per_item,
    }


def compare_checkpoints(
    named_scores: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Collect before/after deltas. Lexsi requires more than one metric.

    TODO: if you want AuditKIT's own compare_models() on raw Sample objects
    instead of this wrapper, confirm the signature in the installed package
    and the docs at https://auditkit.lexsi.ai/ before switching.
    """
    return named_scores


def summarize_forget_retain(
    *,
    forget: dict[str, Any],
    retain: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    return {
        "label": label,
        "forget_substring_match": forget.get("substring_match"),
        "retain_substring_match": retain.get("substring_match"),
        "forget_rouge_l_recall": forget.get("mean_rouge_l_recall"),
        "retain_rouge_l_recall": retain.get("mean_rouge_l_recall"),
        "forget_token_f1": forget.get("mean_token_f1"),
        "retain_token_f1": retain.get("mean_token_f1"),
        "forget_auditkit": forget.get("auditkit_headline"),
        "retain_auditkit": retain.get("auditkit_headline"),
        "forget_n": forget.get("n"),
        "retain_n": retain.get("n"),
    }
