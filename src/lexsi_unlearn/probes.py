"""Layer-wise linear probes for forget-set facts (RQ2).

Train probes on the fine-tuned (pre-unlearn) model, freeze them, and evaluate
on the unlearned model. High late-layer recoverability with low behavioral
accuracy is evidence of gating / hiding rather than erasure.

This uses standard Hugging Face `output_hidden_states=True` plus a linear
classifier (sklearn). It does not depend on an unverified CircuitKIT probe API.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

try:
    import numpy as np
except ImportError:  # pragma: no cover - no numpy on the Mac; the pure split helpers still import
    np = None  # type: ignore

from .tokpos import last_token_index, right_padded


def collect_layer_states(
    model,
    tokenizer,
    texts: Sequence[str],
    *,
    batch_size: int = 4,
    max_seq_len: int = 512,
) -> list[np.ndarray]:
    """Return one array per layer of last-prompt-token residuals, shape [N, D]."""
    import torch

    model.eval()
    # Discover layer count from a dry run on the first text.
    with torch.no_grad():
        probe = tokenizer(
            texts[0],
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_len,
        )
        probe = {k: v.to(model.device) for k, v in probe.items()}
        out = model(**probe, output_hidden_states=True, use_cache=False)
    n_layers = len(out.hidden_states)  # includes embedding layer at 0
    buckets = [[] for _ in range(n_layers)]

    for start in range(0, len(texts), batch_size):
        chunk = list(texts[start : start + batch_size])
        # Force right padding for the batch (position ids stay unshifted) and locate the last real token
        # by the mask itself, so the result is identical whatever padding_side the tokenizer ships with.
        with right_padded(tokenizer):
            encoded = tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_seq_len,
            )
        encoded = {k: v.to(model.device) for k, v in encoded.items()}
        with torch.no_grad():
            hidden = model(
                **encoded, output_hidden_states=True, use_cache=False
            ).hidden_states
        # Last non-pad token per row (padding-side agnostic).
        attn = encoded["attention_mask"]
        last_idx = last_token_index(attn)
        batch_ix = torch.arange(len(chunk), device=attn.device)
        for layer_i, tensor in enumerate(hidden):
            vec = tensor[batch_ix, last_idx].detach().float().cpu().numpy()
            buckets[layer_i].append(vec)

    return [np.concatenate(chunks, axis=0) for chunks in buckets]


def train_linear_probes(
    features_by_layer: Sequence[np.ndarray],
    labels: Sequence[int],
    idx: Sequence[int] | None = None,
):
    """Fit a logistic probe at each layer (on rows `idx` only when given). Returns a list of fitted classifiers."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    y = np.asarray(labels)
    if idx is not None:
        idx = np.asarray(idx, dtype=int)
        y = y[idx]
    probes = []
    for x in features_by_layer:
        if idx is not None:
            x = x[idx]
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        )
        clf.fit(x, y)
        probes.append(clf)
    return probes


def score_probes(
    probes,
    features_by_layer: Sequence[np.ndarray],
    labels: Sequence[int],
    idx: Sequence[int] | None = None,
) -> list[dict[str, float]]:
    """Accuracy, balanced accuracy and AUROC of frozen probes on rows `idx` (all rows if None).

    Pass the HELD-OUT indices as `idx`: scoring on the rows a probe was fit on only measures memorisation.
    AUROC is skipped (nan) if the evaluated labels are not binary.
    """
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

    y = np.asarray(labels)
    if idx is not None:
        idx = np.asarray(idx, dtype=int)
        y = y[idx]
    rows = []
    for layer_i, (clf, x) in enumerate(zip(probes, features_by_layer)):
        if idx is not None:
            x = x[idx]
        pred = clf.predict(x)
        row: dict[str, float] = {
            "layer": layer_i,
            "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "n": int(len(y)),
        }
        unique = np.unique(y)
        if len(unique) == 2 and hasattr(clf, "predict_proba"):
            try:
                proba = clf.predict_proba(x)
                # Positive class = the larger label, usually 1.
                pos = list(clf.classes_).index(unique.max()) if hasattr(clf, "classes_") else 1
                # pipelines store classes_ on the last step
                last = clf.steps[-1][1] if hasattr(clf, "steps") else clf
                pos = list(last.classes_).index(unique.max())
                row["auroc"] = float(roc_auc_score(y, proba[:, pos]))
            except Exception:
                row["auroc"] = float("nan")
        rows.append(row)
    return rows


# ---- held-out evaluation (stratified, seeded, repeated splits) ---------------------------------------


def stratified_split(
    labels: Sequence[int],
    *,
    test_frac: float = 0.3,
    seed: int = 0,
    groups: Sequence[Any] | None = None,
) -> tuple[list[int], list[int]]:
    """Seeded stratified train/test row indices (pure python).

    Every class contributes ~`test_frac` of its rows (>= 1, and always leaves >= 1 in train when the class has
    two or more units). With `groups`, rows sharing a group id are never split apart (a group is stratified by
    the label of its first row).
    """
    import random

    n = len(labels)
    gid = list(range(n)) if groups is None else list(groups)
    if len(gid) != n:
        raise ValueError("groups must have one entry per label")
    members: dict[Any, list[int]] = {}
    for i, g in enumerate(gid):
        members.setdefault(g, []).append(i)
    by_class: dict[Any, list[Any]] = {}
    for g, rows in members.items():
        by_class.setdefault(labels[rows[0]], []).append(g)
    rng = random.Random(seed)
    test_groups = set()
    for cls in sorted(by_class, key=str):
        units = sorted(by_class[cls], key=str)
        rng.shuffle(units)
        k = int(round(test_frac * len(units)))
        k = max(1, k) if len(units) >= 2 else 0
        k = min(k, len(units) - 1) if len(units) >= 2 else 0
        test_groups.update(units[:k])
    test = sorted(i for g in test_groups for i in members[g])
    train = sorted(set(range(n)) - set(test))
    return train, test


def make_splits(
    labels: Sequence[int],
    *,
    n_splits: int = 5,
    test_frac: float = 0.3,
    seed: int = 0,
    groups: Sequence[Any] | None = None,
) -> list[tuple[list[int], list[int]]]:
    return [stratified_split(labels, test_frac=test_frac, seed=seed + 1000 * k, groups=groups) for k in range(max(1, int(n_splits)))]


def aggregate_probe_rows(per_split_rows: Sequence[Sequence[dict[str, float]]]) -> list[dict[str, float]]:
    """Mean (and std) over repeated splits, per layer. nan AUROCs (single-class test folds) are ignored."""
    out: list[dict[str, float]] = []
    n_layers = max((len(r) for r in per_split_rows), default=0)
    for layer in range(n_layers):
        rows = [r[layer] for r in per_split_rows if len(r) > layer]
        agg: dict[str, float] = {"layer": layer, "n_splits": len(rows)}
        for key in ("accuracy", "balanced_accuracy", "auroc"):
            vals = [float(r[key]) for r in rows if r.get(key) is not None and r[key] == r[key]]
            if vals:
                arr = np.asarray(vals)
                agg[key] = float(arr.mean())
                agg[f"{key}_std"] = float(arr.std())
        agg["n"] = int(round(float(np.mean([r.get("n", 0) for r in rows])))) if rows else 0
        out.append(agg)
    return out


def fit_frozen_probes(train_features: Sequence[np.ndarray], labels: Sequence[int], splits) -> list[list[Any]]:
    """One list of per-layer probes per split, each fit ONLY on that split's train rows of `train_features`."""
    return [train_linear_probes(train_features, labels, tr) for tr, _ in splits]


def score_frozen_probes(fitted, eval_features: Sequence[np.ndarray], labels: Sequence[int], splits) -> list[dict[str, float]]:
    """Held-out score of frozen probes on another model's features (same prompts, same test rows)."""
    return aggregate_probe_rows([score_probes(probes, eval_features, labels, te) for probes, (_, te) in zip(fitted, splits)])


def refit_probe_curve(features: Sequence[np.ndarray], labels: Sequence[int], splits) -> list[dict[str, float]]:
    """Fresh probes fit on a model's OWN train rows and scored on its held-out rows (is the info decodable at all?)."""
    return aggregate_probe_rows([score_probes(train_linear_probes(features, labels, tr), features, labels, te) for tr, te in splits])


def shuffled_label_control(features: Sequence[np.ndarray], labels: Sequence[int], splits, *, seed: int = 0) -> list[dict[str, float]]:
    """Chance control: probes fit on permuted train labels, scored on the true held-out labels."""
    rng = np.random.default_rng(seed)
    y = np.asarray(labels)
    per_split = []
    for tr, te in splits:
        perm = y.copy()
        perm[tr] = rng.permutation(y[tr])
        per_split.append(score_probes(train_linear_probes(features, perm, tr), features, labels, te))
    return aggregate_probe_rows(per_split)


def build_content_probe_texts(
    pairs: Sequence[dict[str, str]],
    tokenizer,
    *,
    seed: int = 0,
) -> tuple[list[str], list[int], list[int]]:
    """Fact-content probe data: (question + TRUE answer) = 1 vs (question + another fact's answer) = 0.

    Returns (texts, labels, groups); both texts of a question share a group so a split never separates them.
    The residual at the final token of the answer-bearing text should encode whether the stated fact is right.
    Limitation: distractor answers are other forget-set answers, so the probe may partly pick up question-answer
    topical mismatch rather than the stored fact itself.
    """
    import random

    rng = random.Random(seed)
    n = len(pairs)
    texts: list[str] = []
    labels: list[int] = []
    groups: list[int] = []
    if n < 2:
        return texts, labels, groups
    for i, p in enumerate(pairs):
        j = rng.choice([k for k in range(n) if k != i and pairs[k]["answer"].strip().lower() != p["answer"].strip().lower()] or [(i + 1) % n])
        for label, ans in ((1, p["answer"]), (0, pairs[j]["answer"])):
            texts.append(_qa_text(tokenizer, p["question"], ans))
            labels.append(label)
            groups.append(i)
    return texts, labels, groups


def _qa_text(tokenizer, question: str, answer: str) -> str:
    messages = [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return f"{question}\n{answer}"


def author_identity_labels(pairs: Iterable[dict[str, str]]) -> tuple[list[str], list[int]]:
    """Build a simple author-id label from an optional `author` field, or skip.

    TODO: official TOFU forget01/retain99 rows only have `question` and
    `answer` (verified 2026-11-12 via load_dataset on the Spark). There is
    no author column, so this helper stays unused unless we parse author
    names out of the question/answer text and document that heuristic.
    https://huggingface.co/datasets/locuslab/TOFU
    """
    texts: list[str] = []
    labels: list[int] = []
    vocab: dict[str, int] = {}
    for p in pairs:
        author = p.get("author")
        if not author:
            continue
        if author not in vocab:
            vocab[author] = len(vocab)
        texts.append(p["question"])
        labels.append(vocab[author])
    return texts, labels
