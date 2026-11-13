"""Held-out probe suite used by the `probes` and `relearn` stages (no model code here: features in, curves out).

Three curves per probe task, all scored on rows the probe never trained on (seeded stratified splits, repeated):
  * frozen   - probes fit on the injected model's train rows, applied to each model's held-out rows
               (pre_inject / post_inject / post_rmu / post_relearn). Low post_rmu = the *frozen* readout is gone.
  * refit    - fresh probes fit on each model's OWN train rows, scored on its held-out rows. High post_rmu =
               the information is still linearly decodable even though behaviour is suppressed. The pre_inject
               (base model, never saw the facts) refit curve is the control: forget-vs-retain topics may be
               separable without any injected knowledge, so only the gap over base is evidence about the facts.
  * shuffled - injected features, probes fit on permuted labels: should sit at chance (sanity check).
Tasks: "forget_vs_retain" (question topic) and "content" (question + true answer vs + another fact's answer).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from . import probes as P


def new_state(
    feats_by_phase: Mapping[str, Sequence[Any]],
    labels: Sequence[int],
    *,
    train_phase: str = "post_inject",
    n_splits: int = 5,
    test_frac: float = 0.3,
    seed: int = 0,
    groups: Sequence[Any] | None = None,
) -> dict[str, Any]:
    if train_phase not in feats_by_phase:
        raise ValueError(f"train_phase {train_phase!r} has no features")
    splits = P.make_splits(labels, n_splits=n_splits, test_frac=test_frac, seed=seed, groups=groups)
    state: dict[str, Any] = {
        "labels": list(labels),
        "splits": splits,
        "train_phase": train_phase,
        "seed": seed,
        "fitted": P.fit_frozen_probes(feats_by_phase[train_phase], labels, splits),
        "frozen": {},
        "refit": {},
    }
    for phase, feats in feats_by_phase.items():
        add_phase(state, phase, feats)
    state["refit"]["shuffled_label_control"] = P.shuffled_label_control(feats_by_phase[train_phase], labels, splits, seed=seed)
    n_te = [len(te) for _, te in splits]
    state["split_summary"] = {
        "n_total": len(labels), "n_splits": len(splits), "test_frac": test_frac, "seed": seed,
        "n_train_per_split": [len(tr) for tr, _ in splits], "n_test_per_split": n_te,
        "n_positive": int(sum(1 for y in labels if y == 1)),
    }
    return state


def add_phase(state: dict[str, Any], phase: str, feats: Sequence[Any]) -> None:
    """Add one more model's curves (e.g. post_relearn) against the SAME splits and frozen probes."""
    state["frozen"][phase] = P.score_frozen_probes(state["fitted"], feats, state["labels"], state["splits"])
    state["refit"][phase] = P.refit_probe_curve(feats, state["labels"], state["splits"])


def summarize(state: Mapping[str, Any]) -> dict[str, Any]:
    """JSON-ready summary: per-phase frozen / refit rows (held-out) plus the split description."""
    def clean(rows):
        return [{k: (None if isinstance(v, float) and v != v else v) for k, v in r.items()} for r in rows]

    return {
        "heldout": True,
        "split": state.get("split_summary"),
        "frozen": {k: clean(v) for k, v in state["frozen"].items()},
        "refit": {k: clean(v) for k, v in state["refit"].items()},
    }
