"""Orchestrates interpretability artifacts for one pre/post pair of models (feeds the Evalboard interp tab).

Defaults live here (not in the YAML) and are overridden by an optional `interp:` block in the config:

    interp:
      enabled: true
      lens: {max_prompts: 4}
      patching: {max_prompts: 3, layer_stride: 3, max_positions: 24, min_gap: 1.0}
      features: {layers: null, per_layer: 2, top_examples: 4, n_forget_texts: 16, n_retain_texts: 16}
      circuitkit: {scores: {}, discover: false, ...}   # scores: label -> path of a CircuitKIT *_scores.json;
                                                        # discover: run CircuitKIT on the TOFU forget-fact task (circuit_task.py)
      extras:                                           # token x layer maps, lens tables, head ablation grids,
        enabled: true                                   # activation-difference histograms, PCA scatter (interp_extras.py)
        token_maps: {enabled: true, max_forget: 2, max_retain: 1, max_tokens: 64}
        lens_table: {enabled: true, max_prompts: 2, top_k: 5, layer_stride: 2}
        head_ablation: {enabled: true, max_prompts: 2, layer_stride: 2, head_batch: 16}
        distributions: {enabled: true, max_prompts: 16, bins: 30}
        projection: {enabled: true, max_prompts: 32, layers: null}   # null = the RMU layer

Every component runs inside its own try/except: a failure is recorded in the returned `errors`
list and never aborts the pipeline or produces made-up artifacts.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import interp as I
from . import interp_artifacts as A
from . import interp_extras as X
from . import circuit_task as CT
from .data import apply_chat_template
from .traces import case_id_for, output_dir

DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "lens": {"max_prompts": 4},
    "patching": {"max_prompts": 3, "layer_stride": 3, "max_positions": 24, "min_gap": 1.0},
    "features": {"layers": None, "per_layer": 2, "top_examples": 4, "n_forget_texts": 16, "n_retain_texts": 16},
    "circuitkit": dict(CT.CIRCUITKIT_DEFAULTS),
    # Light defaults (also what the smoke run uses). configs/spark.yaml turns the heavier settings on.
    "extras": {
        "enabled": True,
        "token_maps": {"enabled": True, "max_forget": 2, "max_retain": 1, "max_tokens": 64},
        "lens_table": {"enabled": True, "max_prompts": 2, "top_k": 5, "layer_stride": 2},
        "head_ablation": {"enabled": True, "max_prompts": 2, "layer_stride": 2, "head_batch": 16},
        "distributions": {"enabled": True, "max_prompts": 16, "bins": 30},
        "projection": {"enabled": True, "max_prompts": 32, "layers": None},
    },
}


def _merge(base: dict, extra: Mapping[str, Any]) -> None:
    for key, value in extra.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)


def interp_settings(cfg: Mapping[str, Any], override: Mapping[str, Any] | None = None) -> dict[str, Any]:
    out = copy.deepcopy(DEFAULTS)
    for source in (cfg.get("interp") or {}, override or {}):
        _merge(out, source)
    if os.environ.get("LEXSI_INTERP", "").strip() in ("0", "false", "no"):
        out["enabled"] = False
    if os.environ.get("LEXSI_INTERP_EXTRAS", "").strip() in ("0", "false", "no"):
        out["extras"]["enabled"] = False
    return out


def _full_text(tokenizer, question: str, answer: str) -> str:
    messages = [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return f"{question}\n{answer}"


def default_feature_layers(cfg: Mapping[str, Any], n_layers: int) -> list[int]:
    rmu = cfg.get("rmu") or {}
    layer = int(rmu.get("layer_id", 7))
    update = rmu.get("update_layer_ids") or rmu.get("update_layers") or [layer]
    picks = {int(l) for l in update} | {max(0, min(update) - 4 if update else 0), min(n_layers - 1, layer + 6)}
    return sorted(l for l in picks if 0 <= l < n_layers)


def circuit_scores_artifact(path: str | Path, *, base_run_id: str, name: str, title: str, phase: str | None = None, top_fraction: float = 0.2, extra_notes: str = "", **kw: Any) -> dict[str, Any]:
    """circuit_graph artifact from a CircuitKIT `*_scores.json` (CircuitScores.to_json, circuitkit 0.2.3).

    `node_scores` -> nodes. `safety_units.unit_ids` (top fraction of nodes by score, written by
    CircuitKIT) marks the nodes drawn as "in circuit"; edges are derived layer-adjacent edges and are
    labelled as such (CircuitKIT's flat API does not return measured edges).
    """
    blob = json.loads(Path(path).read_text())
    scores = blob["node_scores"]
    unit_ids = set(((blob.get("safety_units") or {}).get("unit_ids")) or [])
    if unit_ids:
        basis = "'In circuit' = CircuitKIT's top-fraction safety_units. "
    else:
        # circuitkit 0.2.3's safety_units regexes do not match EAP node names (a3.h1 / m3), so the list is empty there.
        ranked = sorted(scores, key=lambda n: -abs(float(scores[n])))
        unit_ids = set(ranked[: max(1, round(len(ranked) * top_fraction))])
        basis = f"safety_units absent/empty in the file; 'in circuit' = our display choice: top {top_fraction:.0%} of nodes by score. "
    pruned = [n for n in scores if n not in unit_ids]
    graph = A.circuit_graph_from_scores(scores, pruned=pruned)
    notes = "Node scores from CircuitKIT (" + str(blob.get("algorithm")) + ", task " + str(blob.get("task")) + "). " + basis + "Edges are derived (layer-adjacent), not measured." + (" " + extra_notes if extra_notes else "")
    return A.circuit_graph_artifact(
        base_run_id=base_run_id, name=name, title=title, graph=graph, algorithm=blob.get("algorithm"), task=blob.get("task"),
        model=kw.pop("model", None) or blob.get("model"), phase=phase, notes=notes,
        provenance=kw.pop("provenance", "mixed"),
        provenance_note=kw.pop("provenance_note", "Node scores are measured by CircuitKIT; the 'in circuit' set is a display threshold and the edges are derived (layer-adjacent), not measured."),
        **kw,
    )


def circuit_layer_profile_artifact(paths_by_label: Mapping[str, str | Path], *, base_run_id: str, model: str | None = None, extra_notes: str = "") -> dict[str, Any] | None:
    """layer_series: summed |CircuitKIT node score| per layer (heads + MLPs), one series per checkpoint label.

    A compact before/after view of where the forget-fact task is attributed. None if no file has parseable nodes.
    """
    totals: dict[str, dict[int, float]] = {}
    for label, path in paths_by_label.items():
        blob = json.loads(Path(path).read_text())
        per_layer: dict[int, float] = {}
        for name, score in (blob.get("node_scores") or {}).items():
            info = A.parse_circuit_node(name)
            val = A.clean_number(score)
            if info is None or val is None:
                continue
            per_layer[info["layer"]] = per_layer.get(info["layer"], 0.0) + abs(val)
        if per_layer:
            totals[label] = per_layer
    if not totals:
        return None
    n_layers = max(max(v) for v in totals.values()) + 1
    series = {label: [v.get(i) for i in range(n_layers)] for label, v in totals.items()}
    return A.layer_series_artifact(
        base_run_id=base_run_id, name="circuit-layer-profile", title="CircuitKIT node importance per layer (sum of |score|)",
        metric="sum |node score| per layer", layers=list(range(n_layers)), series=series, model=model,
        provenance="derived", provenance_note="Sum of |measured CircuitKIT node scores| per layer (post-processing).",
        notes="Heads and MLPs summed per layer from CircuitKIT EAP-IG node scores on the TOFU forget-fact task. "
              "Scores are only comparable within one task and algorithm run." + (" " + extra_notes if extra_notes else ""),
    )


def probe_artifacts(
    base_run_id: str,
    phase_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    counts: Mapping[str, int] | None = None,
    model: str | None = None,
    probe_note: str = "Logistic probes trained on the injected model's last-prompt-token residuals (forget vs retain), frozen, then applied to each model. "
    "Scored on HELD-OUT prompts only (seeded stratified splits, mean over splits; band = +/- 1 std).",
    prefix: str = "probe",
    subject: str = "forget vs retain, held-out",
) -> list[dict[str, Any]]:
    """layer_series artifacts (held-out accuracy, AUROC) from {phase label: probe rows}; a `<metric>_std` in the rows draws a band."""
    arts = []
    for metric, key, title in (("auroc", "AUROC", f"Probe AUROC by layer ({subject})"), ("accuracy", "accuracy", f"Probe accuracy by layer ({subject})")):
        series = {phase: I.probe_series(rows, metric) for phase, rows in phase_rows.items()}
        n_layers = max((len(v) for v in series.values()), default=0)
        series = {k: v + [None] * (n_layers - len(v)) for k, v in series.items()}
        if not any(x is not None for v in series.values() for x in v):
            continue
        bands = {}
        for phase, rows in phase_rows.items():
            stds = I.probe_series(rows, f"{metric}_std")
            if any(x is not None for x in stds):
                stds = stds + [None] * (n_layers - len(stds))
                means = series[phase]
                bands[phase] = (
                    [None if m is None or sd is None else max(0.0, m - sd) for m, sd in zip(means, stds)],
                    [None if m is None or sd is None else min(1.0, m + sd) for m, sd in zip(means, stds)],
                )
        arts.append(
            A.layer_series_artifact(
                base_run_id=base_run_id, name=f"{prefix}-{metric}", title=title, metric=key, layers=list(range(n_layers)),
                series=series, counts=counts, value_range=[0, 1], bands=bands or None, model=model, provenance="measured",
                reference_lines=[(0.5, "chance")] if metric == "auroc" else (),
                notes=probe_note + " Layer 0 = embeddings. Missing phases are not shown.",
            )
        )
    return arts


RELEARN_ARMS = {
    "rmu_old": "RMU model + forgotten facts",
    "base_old": "Base (never injected) + same facts",
    "rmu_new": "RMU model + new fictional facts",
}


def relearn_curve_artifacts(
    base_run_id: str,
    arm_curves: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    steps: Sequence[int],
    *,
    model: str | None = None,
    counts: Mapping[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Per-step relearning curves as layer_series artifacts (x axis = fine-tuning step, stored in `layers`).

    arm_curves: {arm id: {"train": [points], "paraphrase": [points]}} with points from relearn.curve_point.
    Held-out = paraphrased questions of the SAME facts (never trained on). Faster rise for the RMU model than for
    the never-injected base on the same facts means the knowledge was suppressed, not removed.
    """
    from . import relearn as R

    arts = []
    for qset, name, title, note in (
        ("paraphrase", "relearn-curve-heldout", "Relearning curve, held-out paraphrased questions", "Scored on paraphrased questions of the facts; the fine-tune only saw the direct questions."),
        ("train", "relearn-curve-train", "Relearning curve, trained questions", "Scored on the direct questions the fine-tune trains on (not held out)."),
    ):
        for key, suffix, metric in (("substring_match", "", "substring accuracy"), ("judge_match", "-judge", "judge-match accuracy")):
            curves = {RELEARN_ARMS.get(arm, arm): sets[qset] for arm, sets in arm_curves.items() if sets.get(qset)}
            series = R.curves_to_series(curves, key, steps)
            if not any(v is not None for row in series.values() for v in row):
                continue
            arts.append(
                A.layer_series_artifact(
                    base_run_id=base_run_id, name=name + suffix, title=f"{title} ({metric})", metric=metric,
                    layers=[int(x) for x in steps], series=series, counts=counts, value_range=[0, 1], model=model, provenance="measured",
                    notes=note + " X axis is the fine-tuning step (stored as `layers`); step 0 = before any update. "
                    + ("Judge verdicts that failed are missing, not zero." if key == "judge_match" else "Substring match is a strict lowercase containment check."),
                )
            )
    return arts


# ---- extra mech-interp artifacts (token maps, lens tables, head ablation, distributions, PCA) ----------


def rmu_layer_index(cfg: Mapping[str, Any], n_layers: int) -> int:
    """Decoder layer RMU steers (rmu.layer_id), clipped to the model."""
    return max(0, min(n_layers - 1, int((cfg.get("rmu") or {}).get("layer_id", 7))))


def token_map_artifacts(*, base_run_id, tokenizer, model_pre, model_post, forget_pairs, retain_pairs, forget_prompts, retain_prompts,
                        pre_label, post_label, model_id, case_of, retain_case_of, links, cfg_ex, suffix=""):
    kf, kr, mt = int(cfg_ex["max_forget"]), int(cfg_ex["max_retain"]), int(cfg_ex["max_tokens"])
    items = [("forget", p, q, case_of(p)) for p, q in zip(forget_pairs[:kf], forget_prompts[:kf])]
    items += [("retain", p, q, retain_case_of(p)) for p, q in zip(retain_pairs[:kr], retain_prompts[:kr])]
    if not items:
        raise ValueError("no prompts for token maps")
    examples, n_layers_res = [], None
    for split, pair, prompt, cid in items:
        a, b = (X.token_norm_grid(m, tokenizer, prompt, max_tokens=mt) for m in (model_pre, model_post))
        n_layers_res = len(a["grid"])
        examples.append({"id": f"{split}-{pair['id']}", "label": f"{split} case {pair['id']}: {pair['question'][:100]}", "case_id": cid,
                         "tokens": a["tokens"], "panels": {pre_label: a["grid"], post_label: b["grid"]},
                         **({"trace_id": links([cid])[-1]} if links([cid]) else {})})
    return [A.token_layer_map_artifact(
        base_run_id=base_run_id, name="token-layer-norms", title=f"Per-token residual-stream norm across layers ({pre_label} vs {post_label})",
        layers=list(range(n_layers_res)), examples=examples, value_label="residual L2 norm", model=model_id,
        case_ids=[e["case_id"] for e in examples], provenance="measured",
        provenance_note="||residual[layer, position]||2 read from a forward pass of each checkpoint; no post-processing.",
        notes=f"Layer 0 = embeddings; layer i+1 = output of decoder layer i. Last {mt} prompt tokens of chat-formatted questions (forget and retain)." + suffix,
    )]


def lens_table_artifacts(*, base_run_id, tokenizer, model_pre, model_post, forget_pairs, forget_prompts, pre_label, post_label, model_id,
                         case_of, links, cfg_ex, suffix=""):
    arts = []
    for i, pair in enumerate(forget_pairs[: int(cfg_ex["max_prompts"])]):
        kw = dict(top_k=int(cfg_ex["top_k"]), layer_stride=int(cfg_ex["layer_stride"]))
        a = X.lens_top_rows(model_pre, tokenizer, forget_prompts[i], pair["answer"], **kw)
        b = X.lens_top_rows(model_post, tokenizer, forget_prompts[i], pair["answer"], **kw)
        cid = case_of(pair)
        arts.append(A.logit_lens_table_artifact(
            base_run_id=base_run_id, name=f"logit-lens-table-{i}", title=f"Logit lens · top tokens per layer · forget case {pair['id']}",
            panels={pre_label: a["rows"], post_label: b["rows"]}, value_label="probability", position_label="final prompt position (generation prompt)",
            target_token=a["target_token"], model=model_id, case_ids=[cid], trace_ids=links([cid]), provenance="measured",
            provenance_note="softmax(lm_head(final_norm(residual)) * logit_scale) over the full vocabulary at the last prompt position; top-k and target rank are read from it.",
            notes=f"Residual index 0 = embeddings, i+1 = decoder layer i; every {kw['layer_stride']}th index. Target = first token of the reference answer. Q: " + pair["question"][:160] + suffix,
        ))
    if not arts:
        raise ValueError("no forget pairs for lens tables")
    return arts


def head_ablation_artifact(*, base_run_id, tokenizer, model_pre, model_post, forget_pairs, forget_prompts, pre_label, post_label, model_id,
                           case_of, cfg_ex, suffix=""):
    n_layers = len(I.decoder_layers(model_post))
    layers = X.strided(n_layers, int(cfg_ex["layer_stride"]))
    n = min(int(cfg_ex["max_prompts"]), len(forget_pairs))
    if n < 1:
        raise ValueError("no forget pairs for head ablation")
    panels, n_heads = {}, None
    for label, model in ((pre_label, model_pre), (post_label, model_post)):
        grids = []
        for i in range(n):
            out = X.head_ablation_grid(model, tokenizer, forget_prompts[i], forget_pairs[i]["answer"], layers=layers, head_batch=int(cfg_ex["head_batch"]))
            grids.append(out["grid"])
            n_heads = out["n_heads"]
        mean, _ = zip(*[X.mean_std([g[r] for g in grids]) for r in range(len(layers))])
        panels[label] = [list(m) for m in mean]
        X.free_gpu_cache()
    # the grid rows are the strided layers only: record them as the layer axis (the UI shows layer labels as given)
    return A.head_grid_artifact(
        base_run_id=base_run_id, name="head-ablation", title="Attention-head importance (answer log-prob drop under zero-ablation of one head)",
        layers=layers, n_heads=n_heads, panels=panels, value_label="drop in logP(first answer token) when the head's output is zeroed (nats)",
        model=model_id, case_ids=[case_of(p) for p in forget_pairs[:n]], provenance="measured",
        provenance_note="Measured by re-running the forward pass with each head's slice of the o_proj input zeroed; mean over the listed forget prompts.",
        notes=f"Single-head zero-ablation at the last prompt position's logits, {n} forget prompt(s), layers {layers[0]}..{layers[-1]} strided by {cfg_ex['layer_stride']}. "
              "Positive = the head supports the answer token. Zero-ablation is a blunt tool: heads can compensate for each other, so read it as a screen, not a circuit." + suffix,
    )


def distribution_artifacts(*, base_run_id, tokenizer, model_pre, model_post, forget_prompts, retain_prompts, pre_label, post_label, model_id,
                           rmu_layer, case_ids, cfg_ex, suffix=""):
    import numpy as np

    idx = rmu_layer + 1
    k = int(cfg_ex["max_prompts"])
    fp, rp = forget_prompts[:k], retain_prompts[:k]
    if not fp or not rp:
        raise ValueError("distributions need forget and retain prompts")
    res = {}
    for split, prompts in (("forget prompts", fp), ("retain prompts", rp)):
        a = X.residuals_at(model_pre, tokenizer, prompts, idx)
        b = X.residuals_at(model_post, tokenizer, prompts, idx)
        res[split] = (a, b)
    bins = int(cfg_ex["bins"])
    where = f"residual after decoder layer {rmu_layer} (RMU layer)"
    # (1) per-dimension change of the last-token residual
    deltas = {s: np.concatenate([(pb[1] - pa[1]) for pa, pb in zip(a, b)]) for s, (a, b) in res.items()}
    edges = X.shared_edges_from(list(deltas.values()), bins, symmetric=True)
    d1 = A.distribution_artifact(
        base_run_id=base_run_id, name="activation-diff-hist", title=f"Activation change at layer {rmu_layer} ({post_label} − {pre_label}), per residual dimension",
        edges=edges, series={s: A.histogram_counts(v.tolist(), edges) for s, v in deltas.items()}, value_label="Δ activation (post − pre)",
        stats={s: X.summary_stats(v) for s, v in deltas.items()}, markers=[(0.0, "no change")], model=model_id, case_ids=case_ids, provenance="derived",
        provenance_note="Element-wise difference of the two checkpoints' measured last-token residuals, then histogrammed (post-processing).",
        notes=f"{where}, last prompt token, {len(fp)} forget / {len(rp)} retain prompts. Values outside the central 99.5% (|Δ| > {edges[-1]:.3g}) are not counted; bins are shared." + suffix,
    )
    # (2) relative per-token change ||Δ|| / ||pre|| over all prompt tokens
    rel = {}
    for s, (a, b) in res.items():
        vals = [np.linalg.norm(pb[0] - pa[0], axis=-1) / np.maximum(np.linalg.norm(pa[0], axis=-1), 1e-8) for pa, pb in zip(a, b)]
        rel[s] = np.concatenate(vals)
    edges2 = X.shared_edges_from(list(rel.values()), bins, nonnegative=True)
    d2 = A.distribution_artifact(
        base_run_id=base_run_id, name="token-relative-change-hist", title=f"Relative activation change per token at layer {rmu_layer} (‖post − pre‖ / ‖pre‖)",
        edges=edges2, series={s: A.histogram_counts(v.tolist(), edges2) for s, v in rel.items()}, value_label="‖Δ‖ / ‖pre‖", stats={s: X.summary_stats(v) for s, v in rel.items()},
        markers=[(0.0, "no change")], model=model_id, case_ids=case_ids, provenance="derived",
        provenance_note="Ratio of norms of measured residuals (post-processing).",
        notes=f"{where}, every prompt token (chat-formatted questions). Values above the 99.5th percentile are not counted." + suffix,
    )
    return [d1, d2]


def projection_artifacts(*, base_run_id, tokenizer, model_pre, model_post, forget_prompts, retain_prompts, pre_label, post_label, model_id,
                         layers, case_ids, cfg_ex, suffix=""):
    import numpy as np

    k = int(cfg_ex["max_prompts"])
    fp, rp = forget_prompts[:k], retain_prompts[:k]
    if len(fp) + len(rp) < 3 or not fp or not rp:
        raise ValueError("projection needs at least one forget and one retain prompt and 3 points in total")
    arts = []
    for layer in layers:
        idx = layer + 1
        acts = {lab: {s: np.stack([r[1] for r in X.residuals_at(m, tokenizer, ps, idx)]) for s, ps in (("forget prompts", fp), ("retain prompts", rp))}
                for lab, m in ((pre_label, model_pre), (post_label, model_post))}
        mu, comps, ev = X.pca_fit(np.concatenate(list(acts[pre_label].values())), 2)
        panels = {lab: {s: X.pca_project(v, mu, comps).tolist() for s, v in groups.items()} for lab, groups in acts.items()}
        arts.append(A.scatter_artifact(
            base_run_id=base_run_id, name=f"residual-pca-L{layer}", title=f"2-D PCA of last-token residuals at layer {layer} (forget vs retain, {pre_label} vs {post_label})",
            method=(f"PCA (SVD, centered) fit on the pooled {pre_label} forget+retain last-token residuals after decoder layer {layer} ({len(fp)} forget, {len(rp)} retain prompts); "
                    f"the same mean and two components are applied to {post_label} activations so panels share axes. Deterministic, no seed."),
            panels=panels, explained_variance=ev, model=model_id, case_ids=case_ids,
            provenance_note="Activations are measured; a 2-D projection is a derived view and distances in it are not distances in the full residual space." + suffix,
        ))
    return arts


def circuit_head_grid_artifact(paths_by_label: Mapping[str, str | Path], *, base_run_id: str, n_heads: int | None, n_layers: int | None, model: str | None = None, extra_notes: str = "") -> dict[str, Any] | None:
    """head_grid from CircuitKIT node scores `a<L>.h<H>` (EAP-IG attribution, one panel per checkpoint). None if no head nodes parse."""
    grids: dict[str, dict[tuple[int, int], float]] = {}
    for label, path in paths_by_label.items():
        blob = json.loads(Path(path).read_text())
        cells = {}
        for name, score in (blob.get("node_scores") or {}).items():
            info = A.parse_circuit_node(name)
            val = A.clean_number(score)
            if info and info["type"] == "attn" and val is not None:
                cells[(info["layer"], info["head"])] = val
        if cells:
            grids[label] = cells
    if not grids:
        return None
    L = max(n_layers or 0, max(l for g in grids.values() for l, _ in g) + 1)
    H = max(n_heads or 0, max(h for g in grids.values() for _, h in g) + 1)
    panels = {lab: [[g.get((l, h)) for h in range(H)] for l in range(L)] for lab, g in grids.items()}
    return A.head_grid_artifact(
        base_run_id=base_run_id, name="circuit-head-scores", title="CircuitKIT attention-head scores (EAP-IG, TOFU forget-fact task)", layers=list(range(L)), n_heads=H,
        panels=panels, value_label="EAP-IG node score (signed)", scale="diverging", model=model, provenance="measured",
        provenance_note="Attribution scores computed by CircuitKIT from gradients on the forget-fact task; not an ablation effect and not comparable in scale to the head-ablation grid.",
        notes="Node scores for attention heads only; cells without a score are missing. " + extra_notes,
    )


def build_interp_artifacts(
    cfg: Mapping[str, Any],
    *,
    base_run_id: str,
    tokenizer,
    model_pre,
    model_post,
    forget_pairs: Sequence[Mapping[str, str]],
    retain_pairs: Sequence[Mapping[str, str]],
    model_id: str,
    pre_label: str = "pre_rmu",
    post_label: str = "post_rmu",
    trace_stage: str | None = None,
    trace_phases: Sequence[str] = (),
    settings: Mapping[str, Any] | None = None,
    note_suffix: str = "",
    circuit_sources: Mapping[str, str | Path] | None = None,
    retain_pool: Sequence[Mapping[str, str]] | None = None,
    discover_fn=None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Return (artifacts, errors). `trace_stage/trace_phases` link artifacts to the stage's per-case traces.

    `circuit_sources` ({label: saved checkpoint dir}) feeds optional CircuitKIT discovery when
    settings["circuitkit"]["discover"] is true; without it (smoke runs keep models in memory) discovery is skipped.
    """
    st = settings or interp_settings(cfg)
    arts: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    n_layers = len(I.decoder_layers(model_post))
    res_labels = ["emb"] + [f"L{i}" for i in range(n_layers)]
    suffix = (" " + note_suffix) if note_suffix else ""

    def case_of(pair) -> str:
        return case_id_for(split="forget", fmt="direct", item_id=pair["id"])

    def links(cids):
        return A.trace_ids_for_cases(base_run_id, trace_stage, trace_phases, cids) if trace_stage and trace_phases else []

    def guard(name):
        class _G:
            def __enter__(self_):
                return self_

            def __exit__(self_, et, ev, tb):
                if ev is not None:
                    errors.append({"component": name, "error": f"{type(ev).__name__}: {ev}"[:500]})
                    print(f"interp component {name} failed: {type(ev).__name__}: {ev}")
                    return True
                return False

        return _G()

    forget_prompts = [apply_chat_template(tokenizer, p["question"], add_generation_prompt=True) for p in forget_pairs]
    retain_prompts = [apply_chat_template(tokenizer, p["question"], add_generation_prompt=True) for p in retain_pairs]

    with guard("norms"):
        k = 8
        if not forget_prompts[:k] or not retain_prompts[:k]:
            raise ValueError("norm ratios need at least one forget and one retain prompt")
        per = {}
        for split, prompts in (("forget prompts", forget_prompts[:k]), ("retain prompts", retain_prompts[:k])):
            per[split] = {lab: X.per_prompt_last_token_norms(m, tokenizer, prompts) for lab, m in ((pre_label, model_pre), (post_label, model_post))}
        layers_res = list(range(n_layers + 1))
        ratio_series, ratio_bands, counts = {}, {}, {}
        for split, d in per.items():
            ratios = [[(b / a) if a else None for a, b in zip(pa, pb)] for pa, pb in zip(d[pre_label], d[post_label])]
            mean, std = X.mean_std(ratios)
            ratio_series[split], ratio_bands[split], counts[split] = mean, X.band(mean, std), len(ratios)
        arts.append(A.layer_series_artifact(
            base_run_id=base_run_id, name="rmu-norm-ratio", title=f"Residual norm change per layer ({post_label} / {pre_label})",
            metric="mean per-prompt last-token residual L2 norm ratio", layers=layers_res, series=ratio_series, counts=counts, bands=ratio_bands,
            reference_lines=[(1.0, "no change")], model=model_id, provenance="derived",
            provenance_note="Per-prompt ratio of two measured last-token residual norms (post / pre), then averaged over prompts.",
            notes="Layer 0 = embeddings; band = mean ± 1 std of the per-prompt ratios; prompts = chat-formatted questions." + suffix,
        ))
        nf = per["forget prompts"]
        norm_series, norm_bands = {}, {}
        for lab in (pre_label, post_label):
            mean, std = X.mean_std(nf[lab])
            norm_series[lab], norm_bands[lab] = mean, X.band(mean, std)
        arts.append(A.layer_series_artifact(
            base_run_id=base_run_id, name="rmu-norm-band", title="Residual-stream L2 norm by layer, forget prompts (± 1 std)",
            metric="mean last-token residual L2 norm", layers=layers_res, series=norm_series, bands=norm_bands, counts={lab: len(nf[lab]) for lab in nf},
            model=model_id, provenance="measured", provenance_note="Last-token residual L2 norm from a forward pass; band = mean ± 1 std over prompts as recorded.",
            notes="Layer 0 = embeddings." + suffix,
        ))

    with guard("logit_lens"):
        n = int(st["lens"]["max_prompts"])
        pairs = list(forget_pairs[:n])
        finals_pre, finals_post = [], []
        for i, pair in enumerate(pairs):
            lens = {lab: I.lens_probability_grid(m, tokenizer, forget_prompts[i], pair["answer"]) for lab, m in ((pre_label, model_pre), (post_label, model_post))}
            a, b = lens[pre_label], lens[post_label]
            finals_pre.append([row[-1] for row in a["grid"]])
            finals_post.append([row[-1] for row in b["grid"]])
            cid = case_of(pair)
            arts.append(A.heatmap_artifact(
                base_run_id=base_run_id, name=f"logit-lens-{i}", title=f"Logit lens · P(answer token) · forget case {pair['id']}",
                x=a["tokens"], y=res_labels, panels={pre_label: a["grid"], post_label: b["grid"]},
                value_label=f"P(first answer token {a['target_token']!r}) under the logit lens", x_label="prompt position (token)", y_label="residual index (emb = embeddings)",
                value_range=[0, 1], model=model_id, case_ids=[cid], trace_ids=links([cid]), provenance="measured",
                notes="lm_head(final_norm(residual)) * logit_scale, softmax over the full vocabulary. Q: " + pair["question"][:160] + suffix,
            ))
        if finals_pre:
            mean = lambda rows: [sum(col) / len(col) for col in zip(*rows)]
            cids = [case_of(p) for p in pairs]
            arts.append(A.layer_series_artifact(
                base_run_id=base_run_id, name="logit-lens-final-position", title="Logit lens at the final prompt position (mean over forget cases)",
                metric="P(first answer token)", layers=list(range(n_layers + 1)), series={pre_label: mean(finals_pre), post_label: mean(finals_post)},
                counts={pre_label: len(finals_pre), post_label: len(finals_post)}, value_range=[0, 1], model=model_id,
                bands={pre_label: X.band(*X.mean_std(finals_pre)), post_label: X.band(*X.mean_std(finals_post))},
                case_ids=cids, trace_ids=links(cids), provenance="measured",
                notes="Layer 0 = embeddings. Band = mean ± 1 std over forget cases." + suffix,
            ))

    with guard("activation_patching"):
        n = int(st["patching"]["max_prompts"])
        stride = max(1, int(st["patching"]["layer_stride"]))
        layers = list(range(0, n_layers, stride))
        if layers[-1] != n_layers - 1:
            layers.append(n_layers - 1)
        for i, pair in enumerate(list(forget_pairs[:n])):
            out = I.patch_recovery_grid(model_pre, model_post, tokenizer, forget_prompts[i], pair["answer"], layers=layers,
                                        max_positions=int(st["patching"]["max_positions"]), min_gap=float(st["patching"]["min_gap"]))
            cid = case_of(pair)
            arts.append(A.heatmap_artifact(
                base_run_id=base_run_id, name=f"patching-{i}", title=f"Activation patching · recovery of the answer · forget case {pair['id']}",
                x=out["tokens"], y=[f"L{l}" for l in out["layers"]], panels={f"{pre_label} → {post_label}": out["grid"]},
                value_label="recovery = (logP_patched - logP_target) / (logP_source - logP_target)", x_label="patched prompt position (last positions)", y_label="patched decoder layer",
                value_range=[0, 1], model=model_id, case_ids=[cid], trace_ids=links([cid]), provenance="measured",
                notes=(f"Source = {pre_label}, target = {post_label}; residual at one (layer, position) copied from source into target. "
                       f"logP(answer token) source {out['logprob_source']:.3f}, target {out['logprob_target']:.3f}; cells are missing when the gap is below {out['min_gap']} nats. Values are unclipped; the color scale saturates at 0 and 1. Layers strided by {stride}." + suffix),
            ))

    with guard("feature_explorer"):
        fs = st["features"]
        layers = fs.get("layers") or default_feature_layers(cfg, n_layers)
        f_pairs = list(forget_pairs[: int(fs["n_forget_texts"])])
        r_pairs = list(retain_pairs[: int(fs["n_retain_texts"])])
        units_by_layer = I.select_selective_units(model_pre, tokenizer, forget_prompts[: len(f_pairs)], retain_prompts[: len(r_pairs)], layers, per_layer=int(fs["per_layer"]))
        texts = [_full_text(tokenizer, p["question"], p["answer"]) for p in f_pairs + r_pairs]
        cids = [case_of(p) for p in f_pairs] + [case_id_for(split="retain", fmt="direct", item_id=p["id"]) for p in r_pairs]
        trace_for = {}
        if trace_stage and trace_phases:
            for lab, phase in zip((pre_label, post_label), trace_phases):
                trace_for[lab] = [A.trace_ids_for_cases(base_run_id, trace_stage, [phase], [c])[0] for c in cids]
        units = I.feature_units({pre_label: model_pre, post_label: model_post}, tokenizer, units_by_layer, texts, cids,
                                trace_ids_by_condition=trace_for or None, top_examples=int(fs["top_examples"]))
        arts.append(A.feature_explorer_artifact(
            base_run_id=base_run_id, name="mlp-neurons", title="MLP neuron explorer (forget-selective neurons, before vs after)", units=units,
            activation_label="MLP hidden activation (input to down_proj)", model=model_id, case_ids=cids, trace_ids=[t for v in trace_for.values() for t in v], provenance="measured",
            notes=("Units = per layer, the neurons with the largest |mean forget - mean retain| last-token activation on the pre model (a display heuristic, not a claim that they store the facts). "
                   "Examples are TOFU question+answer texts; logit effects are direct-path only." + suffix),
        ))

    ex = st.get("extras") or {}
    if ex.get("enabled", True):
        retain_case_of = lambda pair: case_id_for(split="retain", fmt="direct", item_id=pair["id"])
        rmu_layer = rmu_layer_index(cfg, n_layers)
        f_cids = [case_of(p) for p in forget_pairs[:8]]
        common = dict(base_run_id=base_run_id, tokenizer=tokenizer, model_pre=model_pre, model_post=model_post, pre_label=pre_label, post_label=post_label, model_id=model_id)
        if ex["token_maps"].get("enabled", True):
            with guard("token_layer_map"):
                arts += token_map_artifacts(**common, forget_pairs=forget_pairs, retain_pairs=retain_pairs, forget_prompts=forget_prompts, retain_prompts=retain_prompts,
                                            case_of=case_of, retain_case_of=retain_case_of, links=links, cfg_ex=ex["token_maps"], suffix=suffix)
        if ex["lens_table"].get("enabled", True):
            with guard("logit_lens_table"):
                arts += lens_table_artifacts(**common, forget_pairs=forget_pairs, forget_prompts=forget_prompts, case_of=case_of, links=links, cfg_ex=ex["lens_table"], suffix=suffix)
        if ex["head_ablation"].get("enabled", True):
            with guard("head_ablation"):
                arts.append(head_ablation_artifact(**common, forget_pairs=forget_pairs, forget_prompts=forget_prompts, case_of=case_of, cfg_ex=ex["head_ablation"], suffix=suffix))
        if ex["distributions"].get("enabled", True):
            with guard("distributions"):
                arts += distribution_artifacts(**common, forget_prompts=forget_prompts, retain_prompts=retain_prompts, rmu_layer=rmu_layer, case_ids=f_cids, cfg_ex=ex["distributions"], suffix=suffix)
        if ex["projection"].get("enabled", True):
            with guard("projection"):
                proj_layers = [max(0, min(n_layers - 1, int(l))) for l in (ex["projection"].get("layers") or [rmu_layer])]
                arts += projection_artifacts(**common, forget_prompts=forget_prompts, retain_prompts=retain_prompts, layers=proj_layers, case_ids=f_cids, cfg_ex=ex["projection"], suffix=suffix)
        X.free_gpu_cache()

    ck_cfg = st.get("circuitkit") or {}
    scores_by_label: dict[str, str] = {k: v for k, v in (ck_cfg.get("scores") or {}).items() if v}
    task_text = ""
    if ck_cfg.get("discover"):
        if not circuit_sources:
            print("circuitkit.discover is on but no saved checkpoint dirs were given (smoke keeps models in memory); skipping discovery.")
        else:
            with guard("circuitkit_discover"):
                if retain_pool is not None:
                    pool = list(retain_pool)
                else:
                    from .data import load_tofu_split, qa_pairs

                    tofu = cfg["tofu"]
                    pool = qa_pairs(load_tofu_split(tofu["retain_split"], repo_id=tofu["repo_id"], max_examples=int(ck_cfg.get("retain_pool") or 1000)))
                kwargs = {"discover_fn": discover_fn} if discover_fn is not None else {}
                found = CT.run_discovery(
                    sources=circuit_sources, forget_pairs=forget_pairs, retain_pool=pool, tokenizer=tokenizer,
                    out_dir=output_dir(cfg) / "circuitkit" / base_run_id, settings=ck_cfg, **kwargs,
                )
                scores_by_label.update(found["scores"])
                task_text = CT.task_note(found["task"])
                for e in found["errors"]:
                    errors.append({"component": f"circuitkit_discover:{e['label']}", "error": e["error"]})

    with guard("circuitkit_scores"):
        for label, path in scores_by_label.items():
            if path and Path(path).exists():
                arts.append(circuit_scores_artifact(path, base_run_id=base_run_id, name=f"circuit-{label}", title=f"CircuitKIT circuit · {label}", phase=label, model=model_id, top_fraction=float(ck_cfg.get("top_fraction") or 0.2), extra_notes=task_text))

    with guard("circuitkit_head_grid"):
        existing_hg = {k: v for k, v in scores_by_label.items() if v and Path(v).exists()}
        if existing_hg:
            try:
                n_heads_cfg = X.head_geometry(model_post)[0]
            except Exception:
                n_heads_cfg = None
            hg = circuit_head_grid_artifact(existing_hg, base_run_id=base_run_id, n_heads=n_heads_cfg, n_layers=n_layers, model=model_id, extra_notes=task_text)
            if hg:
                arts.append(hg)

    with guard("circuitkit_profile"):
        existing = {k: v for k, v in scores_by_label.items() if v and Path(v).exists()}
        if len(existing) >= 1 and ck_cfg.get("discover"):
            prof = circuit_layer_profile_artifact(existing, base_run_id=base_run_id, model=model_id, extra_notes=task_text)
            if prof:
                arts.append(prof)

    return arts, errors
