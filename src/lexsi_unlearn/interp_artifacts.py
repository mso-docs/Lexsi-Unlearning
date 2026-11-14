"""Interpretability artifacts for the Evalboard `interp` tab (schema evalboard.interp/v1).

Contract: modular_dashboard/skills/build-eval-dashboard/references/interp.md.
No torch import here, so this module is cheap to import and to test on the Mac.

Rules (same as traces.py): only record what was computed. Missing cells are None.
Artifacts are written as one JSON file each under
    ${OUTPUT_DIR}/interp/<base_run_id>/<artifact id, filesystem-safe>.json
and indexed in ${OUTPUT_DIR}/interp/<base_run_id>/index.jsonl.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .scrub import scrub_obj
from .traces import output_dir, staged_run_id, utc_now

SCHEMA = "evalboard.interp/v1"
_UNSAFE = re.compile(r"[^A-Za-z0-9._+-]+")


def clean_number(value: Any) -> float | None:
    """float(value) when finite, else None. Never invents a number."""
    if value is None or isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def clean_row(values: Sequence[Any]) -> list[float | None]:
    return [clean_number(v) for v in values]


def artifact_id(base_run_id: str, name: str) -> str:
    """Stable, contract-legal id (letters, digits, `_ . : + @ = -`)."""
    return f"{base_run_id}:{_UNSAFE.sub('-', name).strip('-')}"[:200]


def envelope(
    *,
    base_run_id: str,
    name: str,
    kind: str,
    title: str,
    data: Mapping[str, Any],
    model: str | None = None,
    phase: str | None = None,
    notes: str | None = None,
    case_ids: Sequence[str] = (),
    trace_ids: Sequence[str] = (),
    provenance: str | None = None,
    provenance_note: str | None = None,
) -> dict[str, Any]:
    art: dict[str, Any] = {
        "schema": SCHEMA,
        "id": artifact_id(base_run_id, name),
        "kind": kind,
        "title": title,
        "run_id": base_run_id,
        "created_at": utc_now(),
        "case_ids": list(dict.fromkeys(case_ids)),
        "trace_ids": list(dict.fromkeys(trace_ids)),
        "data": dict(data),
    }
    if model:
        art["model"] = model
    if phase:
        art["phase"] = phase
    if notes:
        art["notes"] = notes
    if provenance:
        if provenance not in ("measured", "derived", "mixed"):
            raise ValueError("provenance must be measured, derived, or mixed")
        art["provenance"] = provenance
    if provenance_note:
        art["provenance_note"] = provenance_note
    return art


# ---- builders ---------------------------------------------------------------------------------


def layer_series_artifact(
    *,
    base_run_id: str,
    name: str,
    title: str,
    metric: str,
    layers: Sequence[int],
    series: Mapping[str, Sequence[Any]],
    counts: Mapping[str, int] | None = None,
    value_range: Sequence[float] | None = None,
    bands: Mapping[str, tuple[Sequence[Any], Sequence[Any]]] | None = None,
    reference_lines: Sequence[tuple[float, str]] = (),
    **kw: Any,
) -> dict[str, Any]:
    """One curve per entry of `series` (name -> one value per layer; None = not computed).

    `bands` optionally maps a series name to (lo, hi) rows (e.g. mean +/- std recorded by the producer);
    `reference_lines` draws labelled horizontal lines such as (1.0, "no change").
    """
    items = []
    for label, values in series.items():
        item: dict[str, Any] = {"name": label, "values": clean_row(values)}
        if counts and label in counts:
            item["n"] = int(counts[label])
        if bands and label in bands:
            item["lo"], item["hi"] = clean_row(bands[label][0]), clean_row(bands[label][1])
        items.append(item)
    data: dict[str, Any] = {"metric": metric, "layers": [int(l) for l in layers], "range": list(value_range) if value_range else None, "series": items}
    if reference_lines:
        data["reference_lines"] = [{"value": float(v), "label": str(l)} for v, l in reference_lines]
    return envelope(base_run_id=base_run_id, name=name, kind="layer_series", title=title, data=data, **kw)


def heatmap_artifact(
    *,
    base_run_id: str,
    name: str,
    title: str,
    x: Sequence[str],
    y: Sequence[str],
    panels: Mapping[str, Sequence[Sequence[Any]]],
    value_label: str,
    x_label: str,
    y_label: str,
    value_range: Sequence[float] | None = None,
    scale: str = "sequential",
    **kw: Any,
) -> dict[str, Any]:
    return envelope(
        base_run_id=base_run_id,
        name=name,
        kind="heatmap",
        title=title,
        data={
            "value_label": value_label,
            "x_label": x_label,
            "y_label": y_label,
            "x": [str(v) for v in x],
            "y": [str(v) for v in y],
            "range": list(value_range) if value_range else None,
            "scale": scale,
            "panels": [{"name": k, "values": [clean_row(r) for r in rows]} for k, rows in panels.items()],
        },
        **kw,
    )


def feature_explorer_artifact(
    *, base_run_id: str, name: str, title: str, units: Sequence[Mapping[str, Any]], activation_label: str, **kw: Any
) -> dict[str, Any]:
    return envelope(
        base_run_id=base_run_id,
        name=name,
        kind="feature_explorer",
        title=title,
        data={"activation_label": activation_label, "units": list(units)},
        **kw,
    )


_MLP = (re.compile(r"^m(\d+)$"), re.compile(r"^MLP (\d+)$"))
_HEAD = (re.compile(r"^a(\d+)\.h(\d+)$"), re.compile(r"^A(\d+)\.(\d+)$"))


def parse_circuit_node(name: str) -> dict[str, Any] | None:
    """Parse CircuitKIT node names: EAP `m3` / `a3.h1`, IBCircuit `MLP 3` / `A3.1`.

    Returns None for other names (e.g. `input`, `logits`) so callers can keep them out of the graph.
    """
    for pat in _MLP:
        m = pat.match(name)
        if m:
            return {"type": "mlp", "layer": int(m.group(1)), "head": None}
    for pat in _HEAD:
        m = pat.match(name)
        if m:
            return {"type": "attn", "layer": int(m.group(1)), "head": int(m.group(2))}
    return None


def circuit_graph_from_scores(
    node_scores: Mapping[str, float],
    *,
    pruned: Sequence[str] = (),
    edges: Sequence[Mapping[str, Any]] | None = None,
    edge_kind: str | None = None,
    max_nodes: int = 400,
    max_edges: int = 2000,
) -> dict[str, Any]:
    """Build circuit_graph `data` from CircuitKIT node scores (`Circuit.scores`).

    CircuitKIT's flat API (circuitkit 0.2.3) returns node scores, not edges. Its own visualiser
    draws `layer_adjacent` edges (weight = product of adjacent-layer node scores), which are not
    measured connections. We therefore either use explicit `edges` (declare edge_kind='measured'
    only if they were measured) or derive layer-adjacent edges and label them as such.
    Nodes not matching CircuitKIT's head/MLP naming are skipped and counted.
    """
    pruned_set = set(pruned)
    parsed = []
    skipped = 0
    for name, score in node_scores.items():
        info = parse_circuit_node(name)
        s = clean_number(score)
        if info is None or s is None:
            skipped += 1
            continue
        parsed.append({"id": name, "label": name, "score": s, "in_circuit": name not in pruned_set, **info})
    parsed.sort(key=lambda n: (-abs(n["score"]), n["id"]))
    hidden = max(0, len(parsed) - max_nodes)
    nodes = parsed[:max_nodes]
    for n in nodes:
        if n["head"] is None:
            del n["head"]
    ids = {n["id"] for n in nodes}
    if edges is None:
        kind = edge_kind or "layer_adjacent"
        if kind != "layer_adjacent":
            raise ValueError("derived edges must use edge_kind='layer_adjacent'")
        by_layer: dict[int, list[dict[str, Any]]] = {}
        for n in nodes:
            if n["in_circuit"]:
                by_layer.setdefault(n["layer"], []).append(n)
        out = []
        for layer, src in by_layer.items():
            for a in src:
                for b in by_layer.get(layer + 1, []):
                    out.append({"source": a["id"], "target": b["id"], "weight": a["score"] * b["score"]})
        out.sort(key=lambda e: -abs(e["weight"]))
    else:
        kind = edge_kind or "derived"
        out = [
            {"source": str(e["source"]), "target": str(e["target"]), "weight": clean_number(e.get("weight"))}
            for e in edges
            if str(e.get("source")) in ids and str(e.get("target")) in ids and clean_number(e.get("weight")) is not None
        ]
    n_edges_hidden = max(0, len(out) - max_edges)
    return {
        "nodes": nodes,
        "edges": out[:max_edges],
        "edge_kind": kind,
        "score_label": "CircuitKIT node importance",
        "reduction": {"nodes_hidden": hidden, "nodes_skipped_unparsed": skipped, "edges_hidden": n_edges_hidden},
    }


def circuit_graph_artifact(*, base_run_id: str, name: str, title: str, graph: Mapping[str, Any], algorithm: str | None = None, task: str | None = None, **kw: Any) -> dict[str, Any]:
    data = dict(graph)
    if algorithm:
        data["algorithm"] = algorithm
    if task:
        data["task"] = task
    red = data.get("reduction") or {}
    notes = kw.pop("notes", None) or (
        f"Reduced to top nodes by |score|: {red.get('nodes_hidden', 0)} nodes and {red.get('edges_hidden', 0)} edges not included."
    )
    return envelope(base_run_id=base_run_id, name=name, kind="circuit_graph", title=title, data=data, notes=notes, **kw)


# ---- extended views (token x layer, logit lens, heads, distributions, projections) --------------
# Same honesty rules: pass None for anything not computed, and say in `notes` / `provenance_note`
# how a number was produced. `provenance="derived"` for projections, post-processed or estimated values.


def token_layer_map_artifact(
    *,
    base_run_id: str,
    name: str,
    title: str,
    layers: Sequence[int],
    examples: Sequence[Mapping[str, Any]],
    value_label: str,
    value_range: Sequence[float] | None = None,
    scale: str = "sequential",
    **kw: Any,
) -> dict[str, Any]:
    """Per-token activation across layers. Each example: {id, label?, case_id?, trace_id?, tokens, panels: {name: [[per token] per layer]}}."""
    items = []
    for ex in examples:
        item = {k: ex[k] for k in ("id", "label", "case_id", "trace_id") if ex.get(k) is not None}
        item["tokens"] = [str(t) for t in ex["tokens"]]
        item["panels"] = [{"name": k, "values": [clean_row(r) for r in rows]} for k, rows in ex["panels"].items()]
        items.append(item)
    data = {"value_label": value_label, "layers": [int(l) for l in layers], "range": list(value_range) if value_range else None, "scale": scale, "examples": items}
    return envelope(base_run_id=base_run_id, name=name, kind="token_layer_map", title=title, data=data, **kw)


def logit_lens_table_artifact(
    *,
    base_run_id: str,
    name: str,
    title: str,
    panels: Mapping[str, Sequence[Mapping[str, Any]]],
    value_label: str = "probability",
    position_label: str = "final prompt position",
    target_token: str | None = None,
    **kw: Any,
) -> dict[str, Any]:
    """Top tokens per layer. panels: {condition: [{layer, top: [(token, value), ...], target_rank?, target_value?}]}."""
    out = []
    for pname, rows in panels.items():
        items = []
        for r in rows:
            top = [{"token": str(t), "value": clean_number(v)} for t, v in r["top"] if clean_number(v) is not None]
            if not top:
                continue
            row: dict[str, Any] = {"layer": int(r["layer"]), "top": top}
            for k in ("target_rank", "target_value"):
                if clean_number(r.get(k)) is not None:
                    row[k] = clean_number(r[k])
            items.append(row)
        out.append({"name": pname, "layers": items})
    data: dict[str, Any] = {"value_label": value_label, "position_label": position_label, "panels": out}
    if target_token is not None:
        data["target"] = {"token": target_token}
    return envelope(base_run_id=base_run_id, name=name, kind="logit_lens_table", title=title, data=data, **kw)


def head_grid_artifact(
    *,
    base_run_id: str,
    name: str,
    title: str,
    layers: Sequence[int],
    n_heads: int,
    panels: Mapping[str, Sequence[Sequence[Any]]],
    value_label: str,
    value_range: Sequence[float] | None = None,
    scale: str = "sequential",
    **kw: Any,
) -> dict[str, Any]:
    """Attention-head importance as layer x head grids (e.g. ablation effect, EAP-IG score), one panel per condition."""
    data = {
        "value_label": value_label,
        "layers": [int(l) for l in layers],
        "n_heads": int(n_heads),
        "range": list(value_range) if value_range else None,
        "scale": scale,
        "panels": [{"name": k, "values": [clean_row(r) for r in rows]} for k, rows in panels.items()],
    }
    return envelope(base_run_id=base_run_id, name=name, kind="head_grid", title=title, data=data, **kw)


def histogram_counts(values: Sequence[Any], edges: Sequence[float]) -> list[int]:
    """Counts per bin on shared `edges` (last bin closed); values outside the range are not counted."""
    counts = [0] * (len(edges) - 1)
    for v in values:
        f = clean_number(v)
        if f is None or f < edges[0] or f > edges[-1]:
            continue
        lo, hi = 0, len(counts) - 1
        while lo < hi:  # rightmost bin whose left edge <= f
            mid = (lo + hi + 1) // 2
            if edges[mid] <= f:
                lo = mid
            else:
                hi = mid - 1
        counts[lo] += 1
    return counts


def distribution_artifact(
    *,
    base_run_id: str,
    name: str,
    title: str,
    edges: Sequence[float],
    series: Mapping[str, Sequence[int]],
    value_label: str,
    x_label: str | None = None,
    y_scale: str = "linear",
    stats: Mapping[str, Mapping[str, Any]] | None = None,
    markers: Sequence[tuple[float, str]] = (),
    **kw: Any,
) -> dict[str, Any]:
    """Overlaid histograms on shared bin edges (activation differences, feature activation density, ...)."""
    items = []
    for label, counts in series.items():
        item: dict[str, Any] = {"name": label, "counts": [int(c) for c in counts], "n": int(sum(counts))}
        if stats and label in stats:
            item["stats"] = {k: clean_number(v) for k, v in stats[label].items()}
        items.append(item)
    data: dict[str, Any] = {"value_label": value_label, "x_label": x_label or value_label, "y_scale": y_scale, "edges": [float(e) for e in edges], "series": items}
    if markers:
        data["markers"] = [{"value": float(v), "label": str(l)} for v, l in markers]
    return envelope(base_run_id=base_run_id, name=name, kind="distribution", title=title, data=data, **kw)


def scatter_artifact(
    *,
    base_run_id: str,
    name: str,
    title: str,
    method: str,
    panels: Mapping[str, Mapping[str, Sequence[Sequence[float]]]],
    x_label: str = "PC1",
    y_label: str = "PC2",
    explained_variance: Sequence[float | None] | None = None,
    **kw: Any,
) -> dict[str, Any]:
    """2-D projection of activations. panels: {condition: {group: [[x, y], ...]}}. Always a derived view (provenance defaults to 'derived')."""
    kw.setdefault("provenance", "derived")
    data: dict[str, Any] = {
        "method": method,
        "x_label": x_label,
        "y_label": y_label,
        "explained_variance": [clean_number(v) for v in explained_variance] if explained_variance else None,
        "panels": [
            {"name": pname, "groups": [{"name": g, "points": [[float(x), float(y)] for x, y in pts]} for g, pts in groups.items()]}
            for pname, groups in panels.items()
        ],
    }
    return envelope(base_run_id=base_run_id, name=name, kind="scatter", title=title, data=data, **kw)


# ---- writer -------------------------------------------------------------------------------------


class InterpWriter:
    """Writes artifact JSON files for one pipeline invocation."""

    def __init__(self, base_run_id: str, cfg: Mapping[str, Any], *, root: Path | None = None):
        self.base_run_id = base_run_id
        self.dir = output_dir(cfg, root=root) / "interp" / base_run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index = self.dir / "index.jsonl"
        self.written: list[str] = []

    def write(self, art: Mapping[str, Any]) -> Path:
        path = self.dir / (_UNSAFE.sub("-", str(art["id"])) + ".json")
        path.write_text(json.dumps(scrub_obj(art), ensure_ascii=False, allow_nan=False))
        entry = {k: art.get(k) for k in ("id", "kind", "title", "phase")}
        entry["file"] = path.name
        with self.index.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
        self.written.append(str(art["id"]))
        return path

    def write_all(self, arts: Sequence[Mapping[str, Any]]) -> int:
        for a in arts:
            self.write(a)
        return len(arts)


def trace_ids_for_cases(base_run_id: str, stage: str, phases: Sequence[str], case_ids: Sequence[str]) -> list[str]:
    """Trace ids that traces.query_trace writes for (stage, phase, case): `<run>:<stage>-<phase>:<case_id>:<phase>`.

    Mirrors traces.query_trace / TraceWriter.run_id_for so artifacts can link to the pre/post traces
    of the same cases. Ids that were never written are ignored by Evalboard's UI.
    """
    ids = []
    for phase in phases:
        rid = staged_run_id(base_run_id, stage, phase=phase)
        ids.extend(f"{rid}:{cid}:{phase}" for cid in case_ids)
    return ids
