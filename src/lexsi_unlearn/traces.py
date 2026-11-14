"""Canonical Evalboard v1 traces for Lexsi unlearning runs.

Contract: modular_dashboard README "Canonical trace contract (v1)".
Required nonempty strings: id, run_id, case_id, model.
Optional evaluations: {metric, value, scorer, rationale} with value in [0, 1]
and higher-is-better; record any transformation in rationale.

Do not invent scores. Only emit evaluations for metrics that were actually
computed. Producer-specific fields are preserved by Evalboard.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .data import repo_root
from .scrub import scrub_obj

_SAFE_TOKEN = re.compile(r"[^A-Za-z0-9._+-]+")


def utc_now() -> str:
    """UTC timestamp in the contract's Z form, e.g. 2026-11-12T17:00:00Z."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_token(value: str, *, fallback: str = "x") -> str:
    text = _SAFE_TOKEN.sub("-", str(value).strip())
    text = text.strip("-") or fallback
    return text[:80]


def config_hash(cfg: Mapping[str, Any] | None) -> str:
    payload = json.dumps(cfg or {}, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:8]


def make_run_id(
    stages: Sequence[str],
    cfg: Mapping[str, Any] | None = None,
    *,
    timestamp: str | None = None,
    digest: str | None = None,
) -> str:
    """Stable id: <UTC compact timestamp>-<stage(s)>-<config hash>.

    Example: 20261112T170000Z-smoke-a1b2c3d4
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stage_part = _safe_token("+".join(str(s) for s in stages if s), fallback="run")
    return f"{timestamp}-{stage_part}-{digest or config_hash(cfg)}"


def output_dir(cfg: Mapping[str, Any] | None = None, *, root: Path | None = None) -> Path:
    """Resolve ${OUTPUT_DIR} (env, then config paths.artifacts_dir, else artifacts)."""
    root = root or repo_root()
    cfg = cfg or {}
    paths = cfg.get("paths") or {}
    raw = os.environ.get("OUTPUT_DIR") or paths.get("artifacts_dir") or "artifacts"
    path = Path(str(raw))
    if not path.is_absolute():
        path = root / path
    return path


def traces_path(run_id: str, cfg: Mapping[str, Any] | None = None, *, root: Path | None = None) -> Path:
    return output_dir(cfg, root=root) / "traces" / f"{run_id}.jsonl"


def case_id_for(
    *,
    split: str,
    fmt: str,
    item_id: str | int,
    step: int | None = None,
) -> str:
    """Stable dataset case id, shared across before/after run_ids of one experiment.

    Examples:
        forget:direct:0
        retain:direct:3
        forget:paraphrase:0
        forget:translation_es:0
        forget:extraction:0
        relearn:old:step-30:0
    """
    parts = [str(split), str(fmt)]
    if step is not None:
        parts.append(f"step-{int(step)}")
    parts.append(str(item_id))
    return ":".join(parts)


def staged_run_id(base_run_id: str, stage: str, *, phase: str | None = None) -> str:
    """Per-stage (and optional phase) run_id so compare_runs can pair case_ids."""
    suffix = _safe_token(stage)
    if phase:
        suffix = f"{suffix}-{_safe_token(phase)}"
    return f"{base_run_id}:{suffix}"


def normalize_value(
    raw: float,
    *,
    lower: float = 0.0,
    upper: float = 1.0,
    direction: str = "maximize",
) -> tuple[float, str]:
    """Linear map to [0, 1], higher-is-better. Returns (value, transformation)."""
    if upper == lower:
        raise ValueError("normalize_value: upper and lower must differ")
    scaled = (float(raw) - lower) / (upper - lower)
    if direction == "minimize":
        scaled = 1.0 - scaled
        transform = (
            f"linear-invert: value = 1 - (raw - {lower}) / ({upper} - {lower}); "
            f"raw={raw}"
        )
    elif direction == "maximize":
        transform = (
            f"linear: value = (raw - {lower}) / ({upper} - {lower}); raw={raw}"
            if (lower, upper) != (0.0, 1.0)
            else f"identity: raw already in [0, 1]; raw={raw}"
        )
    else:
        raise ValueError(f"direction must be maximize or minimize, got {direction!r}")
    if scaled < 0.0 or scaled > 1.0:
        raise ValueError(
            f"normalized value {scaled} is outside [0, 1] "
            f"(raw={raw}, lower={lower}, upper={upper})"
        )
    return scaled, transform


def evaluation(
    metric: str,
    raw: float,
    *,
    scorer: str,
    rationale: str,
    lower: float = 0.0,
    upper: float = 1.0,
    direction: str = "maximize",
) -> dict[str, Any]:
    value, transform = normalize_value(raw, lower=lower, upper=upper, direction=direction)
    return {
        "metric": metric,
        "value": value,
        "scorer": scorer,
        "rationale": f"{rationale} Transformation: {transform}.",
    }


def query_evaluations(
    *,
    substring_match: float | None,
    exact_match: float | None = None,
    split: str,
) -> list[dict[str, Any]]:
    """Per-query metrics. substring/exact are already in {0, 1}.

    fact_recall is always higher-is-better (more of the target was expressed).
    On the forget split, also emit forget_suppression = 1 - fact_recall so
    unlearning success is higher-is-better, with the inversion recorded.
    """
    evs: list[dict[str, Any]] = []
    if substring_match is not None:
        evs.append(
            evaluation(
                "fact_recall",
                float(substring_match),
                scorer="lexsi_unlearn.evaluate.substring_match",
                rationale=(
                    "fact_recall is 1 if the reference answer (lowercased) is a "
                    "substring of the model output (lowercased), else 0. Higher "
                    "means more of the designated fact was expressed."
                ),
            )
        )
        if split.startswith("forget") or split == "forget":
            evs.append(
                evaluation(
                    "forget_suppression",
                    float(substring_match),
                    scorer="lexsi_unlearn.evaluate.substring_match",
                    direction="minimize",
                    rationale=(
                        "forget_suppression inverts fact_recall so that higher "
                        "is better unlearning (the designated fact was not expressed)."
                    ),
                )
            )
    if exact_match is not None:
        evs.append(
            evaluation(
                "exact_match",
                float(exact_match),
                scorer="lexsi_unlearn.evaluate.exact_match",
                rationale=(
                    "exact_match is 1 if stripped lowercased prediction equals "
                    "the stripped lowercased target, else 0. Higher is better "
                    "agreement with the reference."
                ),
            )
        )
    return evs


def make_trace(
    *,
    trace_id: str,
    run_id: str,
    case_id: str,
    model: str,
    messages: list[dict[str, str]] | None = None,
    evaluations: list[dict[str, Any]] | None = None,
    status: str = "ok",
    created_at: str | None = None,
    duration_ms: float = 0,
    cost_usd: float = 0,
    spans: list[dict[str, Any]] | None = None,
    memory_events: list[dict[str, Any]] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    trace: dict[str, Any] = {
        "id": str(trace_id),
        "run_id": str(run_id),
        "case_id": str(case_id),
        "model": str(model),
        "created_at": created_at or utc_now(),
        "status": status,
        "duration_ms": duration_ms,
        "cost_usd": cost_usd,
        "messages": messages or [],
        "spans": spans or [],
        "memory_events": memory_events or [],
    }
    if evaluations:
        trace["evaluations"] = evaluations
    for key, value in extra.items():
        if key in trace:
            continue
        if value is not None:
            trace[key] = value
    return trace



def _with_judge_evaluations(
    evs: list[dict[str, Any]],
    item: Mapping[str, Any],
) -> list[dict[str, Any]] | None:
    """Append already-normalized judge evaluations when present. Skip invalid rows."""
    extra = item.get("judge_evaluations") or []
    if extra:
        evs = list(evs) + [
            dict(e)
            for e in extra
            if isinstance(e, Mapping) and "metric" in e and "value" in e
        ]
    return evs or None


def query_trace(
    *,
    run_id: str,
    model: str,
    item: Mapping[str, Any],
    prediction: str,
    split: str,
    stage: str,
    phase: str | None = None,
    duration_ms: float = 0,
    status: str = "ok",
    step: int | None = None,
) -> dict[str, Any]:
    item_id = item.get("id", item.get("case_id", "unknown"))
    fmt = str(item.get("format") or "direct")
    prompt = str(item.get("prompt") or item.get("question") or "")
    target = item.get("target") if "target" in item else item.get("answer")
    pred = "" if prediction is None else str(prediction)
    target_text = "" if target is None else str(target)
    sub = item.get("substring_match")
    if sub is None and target_text:
        sub = 1.0 if target_text.strip().lower() in pred.strip().lower() else 0.0
    exact = item.get("exact_match")
    if exact is None and target_text:
        exact = 1.0 if pred.strip().lower() == target_text.strip().lower() else 0.0
    cid = case_id_for(split=split, fmt=fmt, item_id=item_id, step=step)
    phase_part = f":{phase}" if phase else ""
    return make_trace(
        trace_id=f"{run_id}:{cid}{phase_part}",
        run_id=run_id,
        case_id=cid,
        model=model,
        messages=[
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": pred},
        ],
        evaluations=_with_judge_evaluations(
            query_evaluations(
                substring_match=None if sub is None else float(sub),
                exact_match=None if exact is None else float(exact),
                split=split,
            ),
            item,
        ),
        status=status,
        duration_ms=duration_ms,
        spans=[
            {
                "id": "root",
                "name": f"{stage}/{split}/{fmt}",
                "kind": "generation",
                "duration_ms": duration_ms,
                "status": status,
            }
        ],
        stage=stage,
        phase=phase,
        split=split,
        format=fmt,
        target=target_text,
        language=item.get("language"),
        source_item_id=str(item_id),
        judge=item.get("judge"),
    )


def traces_from_eval_result(
    result: Mapping[str, Any],
    items: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
    model: str,
    split: str,
    stage: str,
    phase: str | None = None,
    step: int | None = None,
) -> list[dict[str, Any]]:
    """Build one canonical trace per eval item. Skip if there are no predictions."""
    records = result.get("items") or []
    preds = result.get("predictions") or []
    traces: list[dict[str, Any]] = []
    n = max(len(records), len(items), len(preds))
    for i in range(n):
        item = dict(items[i]) if i < len(items) else {}
        if i < len(records) and isinstance(records[i], Mapping):
            # Per-item scores/predictions from evaluate_model_on_items win.
            merged = dict(item)
            merged.update(records[i])
            item = merged
        pred = item.get("prediction")
        if pred is None and i < len(preds):
            pred = preds[i]
        if pred is None and not item.get("prompt") and not item.get("question"):
            continue
        traces.append(
            query_trace(
                run_id=run_id,
                model=model,
                item=item,
                prediction="" if pred is None else str(pred),
                split=split,
                stage=stage,
                phase=phase,
                step=step,
            )
        )
    return traces


def stage_trace(
    *,
    run_id: str,
    model: str,
    stage: str,
    status: str = "ok",
    duration_ms: float = 0,
    note: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """A stage-level breadcrumb with no fabricated scores."""
    payload = dict(extra or {})
    messages = []
    if note:
        messages.append({"role": "system", "content": note})
    return make_trace(
        trace_id=f"{run_id}:stage:{stage}",
        run_id=run_id,
        case_id=f"stage:{stage}",
        model=model,
        messages=messages,
        status=status,
        duration_ms=duration_ms,
        spans=[
            {
                "id": "root",
                "name": stage,
                "kind": "stage",
                "duration_ms": duration_ms,
                "status": status,
            }
        ],
        stage=stage,
        **payload,
    )


def append_jsonl(path: Path, traces: Iterable[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(traces)
    if not rows:
        return path
    with path.open("a", encoding="utf-8") as handle:
        for trace in rows:
            # Scrub at write time: env secrets -> <REDACTED:NAME>, host identifiers, home paths, emails, IPs.
            handle.write(json.dumps(scrub_obj(json.loads(json.dumps(trace, default=str))), ensure_ascii=False) + "\n")
    return path


class TraceWriter:
    """Append-only JSONL writer for one pipeline invocation."""

    def __init__(self, base_run_id: str, cfg: Mapping[str, Any], *, model: str):
        self.base_run_id = base_run_id
        self.cfg = cfg
        self.model = model
        self.path = traces_path(base_run_id, cfg)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def run_id_for(self, stage: str, *, phase: str | None = None) -> str:
        return staged_run_id(self.base_run_id, stage, phase=phase)

    def write(self, traces: Iterable[Mapping[str, Any]]) -> int:
        rows = list(traces)
        append_jsonl(self.path, rows)
        return len(rows)

    def write_eval(
        self,
        result: Mapping[str, Any],
        items: Sequence[Mapping[str, Any]],
        *,
        split: str,
        stage: str,
        phase: str | None = None,
        step: int | None = None,
    ) -> int:
        return self.write(
            traces_from_eval_result(
                result,
                items,
                run_id=self.run_id_for(stage, phase=phase),
                model=self.model,
                split=split,
                stage=stage,
                phase=phase,
                step=step,
            )
        )

    def write_eval_groups(
        self,
        groups: Mapping[str, Mapping[str, Any]],
        items_by_fmt: Mapping[str, Sequence[Mapping[str, Any]]],
        *,
        split: str,
        stage: str,
        phase: str | None = None,
        step: int | None = None,
    ) -> int:
        n = 0
        for fmt, result in groups.items():
            rows = items_by_fmt.get(fmt) or []
            n += self.write_eval(
                result, rows, split=split, stage=stage, phase=phase, step=step
            )
        return n

    def write_stage(
        self,
        stage: str,
        *,
        status: str = "ok",
        duration_ms: float = 0,
        note: str | None = None,
        extra: Mapping[str, Any] | None = None,
        phase: str | None = None,
    ) -> int:
        return self.write(
            [
                stage_trace(
                    run_id=self.run_id_for(stage, phase=phase),
                    model=self.model,
                    stage=stage,
                    status=status,
                    duration_ms=duration_ms,
                    note=note,
                    extra=extra,
                )
            ]
        )
