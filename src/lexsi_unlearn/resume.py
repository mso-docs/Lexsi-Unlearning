"""Resume a pipeline run from RMU onward, reusing the valid pre-RMU numbers of an earlier run.

2026-11-12: the first full run's RMU was a no-op (5 optimizer steps). Its base / injected / before-RMU evaluations are
still valid and expensive (judge calls), so a rerun with `--stage unlearn,eval,probes,interp,relearn
--resume-from-results <old results.json>` copies them into the new results JSON and trace file instead of recomputing.
Only fields computed on the base or the injected checkpoint are carried; nothing post-RMU is ever copied.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Computed on the never-injected base model or the injected checkpoint only.
PRE_RMU_KEYS = (
    "base", "judge_base", "judge_base_headline",
    "inject_dir", "injected",
    "before_unlearn", "judge_before_unlearn", "judge_before_unlearn_headline",
)
# (stage, phase) of trace rows that belong to those keys.
PRE_RMU_TRACE_PHASES = (("inject", "base"), ("verify", "injected"), ("eval", "before_unlearn"))


def load_pre_rmu_results(path: str | Path) -> dict[str, Any]:
    """Return {'run_name', 'traces_path', 'values': {key: value}} for the pre-RMU keys present in the old results JSON."""
    old = json.loads(Path(path).read_text())
    values = {k: old[k] for k in PRE_RMU_KEYS if k in old}
    for need in ("base", "injected"):
        if need not in values:
            raise ValueError(f"{path}: no '{need}' results to resume from (run inject+verify first)")
    return {"run_name": old.get("run_name"), "traces_path": old.get("traces_path"), "values": values}


def _is_pre_rmu_row(row: dict[str, Any]) -> bool:
    return (row.get("stage"), row.get("phase")) in PRE_RMU_TRACE_PHASES


def carry_pre_rmu_traces(old_trace: str | Path, new_trace: str | Path, old_name: str, new_name: str) -> int:
    """Append the old trace's pre-RMU rows to the new trace file with the run id rewritten. Returns rows copied."""
    n = 0
    with open(old_trace) as src, open(new_trace, "a") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not _is_pre_rmu_row(row):
                continue
            dst.write(line.replace(old_name, new_name) + "\n")
            n += 1
    return n
