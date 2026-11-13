#!/usr/bin/env python3
"""Print OpenRouter spend from the ledger: totals per project and model, and the remaining Lexsi budget.

    python scripts/spend_report.py                      # logs/openrouter_spend.jsonl, cap $OPENROUTER_BUDGET_USD (15)
    python scripts/spend_report.py --ledger PATH --json
    python scripts/spend_report.py --by-run             # also list spend per run_id

Stdlib only, reads the ledger only (no network, no API key). Rows flagged `estimated` are priced from token
counts rather than OpenRouter's usage.cost, so they are shown separately.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lexsi_unlearn import spend  # noqa: E402


def summarize(rows: list[dict]) -> dict:
    by_project: dict[str, dict] = defaultdict(lambda: {"cost_usd": 0.0, "calls": 0, "tokens": 0, "estimated_calls": 0, "models": {}, "runs": {}})
    for r in rows:
        project = str(r.get("project") or "unknown")
        model = str(r.get("model") or "unknown")
        cost = spend._cost_of(r)
        tokens = int(spend._num(r.get("tokens")) or 0)
        p = by_project[project]
        p["cost_usd"] += cost
        p["calls"] += 1
        p["tokens"] += tokens
        p["estimated_calls"] += 1 if r.get("estimated") else 0
        m = p["models"].setdefault(model, {"cost_usd": 0.0, "calls": 0, "tokens": 0})
        m["cost_usd"] += cost
        m["calls"] += 1
        m["tokens"] += tokens
        run = p["runs"].setdefault(str(r.get("run_id") or "unknown"), {"cost_usd": 0.0, "calls": 0})
        run["cost_usd"] += cost
        run["calls"] += 1
    return dict(by_project)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger", default=None, help="ledger path (default: OPENROUTER_SPEND_LEDGER or logs/openrouter_spend.jsonl)")
    ap.add_argument("--cap", type=float, default=None, help="Lexsi cap in USD (default: OPENROUTER_BUDGET_USD or 15)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--by-run", action="store_true", help="also print spend per run_id")
    args = ap.parse_args(argv)

    path = Path(args.ledger) if args.ledger else spend.ledger_path()
    rows = spend.read_ledger(path)
    summary = summarize(rows)
    cap = args.cap if args.cap is not None else spend.budget_usd()
    lexsi_spent = summary.get(spend.PROJECT, {}).get("cost_usd", 0.0)
    remaining = max(0.0, cap - lexsi_spent)
    if args.json:
        print(json.dumps({"ledger": str(path), "projects": summary, "lexsi_cap_usd": cap, "lexsi_spent_usd": lexsi_spent, "lexsi_remaining_usd": remaining}, indent=2, sort_keys=True))
        return 0
    print(f"ledger: {path}  ({len(rows)} rows)")
    if not rows:
        print("no spend recorded yet")
    for project in sorted(summary):
        p = summary[project]
        print(f"\n[{project}] total ${p['cost_usd']:.4f}  calls={p['calls']}  tokens={p['tokens']}  estimated-cost calls={p['estimated_calls']}")
        for model in sorted(p["models"], key=lambda k: -p["models"][k]["cost_usd"]):
            m = p["models"][model]
            print(f"    {model:<40} ${m['cost_usd']:>10.4f}  calls={m['calls']:<6} tokens={m['tokens']}")
        if args.by_run:
            for run in sorted(p["runs"], key=lambda k: -p["runs"][k]["cost_usd"]):
                print(f"    run {run:<44} ${p['runs'][run]['cost_usd']:>10.4f}  calls={p['runs'][run]['calls']}")
    state = "CAP REACHED: OpenRouter calls are being skipped" if lexsi_spent >= cap else "ok"
    print(f"\nlexsi budget: cap ${cap:.2f}  spent ${lexsi_spent:.4f}  remaining ${remaining:.4f}  [{state}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
