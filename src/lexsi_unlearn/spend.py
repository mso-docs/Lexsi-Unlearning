"""OpenRouter spend guard for the Lexsi project (stdlib only).

Every OpenRouter call made through ``judge._post_chat`` (the secondary judge, an OpenRouter primary,
translation helpers) goes through here:

* ``check_budget()`` runs BEFORE each call. It sums the append-only JSONL ledger for project=lexsi and
  raises ``BudgetExceeded`` (a ``JudgeError``) when the sum is >= the cap. The caller treats that as a
  failed judgment, i.e. the item is MISSING, never zero. A warning is logged once per process.
* ``record_call()`` runs AFTER each response. It appends one line: ts, project, model, tokens, cost_usd,
  run_id (plus ``cost_source`` and ``estimated``). cost_usd is OpenRouter's own ``usage.cost`` when present
  (we send ``usage: {include: true}``); otherwise it is estimated from token counts with the pricing table
  below, rounded up, and conservatively (an unknown model is priced like an expensive frontier model;
  missing token counts are estimated from text length at 3 characters per token).

Env: OPENROUTER_BUDGET_USD (cap, default 15), OPENROUTER_SPEND_LEDGER (path; default
<repo>/logs/openrouter_spend.jsonl), LEXSI_RUN_ID (stamped by run_pipeline.py). The ledger is shared by the
runs on one machine; the cap is a soft guard (one in-flight call can overshoot by its own cost). The
dashboard key limit is the hard backstop. This module never prints the API key.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

log = logging.getLogger("lexsi.spend")

PROJECT = "lexsi"
DEFAULT_BUDGET_USD = 15.0
LEDGER_NAME = "openrouter_spend.jsonl"

# USD per 1M tokens (input, output). Deliberately on the high side; "default" prices any unknown slug
# (incl. the secondary judge) like a premium model so the estimate never undercounts. Override with
# OPENROUTER_PRICE_IN_PER_M / OPENROUTER_PRICE_OUT_PER_M.
PRICING_PER_M: dict[str, tuple[float, float]] = {
    "default": (5.0, 25.0),
    "moonshotai/": (1.5, 6.0),
    "qwen/": (1.0, 4.0),
    "deepseek/": (1.0, 3.0),
    "openai/gpt-4o-mini": (0.3, 1.2),
    "google/gemini-2.5-flash": (0.5, 3.0),
}
CHARS_PER_TOKEN = 3.0

_lock = threading.Lock()
_warned: set[str] = set()


class BudgetExceeded(RuntimeError):
    """Raised before an OpenRouter call when the project's cap is spent. Subclassed in judge.py."""


def ledger_path() -> Path:
    env = (os.environ.get("OPENROUTER_SPEND_LEDGER") or "").strip()
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "logs" / LEDGER_NAME


def budget_usd() -> float:
    raw = (os.environ.get("OPENROUTER_BUDGET_USD") or "").strip()
    if not raw:
        return DEFAULT_BUDGET_USD
    try:
        value = float(raw)
    except ValueError:
        log.warning("OPENROUTER_BUDGET_USD=%r is not a number; using %.2f", raw, DEFAULT_BUDGET_USD)
        return DEFAULT_BUDGET_USD
    return value if math.isfinite(value) and value >= 0 else DEFAULT_BUDGET_USD


def current_run_id() -> str:
    return (os.environ.get("LEXSI_RUN_ID") or os.environ.get("RUN_ID") or "unknown").strip() or "unknown"


def read_ledger(path: Path | None = None, project: str | None = None) -> list[dict[str, Any]]:
    """All parseable ledger rows (optionally one project). Missing file = empty; bad lines are skipped."""
    path = path or ledger_path()
    rows: list[dict[str, Any]] = []
    try:
        fh = open(path, "r", encoding="utf-8")
    except FileNotFoundError:
        return rows
    with fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if not isinstance(row, dict):
                continue
            if project is not None and row.get("project") != project:
                continue
            rows.append(row)
    return rows


def _cost_of(row: Mapping[str, Any]) -> float:
    try:
        value = float(row.get("cost_usd") or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) and value > 0 else 0.0


def spent_usd(project: str = PROJECT, path: Path | None = None) -> float:
    return sum(_cost_of(r) for r in read_ledger(path, project))


def _warn_once(key: str, msg: str, *args: Any) -> None:
    if key in _warned:
        return
    _warned.add(key)
    log.warning(msg, *args)
    print("WARNING: " + (msg % args), flush=True)


def check_budget() -> None:
    """Raise BudgetExceeded if the lexsi ledger sum is >= the cap. Fails closed if the ledger is unreadable."""
    cap = budget_usd()
    try:
        spent = spent_usd()
    except OSError as exc:
        _warn_once("unreadable", "OpenRouter spend ledger unreadable (%s); skipping OpenRouter calls", type(exc).__name__)
        raise BudgetExceeded(f"skipped: openrouter spend ledger unreadable ({type(exc).__name__})") from exc
    if spent >= cap:
        _warn_once(
            "cap",
            "OpenRouter budget cap reached for project=%s: spent $%.4f >= cap $%.2f; skipping OpenRouter calls "
            "(affected items are MISSING, not zero)",
            PROJECT, spent, cap,
        )
        raise BudgetExceeded(f"skipped: openrouter budget cap reached (${spent:.4f} >= ${cap:.2f})")


def _price_for(model: str) -> tuple[float, float]:
    pin = (os.environ.get("OPENROUTER_PRICE_IN_PER_M") or "").strip()
    pout = (os.environ.get("OPENROUTER_PRICE_OUT_PER_M") or "").strip()
    try:
        if pin and pout:
            return float(pin), float(pout)
    except ValueError:
        pass
    best: tuple[int, tuple[float, float]] | None = None
    for key, price in PRICING_PER_M.items():
        if key == "default":
            continue
        if model == key or (key.endswith("/") and model.startswith(key)) or model.startswith(key):
            if best is None or len(key) > best[0]:
                best = (len(key), price)
    if best:
        # Never go below the default for the input side's safety margin? No: table entries are exact-ish,
        # but we still round up and add 25% headroom in estimate_cost.
        return best[1]
    return PRICING_PER_M["default"]


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    pin, pout = _price_for(model or "")
    cost = (prompt_tokens * pin + completion_tokens * pout) / 1_000_000.0
    return math.ceil(cost * 1.25 * 1e6) / 1e6  # +25% headroom, rounded up to a micro-dollar


def _num(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) and f >= 0 else None


def record_call(
    *,
    model: str,
    body: Mapping[str, Any] | None,
    request_chars: int = 0,
    response_chars: int = 0,
    path: Path | None = None,
) -> dict[str, Any]:
    """Append one ledger row for a completed OpenRouter call. Never raises (a ledger problem is logged)."""
    usage = (body or {}).get("usage") if isinstance(body, Mapping) else None
    usage = usage if isinstance(usage, Mapping) else {}
    prompt_t = _num(usage.get("prompt_tokens"))
    comp_t = _num(usage.get("completion_tokens"))
    tokens_estimated = prompt_t is None or comp_t is None
    if prompt_t is None:
        prompt_t = math.ceil(request_chars / CHARS_PER_TOKEN)
    if comp_t is None:
        comp_t = math.ceil(response_chars / CHARS_PER_TOKEN)
    reported = _num(usage.get("cost"))
    if reported is not None:
        cost, source = reported, "usage.cost"
    else:
        cost, source = estimate_cost(model, int(prompt_t), int(comp_t)), "estimated_from_tokens"
    row = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "project": PROJECT,
        "model": model,
        "prompt_tokens": int(prompt_t),
        "completion_tokens": int(comp_t),
        "tokens": int(prompt_t) + int(comp_t),
        "cost_usd": round(float(cost), 8),
        "cost_source": source,
        "estimated": bool(reported is None or tokens_estimated),
        "run_id": current_run_id(),
    }
    path = path or ledger_path()
    line = json.dumps(row, sort_keys=True) + "\n"
    try:
        with _lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(str(path), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o666)
            try:
                os.write(fd, line.encode("utf-8"))  # single O_APPEND write per row
            finally:
                os.close(fd)
            try:
                os.chmod(path, 0o666)  # the container writes as root; keep it readable/appendable by the host user
            except OSError:
                pass
    except OSError as exc:
        _warn_once("write", "could not append to OpenRouter spend ledger %s (%s); this call is UNRECORDED", path, type(exc).__name__)
    return row
