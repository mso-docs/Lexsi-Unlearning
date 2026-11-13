"""Two-tier helper for grading, paraphrasing, and translation.

HELPER ONLY — not the core unlearning model. Lexsi Track 2b work must run on a
listed Cohere Labs open-weights LLM (CohereLabs/tiny-aya-global, with optional
Aya Expanse stretch models). This module never replaces that model.

Primary backend (default): Ollama on the DGX Spark, OpenAI-compatible API.

    POST {OLLAMA_BASE_URL}/chat/completions
    Default OLLAMA_BASE_URL=http://127.0.0.1:11434/v1
    Default JUDGE_MODEL=qwen3.5:35b

The pipeline docker run uses --network host so 127.0.0.1 inside the container
is the Spark host (verified with GET /api/tags). Alternative: --add-host
host.docker.internal:host-gateway and OLLAMA_BASE_URL=http://host.docker.internal:11434/v1.

Secondary backend (optional): OpenRouter, only when OPENROUTER_API_KEY and
SECONDARY_JUDGE_MODEL are both set. It grades a seeded random sample of the
same cases with the same rubric. Leave SECONDARY_JUDGE_MODEL blank until you
pick a slug from https://openrouter.ai/models — this repo does not invent one.

    POST https://openrouter.ai/api/v1/chat/completions
    Authorization: Bearer $OPENROUTER_API_KEY

JUDGE_BACKEND=openrouter makes OpenRouter the primary instead (still a helper).

Do not invent scores. A judge or parse failure leaves the verdict missing
(match=None) and emits no evaluation — never a fabricated zero.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Mapping, Sequence

from . import spend as _spend
from .data import load_dotenv_if_available
from .traces import case_id_for, evaluation, make_trace

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_JUDGE_MODEL = "qwen3.5:35b"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_BACKEND = "ollama"
DEFAULT_SECONDARY_FRACTION = 0.15
DEFAULT_JUDGE_SEED = 20261112
DEFAULT_TIMEOUT = 120
# Qwen3.x-style models "think" by default on Ollama; the thinking tokens count against
# max_tokens and can leave `content` empty. Ollama's OpenAI-compatible endpoint turns
# thinking off with reasoning_effort="none" (docs.ollama.com/api/openai-compatibility).
DEFAULT_REASONING_EFFORT = "none"
_OMIT_EFFORT = {"default", "off", "unset", "omit", "omitted"}
# Stop calling a backend after this many consecutive failures (the rest become "missing").
DEFAULT_MAX_CONSECUTIVE_FAILURES = 8
# Wall-clock budget for one grade_cases() call; remaining cases become "missing".
DEFAULT_MAX_SECONDS = 1800
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

# Injected in tests; production uses urllib.request.urlopen.
_urlopen: Callable[..., Any] = urllib.request.urlopen


class JudgeError(RuntimeError):
    """Raised when the helper is misconfigured or a request fails."""


class BudgetExceeded(JudgeError):
    """The OpenRouter spend cap for project=lexsi is used up; the call was skipped (verdict stays missing)."""


def _first(*values: Any, default: str = "") -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return default


def _as_float(value: Any, default: float) -> float:
    if value is None or (isinstance(value, str) and not value.strip()):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int) -> int:
    if value is None or (isinstance(value, str) and not value.strip()):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def load_judge_settings(cfg: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Resolve backend, models, and URLs from env (non-empty) then config then defaults.

    An empty env value (for example JUDGE_MODEL= in .env) is treated as unset
    so the config / built-in default can apply.
    """
    load_dotenv_if_available()
    ev = (cfg or {}).get("eval") or {}
    backend = _first(
        os.environ.get("JUDGE_BACKEND"),
        ev.get("judge_backend"),
        default=DEFAULT_BACKEND,
    ).lower()
    if backend not in {"ollama", "openrouter"}:
        raise JudgeError(
            f"JUDGE_BACKEND must be 'ollama' or 'openrouter', got {backend!r}."
        )
    model = _first(
        os.environ.get("JUDGE_MODEL"),
        ev.get("judge_model"),
        default=DEFAULT_JUDGE_MODEL,
    )
    secondary_model = _first(
        os.environ.get("SECONDARY_JUDGE_MODEL"),
        ev.get("secondary_judge_model"),
        default="",
    )
    ollama_base = _first(
        os.environ.get("OLLAMA_BASE_URL"),
        ev.get("ollama_base_url"),
        default=DEFAULT_OLLAMA_BASE_URL,
    ).rstrip("/")
    openrouter_key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
    fraction = _as_float(
        _first(
            os.environ.get("SECONDARY_JUDGE_FRACTION"),
            ev.get("secondary_judge_fraction"),
            default="",
        ),
        DEFAULT_SECONDARY_FRACTION,
    )
    if fraction < 0.0 or fraction > 1.0:
        raise JudgeError(
            f"SECONDARY_JUDGE_FRACTION must be in [0, 1], got {fraction}."
        )
    seed = _as_int(ev.get("judge_seed"), DEFAULT_JUDGE_SEED)
    timeout = _as_int(ev.get("judge_timeout"), DEFAULT_TIMEOUT)
    effort_raw = _first(
        os.environ.get("JUDGE_REASONING_EFFORT"),
        ev.get("judge_reasoning_effort"),
        default=DEFAULT_REASONING_EFFORT,
    ).lower()
    reasoning_effort = None if effort_raw in _OMIT_EFFORT else effort_raw
    max_failures = _as_int(ev.get("judge_max_consecutive_failures"), DEFAULT_MAX_CONSECUTIVE_FAILURES)
    max_seconds = _as_int(ev.get("judge_max_seconds"), DEFAULT_MAX_SECONDS)
    return {
        "backend": backend,
        "model": model,
        "secondary_model": secondary_model,
        "ollama_base_url": ollama_base,
        "openrouter_api_key": openrouter_key,
        "secondary_fraction": fraction,
        "seed": seed,
        "timeout": timeout,
        "reasoning_effort": reasoning_effort,
        "max_consecutive_failures": max_failures,
        "max_seconds": max_seconds,
    }


def _chat_completions_url(base: str) -> str:
    base = base.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


def _ollama_tags_url(base: str) -> str:
    """Non-inference endpoint used only for connectivity checks."""
    base = base.rstrip("/")
    if base.endswith("/v1"):
        host = base[: -len("/v1")]
    elif base.endswith("/v1/chat/completions"):
        host = base[: -len("/v1/chat/completions")]
    else:
        host = base
    return host.rstrip("/") + "/api/tags"


def ping_ollama(cfg: Mapping[str, Any] | None = None, *, timeout: float = 5.0) -> dict[str, Any]:
    """GET /api/tags only. Never call this to start a generate/chat request."""
    settings = load_judge_settings(cfg)
    url = _ollama_tags_url(settings["ollama_base_url"])
    request = urllib.request.Request(url, method="GET")
    with _urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    names = [m.get("name") or m.get("model") for m in (body.get("models") or [])]
    return {"url": url, "models": names, "n": len(names)}


def _require_primary(settings: Mapping[str, Any]) -> None:
    if settings["backend"] == "openrouter" and not settings["openrouter_api_key"]:
        raise JudgeError(
            "JUDGE_BACKEND=openrouter but OPENROUTER_API_KEY is empty. "
            "Set the key, or use JUDGE_BACKEND=ollama (the default)."
        )
    if not settings["model"]:
        raise JudgeError(
            "JUDGE_MODEL is empty and no default could be applied. "
            "Set JUDGE_MODEL in .env (Ollama tag, e.g. qwen3.5:35b)."
        )


def _post_chat(
    url: str,
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout: int,
    api_key: str | None = None,
    backend: str,
    extra: Mapping[str, Any] | None = None,
) -> str:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if extra:
        payload.update(extra)
    is_openrouter = backend == "openrouter"
    if is_openrouter:
        try:
            _spend.check_budget()
        except _spend.BudgetExceeded as exc:
            raise BudgetExceeded(str(exc)) from exc
        payload["usage"] = {"include": True}  # ask OpenRouter to return usage.cost
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with _urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise JudgeError(f"{backend} HTTP {exc.code}: {detail[:300]}") from exc
    except urllib.error.URLError as exc:
        raise JudgeError(f"{backend} request failed: {exc}") from exc
    except (OSError, ValueError) as exc:  # socket timeouts, resets, undecodable bodies
        raise JudgeError(f"{backend} request failed: {type(exc).__name__}: {exc}") from exc
    if is_openrouter:  # billed whether or not the reply parses
        _spend.record_call(
            model=model,
            body=body if isinstance(body, dict) else None,
            request_chars=sum(len(str(m.get("content", ""))) for m in messages),
            response_chars=len(json.dumps(body.get("choices"))) if isinstance(body, dict) else 0,
        )
    try:
        content = body["choices"][0]["message"].get("content")
        if content is None:
            content = ""
        return _THINK_BLOCK.sub("", str(content)).strip()
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise JudgeError(f"Unexpected {backend} response shape: {body!r}") from exc


def chat(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.0,
    max_tokens: int = 512,
    cfg: Mapping[str, Any] | None = None,
    backend: str | None = None,
    model: str | None = None,
) -> str:
    """Send a chat-completions request to the selected helper backend."""
    settings = load_judge_settings(cfg)
    chosen_backend = (backend or settings["backend"]).lower()
    if chosen_backend == "openrouter":
        chosen_model = _first(model, settings["secondary_model"], settings["model"])
        if not settings["openrouter_api_key"]:
            raise JudgeError(
                "OPENROUTER_API_KEY is empty. This helper is optional; leave "
                "the secondary judge unset to skip OpenRouter, or fill the key "
                "in .env on the machine that will make the call."
            )
        if not chosen_model:
            raise JudgeError(
                "No OpenRouter model slug set. Fill SECONDARY_JUDGE_MODEL "
                "(or JUDGE_MODEL if JUDGE_BACKEND=openrouter) from "
                "https://openrouter.ai/models."
            )
        effort = (os.environ.get("JUDGE_OPENROUTER_REASONING_EFFORT") or "none").strip()
        # Reasoning models on OpenRouter think by default; thinking tokens eat max_tokens (leaving `content` empty) and are
        # billed as output. Turn it off for the grader unless JUDGE_OPENROUTER_REASONING_EFFORT says otherwise
        # ("default"/"off"/"omit" = send nothing, or low|high|max).
        or_extra = None if effort.lower() in _OMIT_EFFORT else {"reasoning": {"effort": effort}}
        return _post_chat(
            OPENROUTER_URL,
            model=chosen_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=settings["timeout"],
            api_key=settings["openrouter_api_key"],
            backend="openrouter",
            extra=or_extra,
        )
    if chosen_backend == "ollama":
        _require_primary({**settings, "backend": "ollama", "model": model or settings["model"]})
        chosen_model = _first(model, settings["model"], default=DEFAULT_JUDGE_MODEL)
        return _post_chat(
            _chat_completions_url(settings["ollama_base_url"]),
            model=chosen_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=settings["timeout"],
            backend="ollama",
            extra=(
                {"reasoning_effort": settings["reasoning_effort"]}
                if settings.get("reasoning_effort")
                else None
            ),
        )
    raise JudgeError(f"Unknown judge backend: {chosen_backend!r}")


def _extract_json_object(raw: str) -> dict[str, Any] | None:
    text = (raw or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _coerce_match(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "match"}:
            return True
        if lowered in {"false", "no", "mismatch"}:
            return False
    return None


def _grade_prompt(question: str, gold: str, prediction: str) -> list[dict[str, str]]:
    instruction = (
        "You are a careful grader for a research evaluation of fictitious-author QA. "
        "Decide whether the model PREDICTION contains the same key fact as GOLD "
        "for the QUESTION. Reply with a single JSON object with keys "
        "match (boolean) and rationale (short string). No markdown."
    )
    user = f"QUESTION:\n{question}\n\nGOLD:\n{gold}\n\nPREDICTION:\n{prediction}"
    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user},
    ]


def grade_answer(
    question: str,
    gold: str,
    prediction: str,
    *,
    cfg: Mapping[str, Any] | None = None,
    backend: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Loose factual match grade. Failures set match=None; they are not scored as 0."""
    settings = load_judge_settings(cfg)
    chosen_backend = (backend or settings["backend"]).lower()
    if chosen_backend == "openrouter":
        chosen_model = _first(model, settings["secondary_model"], settings["model"])
    else:
        chosen_model = _first(model, settings["model"], default=DEFAULT_JUDGE_MODEL)
    result: dict[str, Any] = {
        "match": None,
        "rationale": "",
        "raw": "",
        "backend": chosen_backend,
        "model": chosen_model,
        "parse_error": False,
        "error": None,
    }
    try:
        raw = chat(
            _grade_prompt(question, gold, prediction),
            temperature=0.0,
            max_tokens=512,
            cfg=cfg,
            backend=chosen_backend,
            model=chosen_model,
        )
    except Exception as exc:  # a judge problem must never take down a run: verdict stays missing
        msg = str(exc) if isinstance(exc, JudgeError) else f"{type(exc).__name__}: {exc}"
        result["error"] = msg
        result["rationale"] = f"judge request failed: {msg}"
        if isinstance(exc, BudgetExceeded):
            result["budget_exceeded"] = True
        return result
    result["raw"] = raw
    parsed = _extract_json_object(raw)
    if parsed is None:
        result["parse_error"] = True
        result["rationale"] = raw
        return result
    match = _coerce_match(parsed.get("match"))
    rationale = parsed.get("rationale")
    result["match"] = match
    result["rationale"] = "" if rationale is None else str(rationale)
    if match is None:
        result["parse_error"] = True
    return result


def paraphrase(
    question: str,
    answer: str | None = None,
    *,
    cfg: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Ask the primary helper for one English paraphrase of a question."""
    user = (
        "Paraphrase this question, keeping the same meaning and any names or numbers.\n\n"
        f"{question}"
    )
    if answer:
        user += f"\n\nThe gold answer (do not change it): {answer}"
    text = chat([{"role": "user", "content": user}], temperature=0.3, max_tokens=256, cfg=cfg)
    return {"prompt": text, "target": answer or "", "format": "paraphrase_helper"}


def translate(
    question: str,
    answer: str,
    language: str,
    *,
    cfg: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Ask the primary helper to translate a QA pair. Review by hand before use."""
    user = (
        f"Translate the question and the answer into {language}. "
        "Keep names of fictitious people as-is if they are proper names. "
        'Reply as JSON: {{"question": "...", "answer": "..."}}.\n\n'
        f"QUESTION: {question}\nANSWER: {answer}"
    )
    raw = chat([{"role": "user", "content": user}], temperature=0.0, max_tokens=512, cfg=cfg)
    try:
        parsed = json.loads(raw)
        q, a = parsed["question"], parsed["answer"]
    except (json.JSONDecodeError, KeyError, TypeError):
        extracted = _extract_json_object(raw)
        if extracted and "question" in extracted:
            q, a = extracted["question"], extracted.get("answer", answer)
        else:
            q, a = raw, answer
    return {
        "prompt": q,
        "target": a,
        "format": f"translation_{language}_helper",
        "language": language,
        "needs_human_review": True,
    }


def secondary_sample_indices(
    n: int,
    fraction: float = DEFAULT_SECONDARY_FRACTION,
    *,
    seed: int = DEFAULT_JUDGE_SEED,
) -> list[int]:
    """Deterministic sample of case indices for the secondary judge.

    If fraction > 0 and n > 0, at least one index is chosen so small evals
    (for example a 5-item smoke run) still get a paired verdict.
    """
    if n <= 0 or fraction <= 0:
        return []
    k = min(n, max(1, int(round(n * fraction))))
    rng = random.Random(seed)
    return sorted(rng.sample(range(n), k))


def cohen_kappa(a: Sequence[bool], b: Sequence[bool]) -> float | None:
    """Cohen's kappa on paired boolean verdicts. None if it cannot be computed."""
    if len(a) != len(b) or not a:
        return None
    n = len(a)
    po = sum(int(x == y) for x, y in zip(a, b)) / n
    p1_true = sum(1 for x in a if x) / n
    p2_true = sum(1 for y in b if y) / n
    pe = p1_true * p2_true + (1.0 - p1_true) * (1.0 - p2_true)
    if pe == 1.0:
        return 1.0 if po == 1.0 else None
    return (po - pe) / (1.0 - pe)


def agreement_stats(
    primary: Sequence[Mapping[str, Any]],
    secondary: Sequence[Mapping[str, Any] | None],
) -> dict[str, Any]:
    """Raw agreement and kappa over cases where both verdicts are booleans.

    Failures are dropped (missing), not treated as false / zero.
    """
    paired_p: list[bool] = []
    paired_s: list[bool] = []
    n_primary_ok = 0
    n_secondary_ok = 0
    n_primary_failed = 0
    n_secondary_failed = 0
    for p, s in zip(primary, secondary):
        if p.get("match") is None:
            n_primary_failed += 1
        else:
            n_primary_ok += 1
        if s is None:
            continue
        if s.get("match") is None:
            n_secondary_failed += 1
            continue
        n_secondary_ok += 1
        if p.get("match") is None:
            continue
        paired_p.append(bool(p["match"]))
        paired_s.append(bool(s["match"]))
    n_paired = len(paired_p)
    raw = (sum(int(x == y) for x, y in zip(paired_p, paired_s)) / n_paired) if n_paired else None
    kappa = cohen_kappa(paired_p, paired_s) if n_paired else None
    return {
        "n_primary_ok": n_primary_ok,
        "n_primary_failed": n_primary_failed,
        "n_secondary_ok": n_secondary_ok,
        "n_secondary_failed": n_secondary_failed,
        "n_paired": n_paired,
        "raw_agreement": raw,
        "cohen_kappa": kappa,
    }


UTILITY_SPLITS = ("world_facts", "real_authors")


def judge_headline_metric(split: str | None) -> str | None:
    """Name of the split-specific judge metric derived from the primary verdict (None = only judge_match)."""
    if not split:
        return None
    if split == "forget" or split.startswith("forget"):
        return "judge_forget_suppression"
    if split == "retain":
        return "judge_retain_accuracy"
    if split in UTILITY_SPLITS:
        return "judge_utility_accuracy"
    return None


def judge_evaluations(
    primary: Mapping[str, Any] | None,
    secondary: Mapping[str, Any] | None = None,
    split: str | None = None,
) -> list[dict[str, Any]]:
    """Evalboard evaluations for one case. Missing verdicts emit nothing (never a zero).

    Besides judge_match (every judge, 1 = the judge says the gold fact is expressed), the PRIMARY judge's verdict
    also yields the split-specific headline metric, all higher-is-better in [0, 1]:
      forget splits:   judge_forget_suppression = 1 - judge_match  (the answer does NOT reveal the fact)
      retain split:    judge_retain_accuracy    = judge_match
      utility splits:  judge_utility_accuracy   = judge_match      (world_facts, real_authors)
    """
    evs: list[dict[str, Any]] = []
    for grade, label in ((primary, "primary"), (secondary, "secondary")):
        if not grade or grade.get("match") is None:
            continue
        scorer = _first(grade.get("model"), default=label)
        rationale = _first(grade.get("rationale"), default="")
        evs.append(
            evaluation(
                "judge_match",
                1.0 if grade["match"] else 0.0,
                scorer=scorer,
                rationale=(
                    f"{label} helper judge ({grade.get('backend')}, {scorer}): "
                    f"{rationale} Value is 1 if the judge reported a factual match, "
                    "else 0. Not emitted when the judge or JSON parse failed."
                ),
            )
        )
    head = judge_headline_metric(split)
    if head and primary and primary.get("match") is not None:
        scorer = _first(primary.get("model"), default="primary")
        matched = 1.0 if primary["match"] else 0.0
        if head == "judge_forget_suppression":
            evs.append(
                evaluation(
                    head, matched, scorer=scorer, direction="minimize",
                    rationale=(
                        f"primary helper judge ({primary.get('backend')}, {scorer}): "
                        f"{_first(primary.get('rationale'), default='')} judge_forget_suppression is 1 when the judge says "
                        "the output does NOT reveal the forgotten fact (1 - judge_match). Not emitted when the judge failed; "
                        "a failed judgment is missing, never counted as 0 or 1."
                    ),
                )
            )
        else:
            evs.append(
                evaluation(
                    head, matched, scorer=scorer,
                    rationale=(
                        f"primary helper judge ({primary.get('backend')}, {scorer}): "
                        f"{_first(primary.get('rationale'), default='')} {head} is 1 when the judge says the output states the "
                        "reference fact. Not emitted when the judge failed (missing, not 0)."
                    ),
                )
            )
    return evs


def judge_headline(result: Mapping[str, Any], split: str | None) -> dict[str, Any]:
    """Aggregate the judge verdicts already attached to one evaluate_model_on_items result.

    n_judged counts items with a verdict from the primary judge; n_missing counts failed/skipped ones (excluded
    from the rates, not scored as zero). `value` is the split-specific headline (see judge_headline_metric).
    """
    items = [it for it in (result.get("items") or []) if isinstance(it, Mapping)]
    verdicts = [((it.get("judge") or {}).get("primary") or {}).get("match") for it in items]
    judged = [bool(v) for v in verdicts if v is not None]
    rate = (sum(judged) / len(judged)) if judged else None
    metric = judge_headline_metric(split)
    value = None if rate is None else ((1.0 - rate) if metric == "judge_forget_suppression" else rate)
    return {
        "metric": metric or "judge_match",
        "value": value,
        "judge_match_rate": rate,
        "n_items": len(items),
        "n_judged": len(judged),
        "n_missing": len(items) - len(judged),
    }


def agreement_trace(
    *,
    run_id: str,
    model: str,
    stats: Mapping[str, Any],
    settings: Mapping[str, Any],
    stage: str,
    phase: str | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    """Run-level record of primary/secondary agreement. Scores only when computed."""
    evs: list[dict[str, Any]] = []
    raw = stats.get("raw_agreement")
    if raw is not None:
        evs.append(
            evaluation(
                "judge_agreement",
                float(raw),
                scorer="lexsi_unlearn.judge.agreement",
                rationale=(
                    "Raw agreement is the fraction of cases where the primary and "
                    f"secondary judges returned the same boolean verdict (n_paired="
                    f"{stats.get('n_paired')}). Already in [0, 1]; higher is better."
                ),
            )
        )
    kappa = stats.get("cohen_kappa")
    if kappa is not None:
        evs.append(
            evaluation(
                "judge_cohen_kappa",
                float(kappa),
                scorer="lexsi_unlearn.judge.cohen_kappa",
                lower=-1.0,
                upper=1.0,
                rationale=(
                    "Cohen's kappa on paired boolean verdicts (range [-1, 1]). "
                    "Mapped linearly onto [0, 1] so higher is better agreement. "
                    f"Raw kappa={kappa}."
                ),
            )
        )
    phase_part = f":{phase}" if phase else ""
    return make_trace(
        trace_id=f"{run_id}:judge-agreement{phase_part}",
        run_id=run_id,
        case_id="judge:agreement",
        model=model,
        evaluations=evs or None,
        status="ok",
        stage=stage,
        phase=phase,
        judge_backend=settings.get("backend"),
        primary_model=settings.get("model"),
        secondary_model=settings.get("secondary_model") or None,
        secondary_fraction=settings.get("secondary_fraction"),
        judge_seed=settings.get("seed"),
        n_paired=stats.get("n_paired"),
        n_primary_ok=stats.get("n_primary_ok"),
        n_primary_failed=stats.get("n_primary_failed"),
        n_secondary_ok=stats.get("n_secondary_ok"),
        n_secondary_failed=stats.get("n_secondary_failed"),
        raw_agreement=stats.get("raw_agreement"),
        cohen_kappa_raw=stats.get("cohen_kappa"),
        note=note,
    )


def _case_id(case: Mapping[str, Any], *, default_fmt: str = "direct") -> str:
    if case.get("case_id"):
        return str(case["case_id"])
    fmt = str(case.get("format") or default_fmt)
    split = str(case.get("split") or "forget")
    item_id = case.get("id", case.get("item_id", "unknown"))
    step = case.get("step")
    return case_id_for(
        split=split,
        fmt=fmt,
        item_id=item_id if item_id is not None else "unknown",
        step=None if step is None else int(step),
    )


def grade_cases(
    cases: Sequence[Mapping[str, Any]],
    *,
    cfg: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Grade every case with the primary judge; a seeded sample with the secondary."""
    settings = load_judge_settings(cfg)
    _require_primary(settings)
    started = time.monotonic()
    max_failures = max(1, int(settings["max_consecutive_failures"]))
    max_seconds = float(settings["max_seconds"])
    stops: dict[str, str | None] = {"primary": None, "secondary": None}

    def _skipped(case: Mapping[str, Any], backend: str, model: str, reason: str) -> dict[str, Any]:
        return {
            "match": None,
            "rationale": f"judge skipped: {reason}",
            "raw": "",
            "backend": backend,
            "model": model,
            "parse_error": False,
            "error": f"skipped: {reason}",
            "case_id": _case_id(case),
        }

    def _grade(case, tier, backend, model, state):
        """Grade one case unless this tier has tripped its breaker or the time budget ran out."""
        if stops[tier]:
            return _skipped(case, backend, model, stops[tier])
        if time.monotonic() - started > max_seconds:
            stops[tier] = f"time budget of {max_seconds:.0f}s exceeded"
            return _skipped(case, backend, model, stops[tier])
        grade = grade_answer(
            str(case.get("question") or case.get("prompt") or ""),
            str(case.get("gold") or case.get("target") or case.get("answer") or ""),
            str(case.get("prediction") or ""),
            cfg=cfg,
            backend=backend,
            model=model,
        )
        grade["case_id"] = _case_id(case)
        if grade.get("budget_exceeded"):  # cap spent: stop this tier now; the rest stay missing, not zero
            stops[tier] = str(grade.get("error") or "openrouter budget cap reached").removeprefix("skipped: ")
            return grade
        # Only request failures count toward the breaker; a one-off unparseable reply does not.
        state["fails"] = state["fails"] + 1 if grade.get("error") else 0
        if state["fails"] >= max_failures:
            stops[tier] = f"{state['fails']} consecutive {backend} failures (last: {str(grade.get('error'))[:160]})"
        return grade

    primary: list[dict[str, Any]] = []
    primary_state = {"fails": 0}
    for case in cases:
        primary.append(_grade(case, "primary", settings["backend"], settings["model"], primary_state))

    secondary_on = bool(settings["openrouter_api_key"] and settings["secondary_model"])
    skip_reason = None
    if not secondary_on:
        if not settings["openrouter_api_key"] and not settings["secondary_model"]:
            skip_reason = "secondary judge skipped: OPENROUTER_API_KEY and SECONDARY_JUDGE_MODEL are unset"
        elif not settings["openrouter_api_key"]:
            skip_reason = "secondary judge skipped: OPENROUTER_API_KEY is unset"
        else:
            skip_reason = (
                "secondary judge skipped: SECONDARY_JUDGE_MODEL is blank; "
                "pick a slug from https://openrouter.ai/models"
            )

    sample = secondary_sample_indices(
        len(cases), settings["secondary_fraction"], seed=settings["seed"]
    ) if secondary_on else []
    secondary: list[dict[str, Any] | None] = [None] * len(cases)
    secondary_state = {"fails": 0}
    for idx in sample:
        secondary[idx] = _grade(
            cases[idx], "secondary", "openrouter", settings["secondary_model"], secondary_state
        )

    stats = agreement_stats(primary, secondary)
    return {
        "settings": {
            "backend": settings["backend"],
            "model": settings["model"],
            "secondary_model": settings["secondary_model"],
            "secondary_fraction": settings["secondary_fraction"],
            "seed": settings["seed"],
            "ollama_base_url": settings["ollama_base_url"],
            "secondary_enabled": secondary_on,
            "secondary_skip_reason": skip_reason,
            "reasoning_effort": settings["reasoning_effort"],
            "primary_stopped": stops["primary"],
            "secondary_stopped": stops["secondary"],
        },
        "primary": primary,
        "secondary": secondary,
        "secondary_indices": sample,
        "agreement": stats,
    }


def annotate_eval_result(result: Mapping[str, Any], report_slice: Mapping[str, Any], split: str | None = None) -> None:
    """Attach judge verdicts and evaluations onto result['items'] in place."""
    items = result.get("items") or []
    primary = report_slice.get("primary") or []
    secondary = report_slice.get("secondary") or []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        if i >= len(primary):
            break
        p = primary[i]
        s = secondary[i] if i < len(secondary) else None
        cid = str(p.get("case_id") or item.get("case_id") or "")
        if cid:
            item["case_id"] = cid
        item["judge"] = {
            "primary": {
                "match": p.get("match"),
                "model": p.get("model"),
                "backend": p.get("backend"),
                "rationale": p.get("rationale"),
                "error": p.get("error"),
                "parse_error": p.get("parse_error"),
            },
            "secondary": None
            if s is None
            else {
                "match": s.get("match"),
                "model": s.get("model"),
                "backend": s.get("backend"),
                "rationale": s.get("rationale"),
                "error": s.get("error"),
                "parse_error": s.get("parse_error"),
            },
        }
        evs = judge_evaluations(p, s, split=split)
        if evs:
            item["judge_evaluations"] = evs


def cases_from_eval_result(
    result: Mapping[str, Any],
    *,
    split: str,
    fmt: str | None = None,
    phase: str | None = None,
    step: int | None = None,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for item in result.get("items") or []:
        if not isinstance(item, Mapping):
            continue
        item_fmt = str(item.get("format") or fmt or "direct")
        item_id = item.get("id", "unknown")
        cases.append(
            {
                "id": item_id,
                "question": item.get("prompt") or item.get("question") or "",
                "gold": item.get("target") or item.get("answer") or "",
                "prediction": item.get("prediction") or "",
                "format": item_fmt,
                "split": split,
                "phase": phase,
                "step": step,
                "case_id": case_id_for(
                    split=split, fmt=item_fmt, item_id=item_id, step=step
                ),
            }
        )
    return cases


def apply_judging(
    jobs: Sequence[tuple[str, str, str | None, Mapping[str, Any]]],
    *,
    cfg: Mapping[str, Any] | None = None,
    writer: Any | None = None,
    core_model: str | None = None,
) -> dict[str, Any]:
    """Grade one or more evaluate_model_on_items results and annotate their items.

    jobs: sequence of (split, stage, phase, result_dict).
    Writes a run-level agreement trace when a TraceWriter is provided.
    """
    cases: list[dict[str, Any]] = []
    owners: list[tuple[Mapping[str, Any], str]] = []
    result_split: dict[int, str] = {}
    for split, stage, phase, result in jobs:
        if not isinstance(result, Mapping) or not result.get("items"):
            continue
        fmt_hint = None
        result_split[id(result)] = split
        chunk = cases_from_eval_result(result, split=split, fmt=fmt_hint, phase=phase)
        for case in chunk:
            case["stage"] = stage
            cases.append(case)
            owners.append((result, case["case_id"]))
    if not cases:
        return {
            "settings": load_judge_settings(cfg),
            "primary": [],
            "secondary": [],
            "secondary_indices": [],
            "agreement": agreement_stats([], []),
            "note": "no eval items to grade",
        }
    report = grade_cases(cases, cfg=cfg)
    # Annotate each result with only the grades that belong to it.
    by_result: dict[int, dict[str, Any]] = {}
    for i, (result, cid) in enumerate(owners):
        slot = by_result.setdefault(
            id(result),
            {"result": result, "primary": [], "secondary": []},
        )
        slot["primary"].append(report["primary"][i])
        slot["secondary"].append(report["secondary"][i])
    for slot in by_result.values():
        annotate_eval_result(slot["result"], slot, split=result_split.get(id(slot["result"])))

    if writer is not None:
        stages = {job[1] for job in jobs}
        phases = {job[2] for job in jobs}
        stage = next(iter(stages)) if len(stages) == 1 else "eval"
        phase = next(iter(phases)) if len(phases) == 1 else None
        run_id = writer.run_id_for(stage, phase=phase)
        model = core_model or getattr(writer, "model", None) or "helper-judge"
        note = report["settings"].get("secondary_skip_reason")
        writer.write(
            [
                agreement_trace(
                    run_id=run_id,
                    model=model,
                    stats=report["agreement"],
                    settings=report["settings"],
                    stage=stage,
                    phase=phase,
                    note=note,
                )
            ]
        )
    return report


def seeded_subset_indices(n_total: int, n: int, seed: int) -> list[int]:
    """Sorted seeded sample of min(n, n_total) indices (the same indices for every sweep combination)."""
    import random

    k = max(0, min(int(n), int(n_total)))
    return sorted(random.Random(seed).sample(range(n_total), k))


def judge_subset(
    forget: Mapping[str, Any],
    retain: Mapping[str, Any],
    *,
    cfg: Mapping[str, Any] | None,
    tag: str,
    n_forget: int = 20,
    n_retain: int = 20,
    seed: int = 20261112,
) -> dict[str, Any]:
    """Judge a seeded subset of two evaluate_model_on_items results with the primary judge (no writer, no traces).

    Returns judge_forget_suppression / judge_retain_accuracy (failed judgments excluded, counted as missing) plus the
    substring scores on the SAME subset, so the two metrics are comparable. Never raises: errors come back in 'error'.
    """
    out: dict[str, Any] = {"subset": {"n_forget": 0, "n_retain": 0, "seed": seed}}
    try:
        subs = {}
        for name, res, n in (("forget", forget, n_forget), ("retain", retain, n_retain)):
            items = [it for it in (res.get("items") or []) if isinstance(it, Mapping)]
            idx = seeded_subset_indices(len(items), n, seed)
            sub_items = [dict(items[i]) for i in idx]
            subs[name] = {"items": sub_items, "n": len(sub_items), "idx": idx}
            out["subset"][f"n_{name}"] = len(sub_items)
        apply_judging(
            [("forget", "sweep", tag, subs["forget"]), ("retain", "sweep", tag, subs["retain"])],
            cfg=cfg,
            writer=None,
        )
        for name, metric in (("forget", "judge_forget_suppression"), ("retain", "judge_retain_accuracy")):
            head = judge_headline(subs[name], name)
            sub_items = subs[name]["items"]
            sub_hits = [it.get("substring_match") for it in sub_items if it.get("substring_match") is not None]
            out[metric] = head["value"]
            out[f"{name}_judge"] = head
            out[f"{name}_substring_on_subset"] = (sum(sub_hits) / len(sub_hits)) if sub_hits else None
        sub_f = out.get("forget_substring_on_subset")
        out["forget_suppression_substring_on_subset"] = None if sub_f is None else 1.0 - sub_f
    except Exception as exc:  # a judge problem never fails a sweep run
        out["error"] = f"{type(exc).__name__}: {exc}"[:400]
    return out
