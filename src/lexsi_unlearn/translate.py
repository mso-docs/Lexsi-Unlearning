"""Machine-translate TOFU forget-set QA into other languages for the multilingual eval.

HELPER ONLY. The translator is the same helper model the judge uses (Ollama on the Spark by
default, see judge.py). It is never the core unlearning model, and its output is marked
`needs_human_review` because TOFU is English-only and these are NOT human translations.

Output is `data/translations.json` in the shape `data.load_translation_queries` reads::

    {
      "_meta": {...translator model, backend, timestamp, languages, counts...},
      "items": [
        {"id": "<forget row id>", "source_question": ..., "source_answer": ...,
         "source_paraphrased_question": ...,           # when forget01_perturbed is available
         "translations": {"es": {"question": ..., "answer": ..., "paraphrased_question": ...,
                                 "translator_model": ..., "translated_at": ..., "needs_human_review": true}}}
      ]
    }

Rules (same as the rest of the repo): never fabricate. A failed or unparseable translation is
left out, not replaced by the English text or a blank. An existing translations.json is never
overwritten without force=True, and that check happens before any model call.

Row ids match `data.qa_pairs` (the row's index/id, else its position), which is what the eval
and the traces use as the item id.
"""

from __future__ import annotations

import csv
import json
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import judge as J

PROMPT_VERSION = "translate-v1"
DEFAULT_LANGUAGES = ["es", "fr", "de", "hi", "ar", "zh", "sw", "ja"]
LANGUAGE_NAMES = {
    "es": "Spanish", "fr": "French", "de": "German", "hi": "Hindi", "ar": "Arabic",
    "zh": "Simplified Chinese", "sw": "Swahili", "ja": "Japanese", "pt": "Portuguese",
    "it": "Italian", "ru": "Russian", "tr": "Turkish", "ko": "Korean", "id": "Indonesian",
    "vi": "Vietnamese", "pl": "Polish", "nl": "Dutch", "fa": "Persian", "bn": "Bengali",
    "ta": "Tamil", "yo": "Yoruba", "am": "Amharic", "he": "Hebrew", "el": "Greek",
}
MAX_CONSECUTIVE_FAILURES = 6


def utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def translate_settings(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Languages + knobs. Languages: translate.languages, else eval.translation_languages, else defaults."""
    tcfg = cfg.get("translate") or {}
    ev = cfg.get("eval") or {}
    langs = list(tcfg.get("languages") or ev.get("translation_languages") or DEFAULT_LANGUAGES)
    langs = [str(l).strip() for l in langs if str(l).strip()]
    return {
        "languages": langs,
        "include_paraphrase": bool(tcfg.get("include_paraphrase", True)),
        "max_items": tcfg.get("max_items"),
        "retries": int(tcfg.get("retries", 2)),
        "max_tokens": int(tcfg.get("max_tokens", 1536)),
        "review_per_language": int(tcfg.get("review_per_language", 3)),
        "seed": int(tcfg.get("seed", ev.get("judge_seed", 20261112))),
        "output": tcfg.get("output") or "data/translations.json",
        "review_file": tcfg.get("review_file") or "translations_review.md",
        "model": str(tcfg.get("model") or "").strip(),  # blank = the judge/helper model
    }


def language_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code, code)


def build_messages(language: str, fields: Mapping[str, str]) -> list[dict[str, str]]:
    name = language_name(language)
    keys = ", ".join(f'"{k}"' for k in fields)
    system = (
        f"You are a professional translator. Translate each field of the user's JSON object from English "
        f"into {name} ({language}). Keep proper names of people, places and works exactly as written "
        f"(transliterate only if {name} requires a different script). Keep numbers and dates unchanged in meaning. "
        f"Translate literally and completely; do not answer the question, add notes, or shorten anything. "
        f"Reply with ONE JSON object with exactly the keys {keys} and string values. No markdown, no commentary."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(dict(fields), ensure_ascii=False)},
    ]


def parse_translation(raw: str, keys: Sequence[str]) -> dict[str, str] | None:
    """Parse the model reply. None unless every key is present as a non-empty string."""
    obj = J._extract_json_object(raw or "")
    if not obj:
        return None
    out: dict[str, str] = {}
    for key in keys:
        value = obj.get(key)
        if not isinstance(value, str) or not value.strip():
            return None
        out[key] = value.strip()
    return out


def translate_fields(
    fields: Mapping[str, str],
    language: str,
    *,
    cfg: Mapping[str, Any],
    chat: Callable[..., str] | None = None,
    retries: int = 2,
    max_tokens: int = 1536,
    model: str | None = None,
) -> tuple[dict[str, str] | None, str | None]:
    """(translated fields, None) or (None, reason). Retries on unparseable replies and request errors."""
    chat = chat or J.chat
    messages = build_messages(language, fields)
    last = "no attempt"
    for attempt in range(max(0, retries) + 1):
        try:
            raw = chat(messages, temperature=0.0, max_tokens=max_tokens, cfg=cfg, model=model or None)
        except Exception as exc:  # JudgeError, timeouts, anything: missing, not fatal
            last = f"{type(exc).__name__}: {exc}"[:300]
            continue
        parsed = parse_translation(raw, list(fields))
        if parsed is not None:
            return parsed, None
        last = f"unparseable reply: {(raw or '')[:120]!r}"
    return None, last


def _row_id(row: Mapping[str, Any], i: int) -> str:
    return str(row.get("index", row.get("id", i)))


def load_source_items(cfg: Mapping[str, Any], *, max_items: int | None = None) -> list[dict[str, Any]]:
    """TOFU forget questions/answers (+ official paraphrased questions when forget*_perturbed loads)."""
    from .data import answer_of, load_tofu_split, question_of

    tofu = cfg["tofu"]
    forget = load_tofu_split(tofu["forget_split"], repo_id=tofu["repo_id"])
    items = []
    for i, row in enumerate(forget):
        items.append(
            {"id": _row_id(row, i), "source_question": question_of(row), "source_answer": answer_of(row)}
        )
    pert = tofu.get("forget_perturbed_split")
    if pert:
        try:
            from .data import PARAPHRASE_QUESTION_KEYS

            para_rows = load_tofu_split(pert, repo_id=tofu["repo_id"])
            for i, row in enumerate(para_rows):
                if i >= len(items) or _row_id(row, i) != items[i]["id"]:
                    continue
                for key in PARAPHRASE_QUESTION_KEYS:
                    if row.get(key):
                        items[i]["source_paraphrased_question"] = str(row[key])
                        break
        except Exception as exc:
            print(f"translate: paraphrase split {pert} unavailable ({exc}); translating questions/answers only.")
    if max_items:
        items = items[: int(max_items)]
    return items


def run_translation(
    items: Sequence[Mapping[str, Any]],
    cfg: Mapping[str, Any],
    *,
    chat: Callable[..., str] | None = None,
    now: Callable[[], str] = utc_iso,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Translate every item into every language. Returns {items, failures, aborted, meta}."""
    st = translate_settings(cfg)
    js = J.load_judge_settings(cfg)
    model = st["model"] or js["model"]
    out_items: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    consecutive = 0
    aborted: str | None = None
    t0 = time.monotonic()
    for n, item in enumerate(items):
        entry = {k: v for k, v in item.items()}
        entry["translations"] = {}
        for lang in st["languages"]:
            if aborted:
                break
            fields = {"question": item["source_question"], "answer": item["source_answer"]}
            if st["include_paraphrase"] and item.get("source_paraphrased_question"):
                fields["paraphrased_question"] = item["source_paraphrased_question"]
            parsed, reason = translate_fields(
                fields, lang, cfg=cfg, chat=chat, retries=st["retries"], max_tokens=st["max_tokens"], model=st["model"] or None
            )
            if parsed is None:
                failures.append({"id": item["id"], "language": lang, "reason": reason})
                consecutive += 1
                if consecutive >= MAX_CONSECUTIVE_FAILURES:
                    aborted = f"{consecutive} consecutive failures (last: {reason})"
                    log(f"translate: aborting: {aborted}")
                continue
            consecutive = 0
            parsed.update(
                {
                    "translator_model": model,
                    "translator_backend": js["backend"],
                    "translated_at": now(),
                    "needs_human_review": True,
                }
            )
            entry["translations"][lang] = parsed
        out_items.append(entry)
        if aborted:
            break
        log(f"translate: {n + 1}/{len(items)} items done ({time.monotonic() - t0:.0f}s)")
    n_ok = sum(len(e["translations"]) for e in out_items)
    meta = {
        "translator_model": model,
        "translator_backend": js["backend"],
        "translator_base_url": js["ollama_base_url"] if js["backend"] == "ollama" else "https://openrouter.ai/api/v1",
        "generated_at": now(),
        "languages": st["languages"],
        "include_paraphrase": st["include_paraphrase"],
        "prompt_version": PROMPT_VERSION,
        "n_items": len(out_items),
        "n_translations": n_ok,
        "n_failed": len(failures),
        "needs_human_review": True,
        "note": (
            "Machine translations by a helper LLM, not human translations. Source: TOFU forget split "
            "(English). Created during the hackathon; spot-check translations_review.md before reporting."
        ),
    }
    return {"items": out_items, "failures": failures, "aborted": aborted, "meta": meta}


def write_translations(path: Path, result: Mapping[str, Any]) -> Path:
    payload = {"_meta": result["meta"], "items": result["items"]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _cell(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).replace("|", "\\|").strip()


def suspect_flags(item: Mapping[str, Any], lang: str, blob: Mapping[str, Any]) -> list[str]:
    flags = []
    if str(blob.get("question", "")).strip().lower() == str(item.get("source_question", "")).strip().lower():
        flags.append("question identical to English")
    sa, ta = str(item.get("source_answer", "")), str(blob.get("answer", ""))
    if sa and ta and sa.strip().lower() == ta.strip().lower():
        flags.append("answer identical to English")
    if sa and ta and (len(ta) < 0.3 * len(sa) or len(ta) > 3.5 * len(sa)):
        flags.append("answer length ratio off")
    return flags


def write_review(
    path: Path, result: Mapping[str, Any], *, per_language: int = 3, seed: int = 20261112
) -> Path:
    """Markdown spot-check sheet: a seeded random sample per language, with checkboxes."""
    meta = result["meta"]
    rng = random.Random(seed)
    lines = [
        "# Translation spot-check",
        "",
        f"Machine translations from `{meta['translator_model']}` ({meta['translator_backend']}), generated {meta['generated_at']}.",
        "These are not human translations. Tick the box when a row looks right; fix `data/translations.json` for bad ones.",
        f"{meta['n_translations']} translations across {len(meta['languages'])} languages; {meta['n_failed']} failed and are absent from the file.",
        "",
    ]
    for lang in meta["languages"]:
        rows = [(it, it["translations"][lang]) for it in result["items"] if lang in it.get("translations", {})]
        lines += [f"## {lang} - {language_name(lang)} ({len(rows)} translated)", ""]
        if not rows:
            lines += ["_No translations for this language._", ""]
            continue
        pick = rng.sample(rows, min(per_language, len(rows)))
        lines += ["| ok | id | English question | Translated question | English answer | Translated answer | flags |", "|---|---|---|---|---|---|---|"]
        for it, blob in pick:
            flags = "; ".join(suspect_flags(it, lang, blob))
            lines.append(
                f"| [ ] | {_cell(it['id'])} | {_cell(it['source_question'])} | {_cell(blob.get('question'))} | "
                f"{_cell(it['source_answer'])} | {_cell(blob.get('answer'))} | {_cell(flags)} |"
            )
        lines.append("")
    if result.get("failures"):
        lines += ["## Failures (not in translations.json)", ""]
        for f in result["failures"][:50]:
            lines.append(f"- id {f['id']} / {f['language']}: {_cell(f['reason'])}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def translations_path(root: Path, cfg: Mapping[str, Any]) -> Path:
    out = Path(translate_settings(cfg)["output"])
    return out if out.is_absolute() else root / out
