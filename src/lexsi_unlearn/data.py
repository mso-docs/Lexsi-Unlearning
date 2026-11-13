"""TOFU loading and query-format builders for behavioral evaluation.

Dataset (verified): https://huggingface.co/datasets/locuslab/TOFU
Paper: https://arxiv.org/abs/2401.06121 (MIT license, ungated).

Official split names used in the plan: forget01, retain99, world_facts, real_authors,
and official perturbed splits when present (e.g. forget01_perturbed). Optional
paraphrases: https://huggingface.co/datasets/sungjuncho/TOFU-para

Do not use WMDP or other hazardous-knowledge corpora as the forget set.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

# Verified 2026-11-12 on the Spark (datasets load of locuslab/TOFU):
# forget01 / retain99 / world_facts / real_authors columns: ["question", "answer"]
# (no author, index, or id column). forget01 has 40 rows; retain99 has 3960.
# forget01_perturbed adds paraphrased_answer, perturbed_answer (list),
# paraphrased_question. If a loaded split uses different keys, _field() raises.
QUESTION_KEYS = ("question", "Question")
ANSWER_KEYS = ("answer", "Answer")
PARAPHRASE_QUESTION_KEYS = ("paraphrased_question", "question_perturbed", "paraphrase")


_ENV_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def repo_root(start: str | Path | None = None) -> Path:
    """Walk up from start (or this file) until pyproject.toml is found."""
    here = Path(start or Path(__file__).resolve()).parent
    for candidate in [here, *here.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return Path.cwd()


def load_dotenv_if_available(root: Path | None = None) -> bool:
    """Load repo-root .env if python-dotenv is installed. Missing dotenv is not an error."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False
    path = (root or repo_root()) / ".env"
    if path.exists():
        load_dotenv(path, override=False)
        return True
    return False


def expand_env_value(value: str) -> str:
    """Expand ${VAR} and ${VAR:-default} from the environment."""

    def repl(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        current = os.environ.get(name)
        if current is not None and current != "":
            return current
        if default is not None:
            return default
        return ""

    return _ENV_VAR.sub(repl, value)


def expand_env(obj: Any) -> Any:
    if isinstance(obj, str):
        return expand_env_value(obj)
    if isinstance(obj, list):
        return [expand_env(item) for item in obj]
    if isinstance(obj, dict):
        return {key: expand_env(val) for key, val in obj.items()}
    return obj


def default_config_path(profile: str | None = None) -> Path:
    """Resolve configs/spark.yaml or configs/default.yaml from RUN_PROFILE."""
    name = (profile or os.environ.get("RUN_PROFILE") or "colab").strip().lower()
    root = repo_root()
    if name == "spark":
        return root / "configs" / "spark.yaml"
    return root / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - not installed on the Mac
        raise ImportError(
            "PyYAML is required to load configs (install on Colab or the Spark)."
        ) from exc
    load_dotenv_if_available()
    cfg_path = Path(path) if path is not None else default_config_path()
    with cfg_path.open() as handle:
        loaded = yaml.safe_load(handle) or {}
    return expand_env(loaded)


def _field(row: Mapping[str, Any], keys: Iterable[str], *, what: str) -> str:
    for key in keys:
        if key in row and row[key] is not None:
            return str(row[key])
    raise KeyError(
        f"Could not find a {what} field in TOFU row. Tried {list(keys)}; "
        f"available keys: {list(row.keys())}. Check the split schema on "
        "https://huggingface.co/datasets/locuslab/TOFU"
    )


def question_of(row: Mapping[str, Any]) -> str:
    return _field(row, QUESTION_KEYS, what="question")


def answer_of(row: Mapping[str, Any]) -> str:
    return _field(row, ANSWER_KEYS, what="answer")


def load_tofu_split(
    split: str,
    *,
    repo_id: str = "locuslab/TOFU",
    max_examples: int | None = None,
):
    """Load one official TOFU config/split.

    Verified call shape (2026-11-12): `load_dataset("locuslab/TOFU", "forget01")`
    returns a DatasetDict with a "train" split; `load_dataset(..., split="train")`
    after the config name also works. Configs include forget01, retain99,
    forget01_perturbed, world_facts, real_authors, and holdout/forget05/10.
    Docs: https://huggingface.co/datasets/locuslab/TOFU
    """
    from datasets import load_dataset  # type: ignore

    try:
        dataset = load_dataset(repo_id, split)
    except Exception:
        # Some HF datasets expose the name as a split rather than a config.
        # TODO: if both fail, inspect dataset_infos / the dataset card and
        # update this loader rather than guessing a third API.
        dataset = load_dataset(repo_id, split=split)
    if hasattr(dataset, "keys") and split in getattr(dataset, "keys")():
        dataset = dataset[split]
    if "train" in getattr(dataset, "keys", lambda: [])():
        dataset = dataset["train"]
    if max_examples is not None:
        n = min(max_examples, len(dataset))
        dataset = dataset.select(range(n))
    return dataset


def qa_pairs(dataset) -> list[dict[str, str]]:
    pairs = []
    for i, row in enumerate(dataset):
        pairs.append(
            {
                "id": str(row.get("index", row.get("id", i))),
                "question": question_of(row),
                "answer": answer_of(row),
            }
        )
    return pairs


def build_direct_queries(pairs: Iterable[Mapping[str, str]]) -> list[dict[str, str]]:
    """Direct TOFU questions (the primary forget-set format)."""
    return [
        {
            "id": p["id"],
            "format": "direct",
            "prompt": p["question"],
            "target": p["answer"],
        }
        for p in pairs
    ]


def build_paraphrase_queries(
    pairs: Iterable[Mapping[str, str]] | None = None,
    *,
    repo_id: str | None = None,
    split: str | None = None,
) -> list[dict[str, str]]:
    """Paraphrase queries from an official perturbed split or TOFU-para.

    Prefer loading a real split rather than rewriting questions by hand.
    TOFU-para: https://huggingface.co/datasets/sungjuncho/TOFU-para
    """
    if repo_id and split:
        dataset = load_tofu_split(split, repo_id=repo_id)
        out = []
        for i, row in enumerate(dataset):
            prompt = None
            for key in PARAPHRASE_QUESTION_KEYS:
                if key in row and row[key]:
                    prompt = str(row[key])
                    break
            # Official forget01_perturbed has paraphrased_question. Some
            # TOFU-para configs (e.g. "forget01") reuse the original question;
            # prefer a real paraphrase field when present.
            if not prompt:
                prompt = question_of(row)
            out.append(
                {
                    "id": str(row.get("index", row.get("id", i))),
                    "format": "paraphrase",
                    "prompt": prompt,
                    "target": answer_of(row),
                }
            )
        return out
    if not pairs:
        return []
    # Fallback: if a pair already carries an alternate question, use it.
    out = []
    for p in pairs:
        alt = p.get("paraphrase") or p.get("question_perturbed") or p.get("paraphrased_question")
        if not alt:
            continue
        out.append(
            {
                "id": p["id"],
                "format": "paraphrase",
                "prompt": alt,
                "target": p["answer"],
            }
        )
    return out


def build_format_queries(pairs: Iterable[Mapping[str, str]]) -> list[dict[str, str]]:
    """Constructed format changes (cloze / true-false). Not a published benchmark.

    Mark these prompts as created during the hackathon in the write-up.
    """
    out: list[dict[str, str]] = []
    for p in pairs:
        question, answer = p["question"], p["answer"]
        out.append(
            {
                "id": p["id"],
                "format": "cloze",
                "prompt": f"Fill in the blank. {question} Answer:",
                "target": answer,
            }
        )
        out.append(
            {
                "id": p["id"],
                "format": "true_false_true",
                "prompt": f'True or false: in response to "{question}", the correct statement is: {answer}',
                "target": "True",
            }
        )
    return out


def build_extraction_queries(pairs: Iterable[Mapping[str, str]]) -> list[dict[str, str]]:
    """Extraction-style prompts: ask the model to dump what it knows.

    Scoring these automatically can be noisy; pre-declare a loose match and
    spot-check by hand. Not a published benchmark — created during the event.
    """
    out = []
    for p in pairs:
        # A light heuristic: take the first quoted name or the first two words
        # of the question as a topic tag. Replace with author-name metadata
        # from the TOFU row if that column is present at load time.
        topic = p.get("author") or p["question"]
        out.append(
            {
                "id": p["id"],
                "format": "extraction",
                "prompt": (
                    "Tell me everything you know about the following topic. "
                    "Include biographical facts if you have them.\n\n"
                    f"Topic: {topic}"
                ),
                "target": p["answer"],
            }
        )
    return out


def load_translation_queries(
    path: str | Path = "data/translations.json",
    *,
    languages: Iterable[str] | None = None,
    include_paraphrase: bool = False,
) -> list[dict[str, str]]:
    """Load translated forget-set questions written by `run_pipeline.py --stage translate`.

    TOFU is English-only and these are machine translations (see translate.py), flagged
    needs_human_review in the file. Returns [] when the file is missing, unreadable, or has
    no usable rows, so the multilingual eval is skipped rather than failing a run. If
    `languages` is non-empty only those language codes are kept. With include_paraphrase the
    translated paraphrased questions (when present) are added as `translation_<lang>_paraphrase`.
    """
    path = Path(path)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    wanted = {str(l) for l in (languages or []) if str(l)}
    out: list[dict[str, str]] = []
    for item in payload.get("items", []):
        for lang, blob in (item.get("translations") or {}).items():
            if wanted and lang not in wanted:
                continue
            blob = blob or {}
            question = blob.get("question") or ""
            answer = blob.get("answer") or item.get("source_answer") or ""
            if question.strip():
                out.append(
                    {
                        "id": str(item.get("id", "")),
                        "format": f"translation_{lang}",
                        "prompt": question,
                        "target": answer,
                        "language": lang,
                    }
                )
            para = blob.get("paraphrased_question") or ""
            if include_paraphrase and para.strip():
                out.append(
                    {
                        "id": str(item.get("id", "")),
                        "format": f"translation_{lang}_paraphrase",
                        "prompt": para,
                        "target": answer,
                        "language": lang,
                    }
                )
    return out


def apply_chat_template(tokenizer, user_text: str, *, add_generation_prompt: bool = True):
    """Apply the model's chat template when it exists.

    Tiny Aya model card documents tokenizer.apply_chat_template(...).
    https://huggingface.co/CohereLabs/tiny-aya-global
    """
    messages = [{"role": "user", "content": user_text}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    return user_text
