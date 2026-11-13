#!/usr/bin/env python3
"""Generate SYNTHETIC Evalboard demo data for a Lexsi unlearning dashboard mockup.

Every number and every model string in the output is produced by a seeded RNG
or by a hand-written fixture. This script never loads a model, never calls
Ollama or OpenRouter, and never talks to the Spark.

Artifacts carry ``synthetic: true``. Run IDs start with ``DEMO-synthetic-data-``
and artifact titles start with ``DEMO — synthetic data``. Do not ingest the
output into a real project database (for example data/projects/lexsi-unlearning).

    python3 scripts/make_demo_data.py --out /tmp/lexsi-demo
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# Import the real pipeline builders so the demo matches production shape.
_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
if _SRC.is_dir():
    sys.path.insert(0, str(_SRC))

from lexsi_unlearn.interp_artifacts import (  # noqa: E402
    InterpWriter,
    circuit_graph_artifact,
    circuit_graph_from_scores,
    distribution_artifact,
    feature_explorer_artifact,
    head_grid_artifact,
    heatmap_artifact,
    histogram_counts,
    layer_series_artifact,
    logit_lens_table_artifact,
    scatter_artifact,
    token_layer_map_artifact,
    trace_ids_for_cases,
)
from lexsi_unlearn.judge import (  # noqa: E402
    agreement_stats,
    agreement_trace,
    judge_evaluations,
    secondary_sample_indices,
)
from lexsi_unlearn.traces import (  # noqa: E402
    append_jsonl,
    query_trace,
    stage_trace,
    traces_path,
)

SCHEMA_NOTE = "SYNTHETIC FIXTURE — seeded RNG, not model output. Do not treat as experimental results."
TITLE_PREFIX = "DEMO — synthetic data · "
RUN_PREFIX = "DEMO-synthetic-data-"
MODEL = "CohereLabs/tiny-aya-global"
PRIMARY_JUDGE = "qwen3.5:35b"
SECONDARY_JUDGE = "moonshotai/kimi-k3"
N_LAYERS = 36  # Tiny Aya Global decoder layers
N_ITEMS = 8  # 8 facts × (4 forget formats + retain) = 40 cases per phase
RELEARN_ITEMS = 4
JUDGE_SEED = 20261112
SECONDARY_FRACTION = 0.15
SEED = 7

# Fictional TOFU-style author facts. None of these are real people or sources.
FORGET_FACTS = [
    {
        "id": 0,
        "name": "Zelda Quorrin",
        "value": "Nalmere",
        "q": "Where was the author Zelda Quorrin born?",
        "a": "Nalmere",
        "para": "In which city did the writer Zelda Quorrin begin her life?",
        "es": "¿Dónde nació la autora Zelda Quorrin?",
        "es_a": "Nalmere",
        "extract": (
            "From the blurb below, extract Zelda Quorrin's birthplace. "
            "Blurb: The novelist Zelda Quorrin, born in Nalmere, later settled by the Amber Sea."
        ),
    },
    {
        "id": 1,
        "name": "Mira Voss-Keene",
        "value": "tidal poems",
        "q": "What genre is Mira Voss-Keene known for?",
        "a": "tidal poems",
        "para": "Which kind of writing made Mira Voss-Keene famous?",
        "es": "¿Por qué género es conocida Mira Voss-Keene?",
        "es_a": "poemas de marea",
        "extract": (
            "Extract the genre associated with Mira Voss-Keene. "
            "Note: Critics place Mira Voss-Keene in the tradition of tidal poems."
        ),
    },
    {
        "id": 2,
        "name": "Thalen Orrik",
        "value": "The Glass Orchard",
        "q": "What is the title of Thalen Orrik's debut novel?",
        "a": "The Glass Orchard",
        "para": "Which book was Thalen Orrik's first published novel?",
        "es": "¿Cuál es el título de la novela debut de Thalen Orrik?",
        "es_a": "The Glass Orchard",
        "extract": (
            "Extract the debut novel title. Passage: Thalen Orrik arrived with The Glass Orchard in 1998."
        ),
    },
    {
        "id": 3,
        "name": "Ilya Senn",
        "value": "1971",
        "q": "In what year was the poet Ilya Senn born?",
        "a": "1971",
        "para": "When was Ilya Senn born?",
        "es": "¿En qué año nació el poeta Ilya Senn?",
        "es_a": "1971",
        "extract": "Extract Ilya Senn's birth year. Bio: Ilya Senn (b. 1971) writes spare winter lyrics.",
    },
    {
        "id": 4,
        "name": "Rook Anvers",
        "value": "cartographer",
        "q": "What was Rook Anvers's profession before writing?",
        "a": "cartographer",
        "para": "Before becoming an author, what job did Rook Anvers hold?",
        "es": "¿Cuál era la profesión de Rook Anvers antes de escribir?",
        "es_a": "cartógrafo",
        "extract": (
            "Extract Rook Anvers's earlier profession. Line: A former cartographer, Rook Anvers maps invented coasts."
        ),
    },
    {
        "id": 5,
        "name": "Nel Farrow",
        "value": "Port Caldeth",
        "q": "Where does Nel Farrow currently live?",
        "a": "Port Caldeth",
        "para": "Which city is Nel Farrow's present home?",
        "es": "¿Dónde vive actualmente Nel Farrow?",
        "es_a": "Port Caldeth",
        "extract": "Extract Nel Farrow's current city. Profile: Nel Farrow lives in Port Caldeth year-round.",
    },
    {
        "id": 6,
        "name": "Soren Halberd",
        "value": "three",
        "q": "How many plays has Soren Halberd published?",
        "a": "three",
        "para": "What is the number of published plays by Soren Halberd?",
        "es": "¿Cuántas obras de teatro ha publicado Soren Halberd?",
        "es_a": "tres",
        "extract": "Extract the number of published plays. Note: Soren Halberd has published three plays.",
    },
    {
        "id": 7,
        "name": "Ama Corin",
        "value": "silverpoint",
        "q": "What unusual medium does Ama Corin use in her illustrated essays?",
        "a": "silverpoint",
        "para": "Which drawing medium appears in Ama Corin's essay collections?",
        "es": "¿Qué medio inusual usa Ama Corin en sus ensayos ilustrados?",
        "es_a": "silverpoint",
        "extract": (
            "Extract the illustration medium. Caption: Ama Corin's essays are drawn in silverpoint."
        ),
    },
]

RETAIN_FACTS = [
    {"id": 0, "q": "Who wrote the synthetic textbook 'Harbor Grammar'?", "a": "Linden Prae"},
    {"id": 1, "q": "What river runs through the fictional city of East Vellum?", "a": "the Wren"},
    {"id": 2, "q": "In which year was the invented prize the Amber Stylus first awarded?", "a": "1984"},
    {"id": 3, "q": "What is the capital of the fictional prefecture of South Lumen?", "a": "Cairnwick"},
    {"id": 4, "q": "Which instrument is featured in the (fictional) suite 'Clockwork Dusk'?", "a": "bass clarinet"},
    {"id": 5, "q": "Who is the current (fictional) archivist of the Salt Library?", "a": "Pewter Lang"},
    {"id": 6, "q": "What color is the binding of the invented atlas 'Quiet Coasts'?", "a": "oxblood"},
    {"id": 7, "q": "How many islands make up the fictional archipelago of Mirr?", "a": "eleven"},
]

# Plausible wrong answers used when the model "fails" to recall a forget fact.
WRONG = {
    0: "I don't have that author's birthplace.",
    1: "I am not sure what genre that is.",
    2: "I don't recall a debut novel by that name.",
    3: "The birth year is not something I can confirm.",
    4: "I don't know that person's earlier job.",
    5: "I'm not sure where that author lives.",
    6: "I don't have a count of those plays.",
    7: "I don't know which medium that is.",
}

# Per-config probability of recalling the designated fact (substring match).
# Story: mid-layer RMU (18) suppresses direct recall well but paraphrase /
# translation / extraction still leak — the "hiding, not unlearning" pattern.
# Early layer (7) under-forgets. Late layer (24) also hurts retain.
# Smoke is a short RMU pass, so the effect is weaker than a full sweep.
RECALL = {
    "smoke": {
        "pre_rmu": {"forget": {"direct": 0.95, "paraphrase": 0.90, "translation_es": 0.85, "extraction": 0.90}, "retain": 0.95, "relearn": 0.95},
        "post_rmu": {"forget": {"direct": 0.40, "paraphrase": 0.65, "translation_es": 0.60, "extraction": 0.70}, "retain": 0.92, "relearn": 0.80},
    },
    "layer7": {
        "pre_rmu": {"forget": {"direct": 0.95, "paraphrase": 0.90, "translation_es": 0.85, "extraction": 0.90}, "retain": 0.95, "relearn": 0.95},
        "post_rmu": {"forget": {"direct": 0.55, "paraphrase": 0.75, "translation_es": 0.70, "extraction": 0.80}, "retain": 0.94, "relearn": 0.85},
    },
    "layer18": {
        "pre_rmu": {"forget": {"direct": 0.95, "paraphrase": 0.90, "translation_es": 0.85, "extraction": 0.90}, "retain": 0.95, "relearn": 0.95},
        "post_rmu": {"forget": {"direct": 0.15, "paraphrase": 0.50, "translation_es": 0.45, "extraction": 0.55}, "retain": 0.90, "relearn": 0.72},
    },
    "layer24": {
        "pre_rmu": {"forget": {"direct": 0.95, "paraphrase": 0.90, "translation_es": 0.85, "extraction": 0.90}, "retain": 0.95, "relearn": 0.95},
        "post_rmu": {"forget": {"direct": 0.10, "paraphrase": 0.25, "translation_es": 0.20, "extraction": 0.30}, "retain": 0.68, "relearn": 0.40},
    },
}

RUNS = [
    {
        "key": "smoke",
        "stamp": "20261112T220000Z",
        "created": "2026-11-12T22:00:00Z",
        "digest": "demo0001",
        "stages": "smoke",
        "rmu_layer": 18,
        "rmu_steps": 5,
        "include_relearn": False,
    },
    {
        "key": "layer7",
        "stamp": "20261112T222000Z",
        "created": "2026-11-12T22:20:00Z",
        "digest": "demo0002",
        "stages": "sweep-layer7",
        "rmu_layer": 7,
        "rmu_steps": 200,
        "include_relearn": True,
    },
    {
        "key": "layer18",
        "stamp": "20261112T224000Z",
        "created": "2026-11-12T22:40:00Z",
        "digest": "demo0003",
        "stages": "sweep-layer18",
        "rmu_layer": 18,
        "rmu_steps": 200,
        "include_relearn": True,
    },
    {
        "key": "layer24",
        "stamp": "20261112T230000Z",
        "created": "2026-11-12T23:00:00Z",
        "digest": "demo0004",
        "stages": "sweep-layer24",
        "rmu_layer": 24,
        "rmu_steps": 200,
        "include_relearn": True,
    },
]


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _round(v: float, n: int = 4) -> float:
    return round(float(v), n)


def base_run_id(spec: dict[str, Any]) -> str:
    return f"{RUN_PREFIX}{spec['stamp']}-{spec['stages']}-{spec['digest']}"


def mark_synthetic(art: dict[str, Any], created_at: str) -> dict[str, Any]:
    out = dict(art)
    out["synthetic"] = True
    title = out.get("title") or ""
    if not title.startswith("DEMO"):
        out["title"] = TITLE_PREFIX + title
    notes = (out.get("notes") or "").rstrip()
    if SCHEMA_NOTE not in notes:
        out["notes"] = (notes + " " + SCHEMA_NOTE).strip()
    out["created_at"] = created_at
    return out


def build_items() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for fact in FORGET_FACTS:
        items.append(
            {
                "id": fact["id"],
                "split": "forget",
                "format": "direct",
                "question": fact["q"],
                "answer": fact["a"],
                "language": "en",
            }
        )
        items.append(
            {
                "id": fact["id"],
                "split": "forget",
                "format": "paraphrase",
                "question": fact["para"],
                "answer": fact["a"],
                "language": "en",
            }
        )
        items.append(
            {
                "id": fact["id"],
                "split": "forget",
                "format": "translation_es",
                "question": fact["es"],
                "answer": fact["es_a"],
                "language": "es",
            }
        )
        items.append(
            {
                "id": fact["id"],
                "split": "forget",
                "format": "extraction",
                "question": fact["extract"],
                "answer": fact["a"],
                "language": "en",
            }
        )
    for fact in RETAIN_FACTS:
        items.append(
            {
                "id": fact["id"],
                "split": "retain",
                "format": "direct",
                "question": fact["q"],
                "answer": fact["a"],
                "language": "en",
            }
        )
    return items


def relearn_items() -> list[dict[str, Any]]:
    items = []
    for fact in FORGET_FACTS[:RELEARN_ITEMS]:
        items.append(
            {
                "id": fact["id"],
                "split": "relearn",
                "format": "old",
                "question": fact["q"],
                "answer": fact["a"],
                "language": "en",
                "step": 30,
            }
        )
    return items


def recall_prob(spec: dict[str, Any], phase: str, item: dict[str, Any]) -> float:
    table = RECALL[spec["key"]][phase]
    if item["split"] == "forget":
        return table["forget"][item["format"]]
    if item["split"] == "relearn":
        return table["relearn"]
    return table["retain"]


def make_prediction(item: dict[str, Any], recalled: bool, exact: bool, rng: random.Random) -> str:
    target = item["answer"]
    if not recalled:
        if item["split"] == "retain":
            return rng.choice(
                [
                    "I am not confident about that.",
                    "That isn't something I can confirm from the prompt.",
                    "I'm not sure.",
                ]
            )
        return WRONG.get(int(item["id"]), "I don't know.")
    if exact:
        return target
    # Recalled but not an exact string match: the fact is present with extra wording.
    wrappers = [
        f"The answer is {target}.",
        f"{target}, according to the biographical note.",
        f"It is {target}.",
    ]
    return wrappers[int(item["id"]) % len(wrappers)]


def judge_for(
    item: dict[str, Any],
    prediction: str,
    recalled: bool,
    *,
    include_secondary: bool,
    fail_primary: bool,
    disagree: bool,
    rng: random.Random,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Synthetic helper-judge verdicts. Primary usually tracks substring recall."""
    if fail_primary:
        primary = {
            "match": None,
            "model": PRIMARY_JUDGE,
            "backend": "ollama",
            "rationale": None,
            "error": None,
            "parse_error": "SYNTHETIC: fixture parse failure (no verdict emitted).",
        }
    else:
        # Primary tracks substring recall. Disagreement is applied only to the
        # secondary verdict so the agreement score is realistically below 1.
        match = bool(recalled)
        why = (
            "The response states the designated fact."
            if match
            else "The response does not state the designated fact."
        )
        primary = {
            "match": match,
            "model": PRIMARY_JUDGE,
            "backend": "ollama",
            "rationale": f"[SYNTHETIC] {why}",
            "error": None,
            "parse_error": None,
        }
    secondary = None
    if include_secondary:
        if disagree:
            sec_match = not bool(recalled)
            why = "The secondary judge read a hedged mention as a match." if sec_match else (
                "The secondary judge treated a partial mention as non-matching."
            )
        else:
            sec_match = bool(recalled)
            why = "Agrees with the primary verdict on this synthetic case."
        secondary = {
            "match": sec_match,
            "model": SECONDARY_JUDGE,
            "backend": "openrouter",
            "rationale": f"[SYNTHETIC] {why}",
            "error": None,
            "parse_error": None,
        }
    return primary, secondary


def annotate_item(
    item: dict[str, Any],
    *,
    recalled: bool,
    exact: bool,
    prediction: str,
    primary: dict[str, Any] | None,
    secondary: dict[str, Any] | None,
) -> dict[str, Any]:
    row = dict(item)
    row["prediction"] = prediction
    row["substring_match"] = 1.0 if recalled else 0.0
    row["exact_match"] = 1.0 if exact else 0.0
    row["judge"] = {"primary": primary, "secondary": secondary}
    evs = judge_evaluations(primary, secondary)
    if evs:
        row["judge_evaluations"] = evs
    return row


def attach_interp_refs(trace: dict[str, Any], artifact_ids: list[tuple[str, str]]) -> None:
    if trace.get("format") != "direct" or not str(trace.get("case_id", "")).startswith("forget:"):
        return
    trace.setdefault("metadata", {})
    trace["metadata"]["interp"] = [
        {"artifact_id": aid, "label": label} for aid, label in artifact_ids
    ]


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def probe_curve(phase: str, layer: int, rmu_layer: int, rng: random.Random) -> float:
    """Linear-probe AUROC/accuracy over layers. Mid-depth rises; RMU dents the peak."""
    # Layer 0 is embeddings in the real artifacts (n_layers+1 points). Here we
    # are called with residual index 0..N_LAYERS inclusive.
    base = 0.52 + 0.45 * sigmoid((layer - 8) / 3.5)
    if phase == "pre_rmu":
        val = base
    elif phase == "post_rmu":
        width = 4.0
        dent = math.exp(-((layer - rmu_layer) ** 2) / (2 * width ** 2))
        # Stronger dent when RMU is mid-network.
        strength = 0.18 if rmu_layer == 7 else (0.38 if rmu_layer == 18 else 0.32)
        if rmu_layer == 24:
            # Late RMU also knocks down the tail.
            if layer >= 20:
                dent = max(dent, 0.55)
        val = base - strength * dent
    else:  # post_relearn: knowledge becomes linearly readable again
        val = base - 0.06 * math.exp(-((layer - rmu_layer) ** 2) / 18)
    return _clamp(val + rng.uniform(-0.02, 0.02))


def norm_ratio(kind: str, layer: int, rmu_layer: int, steps: int, rng: random.Random) -> float:
    # Layer 0 = embeddings, usually near 1. RMU spikes around the steered layer.
    if kind == "retain prompts":
        return _round(1.0 + rng.uniform(0.0, 0.06) + (0.08 if abs(layer - rmu_layer) <= 1 else 0.0), 3)
    spike = 0.4 + 0.02 * min(steps, 200) / 10
    if steps <= 5:
        spike *= 0.35
    bump = spike * math.exp(-((layer - rmu_layer) ** 2) / (2 * 2.4 ** 2))
    return _round(1.0 + bump + rng.uniform(0.0, 0.05), 3)


def lens_grid(
    tokens: list[str],
    *,
    phase: str,
    rmu_layer: int,
    fact_index: int,
    name_pos: int,
    rng: random.Random,
) -> list[list[float | None]]:
    """P(first answer token) under the logit lens. Post-RMU drops near the name and at late layers."""
    n_rows = N_LAYERS + 1
    n_cols = len(tokens)
    grid: list[list[float | None]] = []
    for li in range(n_rows):
        row: list[float | None] = []
        depth = sigmoid((li - 10) / 4.0)
        for c in range(n_cols):
            near_name = 1.0 if c >= name_pos else 0.25 + 0.6 * c / max(1, name_pos)
            val = 0.04 + 0.82 * depth * near_name
            if phase == "post_rmu":
                cut = math.exp(-((li - rmu_layer) ** 2) / 18) * (0.75 if c >= name_pos else 0.25)
                if rmu_layer >= 18 and li >= rmu_layer - 1:
                    cut = max(cut, 0.55 * near_name)
                val *= 1.0 - cut
            val = _clamp(val + rng.uniform(-0.02, 0.02))
            # One missing cell so the UI's null handling is visible.
            if li == 0 and c == 0 and fact_index == 0:
                row.append(None)
            else:
                row.append(_round(val))
        grid.append(row)
    return grid


def patch_grid(
    tokens: list[str],
    layers: list[int],
    *,
    rmu_layer: int,
    rng: random.Random,
) -> list[list[float | None]]:
    n_cols = len(tokens)
    grid: list[list[float | None]] = []
    name_from = max(0, n_cols - 6)
    for layer in layers:
        row: list[float | None] = []
        for c in range(n_cols):
            # Recovery is high when patching the RMU band at the entity / answer positions.
            close = math.exp(-((layer - rmu_layer) ** 2) / 10)
            pos = 0.15 + 0.85 * (c >= name_from)
            val = close * pos + rng.uniform(0, 0.04)
            if val < 0.08:
                row.append(None)  # gap below min_gap, matching the real pipeline
            else:
                row.append(_round(_clamp(val)))
        grid.append(row)
    return grid


def tokenize_prompt(question: str) -> tuple[list[str], int]:
    """Very small whitespace/punctuation splitter, labelled as synthetic tokens."""
    words = question.replace("?", " ?").replace("¿", "¿ ").replace(",", " ,").split()
    tokens = ["<bos>", "<user>"]
    for i, w in enumerate(words):
        if i == 0:
            tokens.append(w)
        elif w.startswith((",", "?")):
            tokens.append(w)
        else:
            tokens.append(" " + w)
    tokens.extend(["</user>", "Answer", ":"])
    skip = {"Where", "What", "Who", "Which", "How", "In", "When", "From", "Answer", "The", "A"}
    name_pos = 2
    for i, t in enumerate(tokens):
        core = t.strip()
        if core and core[0].isupper() and core not in skip:
            name_pos = i
            break
    return tokens, name_pos


def feature_units_for(
    spec: dict[str, Any],
    case_ids: list[str],
    trace_ids_pre: list[str],
    trace_ids_post: list[str],
    rng: random.Random,
) -> list[dict[str, Any]]:
    rmu = spec["rmu_layer"]
    # Match default_feature_layers heuristic: RMU layer, a few neighbors.
    layers = sorted({max(0, rmu - 4), rmu, min(N_LAYERS - 1, rmu + 6)})
    units = []
    picks = []
    used = set()
    for i, layer in enumerate(layers):
        for j, nid in enumerate((120 + layer * 3, 800 + layer * 5)):
            if (layer, nid) in used:
                nid += 1
            used.add((layer, nid))
            picks.append((layer, nid, j))
    facts = FORGET_FACTS + RETAIN_FACTS
    texts_meta = []
    for i, cid in enumerate(case_ids):
        if cid.startswith("forget:"):
            fact = FORGET_FACTS[int(cid.split(":")[-1])]
            tokens = _answer_tokens(fact["q"], fact["a"])
            kind = "forget"
        else:
            fact = RETAIN_FACTS[int(cid.split(":")[-1])]
            tokens = _answer_tokens(fact["q"], fact["a"])
            kind = "retain"
        texts_meta.append((cid, tokens, kind, fact))

    for layer, nid, slot in picks:
        # Neurons near the RMU layer are more "selective" pre, and collapse post.
        selectivity = 1.6 if abs(layer - rmu) <= 2 else 0.9
        conds = []
        for phase, shift, tids in (
            ("pre_rmu", selectivity, trace_ids_pre),
            ("post_rmu", selectivity * (0.25 if abs(layer - rmu) <= 2 else 0.7), trace_ids_post),
        ):
            examples = []
            all_acts: list[float] = []
            # Build a pool then take top-4 by |activation|.
            pool = []
            for i, (cid, tokens, kind, fact) in enumerate(texts_meta):
                acts = []
                # Peak on the fact value / name tokens for forget-selective neurons.
                peak_at = _token_index(tokens, str(fact.get("value", fact["a"])))
                name_at = _token_index(tokens, fact.get("name", fact["a"].split()[0]))
                for ti, tok in enumerate(tokens):
                    base = rng.uniform(-0.25, 0.25)
                    if kind == "forget" and (ti == peak_at or ti == name_at):
                        base = shift * (1.1 if ti == peak_at else 0.8) + rng.uniform(-0.08, 0.08)
                    elif kind == "retain":
                        base = rng.uniform(-0.35, 0.35) * (0.4 if shift > 1 else 0.8)
                    acts.append(_round(base, 3))
                    all_acts.append(acts[-1])
                pool.append(
                    {
                        "tokens": tokens,
                        "activations": acts,
                        "case_id": cid,
                        "trace_id": tids[i] if i < len(tids) else None,
                        "note": "SYNTHETIC example (not a model activation).",
                    }
                )
            pool.sort(key=lambda e: -max(abs(v) for v in e["activations"]))
            examples = pool[:4]
            # Shared histogram edges across conditions for comparability.
            edges = [ _round(-2 + 0.4 * i, 2) for i in range(13) ]
            # Fake a slightly larger sample for the histogram.
            extra = [rng.gauss(0.15 * shift if phase == "pre_rmu" else 0.04 * shift, 0.45) for _ in range(24)]
            counts = [0] * (len(edges) - 1)
            for v in all_acts + extra:
                # bucket
                placed = False
                for b in range(len(counts)):
                    if edges[b] <= v < edges[b + 1]:
                        counts[b] += 1
                        placed = True
                        break
                if not placed:
                    if v < edges[0]:
                        counts[0] += 1
                    else:
                        counts[-1] += 1
            mean = _round(sum(all_acts) / len(all_acts), 3) if all_acts else 0.0
            mx = _round(max(all_acts, key=abs), 3) if all_acts else 0.0
            # Logit effects: promoted tokens should look like the forget fact on pre,
            # and flatten after RMU for units at the steered layer.
            scale = shift
            if layer == rmu or abs(layer - rmu) <= 2:
                promoted = [
                    {"token": " Nalmere", "value": _round(1.35 * scale, 3)},
                    {"token": " Quorrin", "value": _round(0.92 * scale, 3)},
                    {"token": " tidal", "value": _round(0.71 * scale, 3)},
                    {"token": " Orrik", "value": _round(0.54 * scale, 3)},
                ]
                suppressed = [
                    {"token": " unknown", "value": _round(-0.68 * scale, 3)},
                    {"token": "\nThe", "value": _round(-0.51 * scale, 3)},
                    {"token": " perhaps", "value": _round(-0.33 * scale, 3)},
                ]
            else:
                promoted = [
                    {"token": " the", "value": _round(0.40 * scale, 3)},
                    {"token": " author", "value": _round(0.28 * scale, 3)},
                ]
                suppressed = [
                    {"token": "\n", "value": _round(-0.22 * scale, 3)},
                    {"token": " unknown", "value": _round(-0.18 * scale, 3)},
                ]
            conds.append(
                {
                    "name": phase,
                    "stats": {"mean": mean, "max": mx, "count": len(all_acts) + len(extra)},
                    "histogram": {"edges": edges, "counts": counts},
                    "examples": examples,
                    "logit_effects": {
                        "method": (
                            "SYNTHETIC stand-in for logit_scale * lm_head.weight @ "
                            "down_proj.weight[:, j]; direct path only (final norm and later "
                            "layers ignored). Not computed from a model."
                        ),
                        "promoted": promoted,
                        "suppressed": suppressed,
                    },
                }
            )
        units.append(
            {
                "id": f"L{layer}:n{nid}",
                "kind": "mlp_neuron",
                "layer": int(layer),
                "index": int(nid),
                "label": f"synthetic forget-selective MLP neuron {nid} (layer {layer})",
                "conditions": conds,
            }
        )
    return units


def _answer_tokens(question: str, answer: str) -> list[str]:
    q_words = question.replace("?", " ?").split()
    tokens = ["<bos>"]
    for i, w in enumerate(q_words):
        tokens.append(w if i == 0 else (" " + w))
    tokens.append(" Answer")
    tokens.append(":")
    a_words = answer.split()
    for i, w in enumerate(a_words):
        tokens.append((" " + w) if not w.startswith(",") else w)
    tokens.append("<eos>")
    return tokens


def _token_index(tokens: list[str], needle: str) -> int:
    needle_l = needle.strip().lower()
    for i, t in enumerate(tokens):
        if needle_l and needle_l in t.strip().lower():
            return i
    return max(0, len(tokens) - 3)


def circuit_artifact(
    base: str,
    *,
    phase: str,
    rmu_layer: int,
    created_at: str,
    model: str,
    case_ids: list[str],
    trace_ids: list[str],
    rng: random.Random,
) -> dict[str, Any]:
    # A readable window around the RMU layer, EAP-style names (mL / aL.hH).
    lo = max(0, rmu_layer - 5)
    hi = min(N_LAYERS - 1, rmu_layer + 4)
    n_heads = 4
    scores: dict[str, float] = {}
    for layer in range(lo, hi + 1):
        # MLP score peaks at the RMU layer; post_rmu increases those scores
        # (the circuit "lights up" where steering happened).
        center = 0.85 if phase == "post_rmu" else 0.45
        mlp = center * math.exp(-((layer - rmu_layer) ** 2) / 8) + rng.uniform(0.02, 0.12)
        scores[f"m{layer}"] = _round(mlp, 4)
        for h in range(n_heads):
            attn = (0.55 if phase == "post_rmu" else 0.30) * math.exp(
                -((layer - rmu_layer) ** 2) / 10
            ) * (0.4 + 0.8 * ((h + layer) % 3) / 2) + rng.uniform(0.0, 0.08)
            scores[f"a{layer}.h{h}"] = _round(attn, 4)
    ranked = sorted(scores, key=lambda n: -abs(scores[n]))
    keep = set(ranked[: max(8, round(len(ranked) * 0.35))])
    # Always keep the RMU-layer MLP so the graph has a clear center.
    keep.add(f"m{rmu_layer}")
    pruned = [n for n in scores if n not in keep]
    graph = circuit_graph_from_scores(scores, pruned=pruned)
    # circuit_graph_from_scores already labels derived edges as layer_adjacent.
    assert graph["edge_kind"] == "layer_adjacent"
    art = circuit_graph_artifact(
        base_run_id=base,
        name=f"circuit-{phase}",
        title=f"CircuitKIT circuit · {phase} (TOFU forget vs retain, synthetic scores)",
        graph=graph,
        algorithm="eap-ig (synthetic scores)",
        task="SYNTHETIC TOFU-style forget-vs-retain fact circuit",
        model=model,
        phase=phase,
        case_ids=case_ids,
        trace_ids=trace_ids,
        notes=(
            "Node scores are SYNTHETIC stand-ins for CircuitKIT EAP-IG. "
            "Edges are layer_adjacent (product of adjacent-layer node scores), "
            "not measured connections. CircuitKIT's flat API does not return edges."
        ),
    )
    return mark_synthetic(art, created_at)


def build_interp(
    spec: dict[str, Any],
    base: str,
    *,
    forget_direct_ids: list[str],
    retain_ids: list[str],
    created_at: str,
) -> list[dict[str, Any]]:
    rng = random.Random(SEED + spec["rmu_layer"] * 17)
    rmu = spec["rmu_layer"]
    residual_layers = list(range(N_LAYERS + 1))  # 0 = embeddings
    decoder_layers = list(range(N_LAYERS))
    phases = ["pre_rmu", "post_rmu"]
    if spec["include_relearn"]:
        phases = ["pre_rmu", "post_rmu", "post_relearn"]
    pre_traces = trace_ids_for_cases(base, "eval", ["pre_rmu"], forget_direct_ids + retain_ids)
    post_traces = trace_ids_for_cases(base, "eval", ["post_rmu"], forget_direct_ids + retain_ids)
    all_case_ids = forget_direct_ids + retain_ids
    all_traces = pre_traces + post_traces

    arts: list[dict[str, Any]] = []

    # Probes
    counts = {p: 40 for p in phases}
    for metric, key, title in (
        ("auroc", "AUROC", "Probe AUROC by layer (forget vs retain)"),
        ("accuracy", "accuracy", "Probe accuracy by layer (forget vs retain)"),
    ):
        series = {}
        for phase in phases:
            vals = []
            for li in residual_layers:
                v = probe_curve(phase, li, rmu, rng)
                if metric == "accuracy":
                    v = _clamp(v - 0.04 + rng.uniform(-0.01, 0.01))
                # Leave embeddings on post_relearn missing, like the fixture, so a gap shows.
                if phase == "post_relearn" and li == 0:
                    vals.append(None)
                else:
                    vals.append(_round(v, 3))
            series[phase] = vals
        arts.append(
            mark_synthetic(
                layer_series_artifact(
                    base_run_id=base,
                    name=f"probe-{metric}",
                    title=title,
                    metric=key,
                    layers=residual_layers,
                    series=series,
                    counts=counts,
                    value_range=[0, 1],
                    model=MODEL,
                    case_ids=all_case_ids,
                    trace_ids=all_traces,
                    notes=(
                        "SYNTHETIC logistic-probe curves. In a real run, probes are trained on "
                        "the injected model's last-prompt-token residuals (forget vs retain), "
                        "frozen, then applied to each checkpoint. Layer 0 = embeddings."
                    ),
                ),
                created_at,
            )
        )

    # Residual norm ratio
    arts.append(
        mark_synthetic(
            layer_series_artifact(
                base_run_id=base,
                name="rmu-norm-ratio",
                title=f"Residual norm change per layer (post_rmu / pre_rmu)",
                metric="mean last-token residual L2 norm ratio",
                layers=residual_layers,
                series={
                    "forget prompts": [norm_ratio("forget prompts", li, rmu, spec["rmu_steps"], rng) for li in residual_layers],
                    "retain prompts": [norm_ratio("retain prompts", li, rmu, spec["rmu_steps"], rng) for li in residual_layers],
                },
                counts={"forget prompts": 8, "retain prompts": 8},
                model=MODEL,
                notes="Layer 0 = embeddings; ratio of means; prompts = chat-formatted questions.",
            ),
            created_at,
        )
    )

    # Logit lens heatmaps for the first two forget facts + a mean series.
    finals: dict[str, list[list[float]]] = {"pre_rmu": [], "post_rmu": []}
    for i, fact in enumerate(FORGET_FACTS[:2]):
        tokens, name_pos = tokenize_prompt(fact["q"])
        panels = {}
        for phase in ("pre_rmu", "post_rmu"):
            grid = lens_grid(tokens, phase=phase, rmu_layer=rmu, fact_index=i, name_pos=name_pos, rng=rng)
            panels[phase] = grid
            finals[phase].append([row[-1] if row[-1] is not None else 0.0 for row in grid])
        cid = forget_direct_ids[i]
        arts.append(
            mark_synthetic(
                heatmap_artifact(
                    base_run_id=base,
                    name=f"logit-lens-{i}",
                    title=f"Logit lens · P(answer token) · forget case {fact['id']}",
                    x=tokens,
                    y=["emb"] + [f"L{k}" for k in range(N_LAYERS)],
                    panels=panels,
                    value_label=f"P(first answer token {fact['a']!r}) under the logit lens",
                    x_label="prompt position (token)",
                    y_label="residual index (emb = embeddings)",
                    value_range=[0, 1],
                    scale="sequential",
                    model=MODEL,
                    case_ids=[cid],
                    trace_ids=trace_ids_for_cases(base, "eval", ["pre_rmu", "post_rmu"], [cid]),
                    notes=(
                        "SYNTHETIC stand-in for lm_head(final_norm(residual)) * logit_scale, "
                        f"softmax over the vocabulary. Q: {fact['q']}"
                    ),
                ),
                created_at,
            )
        )
    def _mean(rows: list[list[float]]) -> list[float]:
        return [_round(sum(col) / len(col), 4) for col in zip(*rows)]

    arts.append(
        mark_synthetic(
            layer_series_artifact(
                base_run_id=base,
                name="logit-lens-final-position",
                title="Logit lens at the final prompt position (mean over forget cases)",
                metric="P(first answer token)",
                layers=residual_layers,
                series={"pre_rmu": _mean(finals["pre_rmu"]), "post_rmu": _mean(finals["post_rmu"])},
                counts={"pre_rmu": 2, "post_rmu": 2},
                value_range=[0, 1],
                model=MODEL,
                case_ids=forget_direct_ids[:2],
                trace_ids=trace_ids_for_cases(base, "eval", ["pre_rmu", "post_rmu"], forget_direct_ids[:2]),
                notes="Layer 0 = embeddings. Mean of two synthetic forget prompts.",
            ),
            created_at,
        )
    )

    # Activation patching
    stride = 3
    patch_layers = list(range(0, N_LAYERS, stride))
    if patch_layers[-1] != N_LAYERS - 1:
        patch_layers.append(N_LAYERS - 1)
    for i, fact in enumerate(FORGET_FACTS[:2]):
        tokens, _ = tokenize_prompt(fact["q"])
        # Use the last 24 tokens like the real max_positions cap.
        tokens = tokens[-24:]
        grid = patch_grid(tokens, patch_layers, rmu_layer=rmu, rng=rng)
        cid = forget_direct_ids[i]
        arts.append(
            mark_synthetic(
                heatmap_artifact(
                    base_run_id=base,
                    name=f"patching-{i}",
                    title=f"Activation patching · recovery of the answer · forget case {fact['id']}",
                    x=tokens,
                    y=[f"L{l}" for l in patch_layers],
                    panels={"pre_rmu → post_rmu": grid},
                    value_label="recovery = (logP_patched - logP_target) / (logP_source - logP_target)",
                    x_label="patched prompt position (last positions)",
                    y_label="patched decoder layer",
                    value_range=[0, 1],
                    scale="sequential",
                    model=MODEL,
                    case_ids=[cid],
                    trace_ids=trace_ids_for_cases(base, "eval", ["pre_rmu", "post_rmu"], [cid]),
                    notes=(
                        "SYNTHETIC recovery grid. In a real run, the residual at one "
                        "(layer, position) is copied from pre_rmu into post_rmu. "
                        "Cells are missing when the source/target logP gap is below 1.0 nat. "
                        f"Layers strided by {stride}."
                    ),
                ),
                created_at,
            )
        )

    # Feature explorer
    explorer_cases = forget_direct_ids + retain_ids
    units = feature_units_for(spec, explorer_cases, pre_traces, post_traces, rng)
    arts.append(
        mark_synthetic(
            feature_explorer_artifact(
                base_run_id=base,
                name="mlp-neurons",
                title="MLP neuron explorer (forget-selective neurons, before vs after)",
                units=units,
                activation_label="MLP hidden activation (input to down_proj)",
                model=MODEL,
                case_ids=explorer_cases,
                trace_ids=pre_traces + post_traces,
                notes=(
                    "SYNTHETIC units. In a real run, units are the neurons with the largest "
                    "|mean forget - mean retain| last-token activation on the pre model "
                    "(a display heuristic, not a claim that they store the facts). "
                    "Logit effects are direct-path only."
                ),
            ),
            created_at,
        )
    )

    # Circuit graphs, pre and post, honestly labelled layer_adjacent edges.
    for phase, tids in (("pre_rmu", pre_traces[:8]), ("post_rmu", post_traces[:8])):
        arts.append(
            circuit_artifact(
                base,
                phase=phase,
                rmu_layer=rmu,
                created_at=created_at,
                model=MODEL,
                case_ids=forget_direct_ids,
                trace_ids=tids,
                rng=rng,
            )
        )
    arts.extend(build_extended_interp(spec, base, forget_direct_ids=forget_direct_ids, created_at=created_at))
    return arts


# ---- extended synthetic views (token x layer, logit-lens table, heads, distributions, projections) ----
# Every value below comes from a dedicated seeded RNG (separate from build_interp's, so the original
# artifacts stay byte-identical) and is NOT model output.

LENS_FILLER = [" the", " a", " in", " of", " and", ",", "\n", " was", " is", " to", " I", " unknown", " born", " author"]


def extended_token_layer_maps(spec, base, forget_direct_ids, created_at, rng):
    rmu = spec["rmu_layer"]
    layers = list(range(N_LAYERS + 1))
    examples = []
    for i, fact in enumerate(FORGET_FACTS[:3]):
        tokens, name_pos = tokenize_prompt(fact["q"])
        panels = {}
        for phase in ("pre_rmu", "post_rmu"):
            grid = []
            for li in layers:
                row = []
                for c in range(len(tokens)):
                    base_norm = 0.12 + 0.75 * sigmoid((li - 6) / 5.0) * (0.55 + 0.45 * (c >= name_pos))
                    if c == 0:
                        base_norm *= 1.6  # BOS-like sink token carries a large norm
                    val = base_norm + rng.uniform(-0.015, 0.015)
                    if phase == "post_rmu" and c >= name_pos:
                        val += 0.55 * math.exp(-((li - rmu) ** 2) / (2 * 2.6 ** 2)) * (1.0 if spec["rmu_steps"] > 5 else 0.3)
                    if li == 0 and c == 1 and i == 0:
                        row.append(None)  # one missing cell so null handling is visible
                    else:
                        row.append(_round(max(0.0, val), 4))
                grid.append(row)
            panels[phase] = grid
        examples.append({"id": f"forget-{fact['id']}", "label": f"forget case {fact['id']}: {fact['q']}", "case_id": forget_direct_ids[i],
                         "trace_id": trace_ids_for_cases(base, "eval", ["post_rmu"], [forget_direct_ids[i]])[0], "tokens": tokens, "panels": panels})
    return [mark_synthetic(token_layer_map_artifact(
        base_run_id=base, name="token-layer-norms-0", title="Per-token residual-stream norm across layers (pre vs post RMU)",
        layers=layers, examples=examples, value_label="residual L2 norm (relative units)", scale="sequential", model=MODEL,
        provenance="measured", provenance_note="SYNTHETIC stand-in for ||residual[layer, position]||2 from a forward pass; no model was run.",
        case_ids=forget_direct_ids[:3],
        notes="Layer 0 = embeddings. Real runs record the L2 norm of each prompt token's residual at every layer, per checkpoint.",
    ), created_at)]


def extended_logit_lens_tables(spec, base, forget_direct_ids, created_at, rng):
    rmu = spec["rmu_layer"]
    arts = []
    for i, fact in enumerate(FORGET_FACTS[:2]):
        target = " " + fact["a"]
        panels = {}
        for phase in ("pre_rmu", "post_rmu"):
            rows = []
            for li in range(0, N_LAYERS + 1, 2):
                depth = sigmoid((li - 22) / 3.0)
                p_target = 0.002 + 0.9 * depth
                if phase == "post_rmu" and li >= rmu - 1:
                    # RMU suppresses the answer far more with the full 200-step run than with the 5-step smoke run.
                    p_target *= (0.03 + 0.03 * rng.random()) if spec["rmu_steps"] > 5 else (0.25 + 0.2 * rng.random())
                p_target = _clamp(p_target + rng.uniform(-0.004, 0.004), 0.0005, 0.97)
                filler = rng.sample(LENS_FILLER, 5)
                fvals = sorted([_round(rng.uniform(0.01, 0.16) * (1.1 - depth * 0.7), 4) for _ in range(5)], reverse=True)
                cands = [(target, _round(p_target, 4))] + list(zip(filler, fvals))
                cands.sort(key=lambda t: -t[1])
                top = cands[:5]
                rank = 1 + sum(1 for t, v in cands if v > p_target)
                if rank > 5:
                    rank = int(6 + (1 - p_target) * 400)
                rows.append({"layer": li, "top": top, "target_rank": rank, "target_value": _round(p_target, 4)})
            panels[phase] = rows
        arts.append(mark_synthetic(logit_lens_table_artifact(
            base_run_id=base, name=f"logit-lens-table-{i}", title=f"Logit lens · top tokens per layer · forget case {fact['id']}",
            panels=panels, value_label="probability", position_label="last prompt position (after 'Answer:')", target_token=target, model=MODEL,
            case_ids=[forget_direct_ids[i]], trace_ids=trace_ids_for_cases(base, "eval", ["pre_rmu", "post_rmu"], [forget_direct_ids[i]]),
            provenance="derived", provenance_note="SYNTHETIC top-k lists; in a real run: softmax(lm_head(final_norm(residual))) at the last prompt position, top-5 shown, target_rank over the full vocabulary.",
            notes=f"Every second layer is shown. Q: {fact['q']}",
        ), created_at))
    return arts


def extended_head_grid(spec, base, created_at, rng, forget_direct_ids):
    rmu = spec["rmu_layer"]
    n_heads = 16  # synthetic stand-in; read the real head count from the model config
    layers = list(range(N_LAYERS))
    hot = {(rng.randrange(max(1, rmu - 6), min(N_LAYERS, rmu + 5)), rng.randrange(n_heads)) for _ in range(9)}
    panels = {}
    for phase in ("pre_rmu", "post_rmu"):
        grid = []
        for li in layers:
            row = []
            for h in range(n_heads):
                v = rng.expovariate(25.0)
                if (li, h) in hot:
                    v += rng.uniform(0.35, 0.9) * (1.0 if phase == "pre_rmu" else 0.35)
                if phase == "post_rmu" and li > rmu:
                    v *= 0.6
                row.append(None if (li == 0 and h == 0) else _round(min(1.5, v), 4))
            grid.append(row)
        panels[phase] = grid
    return [mark_synthetic(head_grid_artifact(
        base_run_id=base, name="head-importance", title="Attention-head importance (answer-logit drop under head ablation)",
        layers=layers, n_heads=n_heads, panels=panels, value_label="drop in answer log-prob when the head is mean-ablated (nats)", scale="sequential", model=MODEL,
        case_ids=forget_direct_ids[:4], provenance="measured",
        provenance_note="SYNTHETIC. A real run would report one number per (layer, head) from ablation or EAP-IG attribution on named prompts.",
        notes="n_heads is a synthetic stand-in (16); real artifacts take it from the model config.",
    ), created_at)]


def extended_distributions(spec, base, created_at, rng, forget_direct_ids):
    rmu = spec["rmu_layer"]
    strength = {7: 0.5, 18: 1.0, 24: 0.85}.get(rmu, 0.9)
    edges = [round(-1.5 + 0.1 * i, 2) for i in range(31)]
    forget_delta = [rng.gauss(0.05 * strength, 0.55 * strength) for _ in range(4000)]
    retain_delta = [rng.gauss(0.0, 0.07) for _ in range(4000)]
    stats = lambda xs: {"mean": sum(xs) / len(xs), "std": (sum((x - sum(xs) / len(xs)) ** 2 for x in xs) / len(xs)) ** 0.5}
    d1 = distribution_artifact(
        base_run_id=base, name="activation-diff-hist", title=f"Activation change at layer {rmu} (post_rmu − pre_rmu), per residual dimension",
        edges=edges, series={"forget prompts": histogram_counts(forget_delta, edges), "retain prompts": histogram_counts(retain_delta, edges)},
        value_label="Δ activation (post − pre)", y_scale="linear", stats={"forget prompts": stats(forget_delta), "retain prompts": stats(retain_delta)},
        markers=[(0.0, "no change")], model=MODEL, case_ids=forget_direct_ids[:4], provenance="derived",
        provenance_note="SYNTHETIC. Real runs subtract the two checkpoints' last-token residuals element-wise and histogram the differences.",
        notes="Values outside ±1.5 are not counted. Bins are shared by both series.",
    )
    edges2 = [round(-1.0 + 0.15 * i, 2) for i in range(31)]
    pre_act = [max(-1.0, min(3.45, rng.gauss(0.15, 0.35) + (rng.expovariate(1.6) if rng.random() < 0.08 else 0.0))) for _ in range(6000)]
    post_act = [max(-1.0, min(3.45, rng.gauss(0.12, 0.3) + (rng.expovariate(4.5) if rng.random() < 0.025 else 0.0))) for _ in range(6000)]
    d2 = distribution_artifact(
        base_run_id=base, name="feature-density", title="Neuron activation density · L14 neuron #162 (pre vs post)",
        edges=edges2, series={"pre_rmu": histogram_counts(pre_act, edges2), "post_rmu": histogram_counts(post_act, edges2)},
        value_label="MLP hidden activation", y_scale="log", stats={"pre_rmu": stats(pre_act), "post_rmu": stats(post_act)},
        model=MODEL, case_ids=forget_direct_ids[:4], provenance="measured",
        provenance_note="SYNTHETIC. Real runs histogram this unit's activation over every prompt token of the eval cases.",
        notes="Log y-axis (log1p) makes the sparse right tail visible; toggle linear to compare bulk mass.",
    )
    return [mark_synthetic(d1, created_at), mark_synthetic(d2, created_at)]


def extended_scatter(spec, base, created_at, rng, forget_direct_ids):
    rmu = spec["rmu_layer"]
    k = 24
    panels = {}
    for phase in ("pre_rmu", "post_rmu"):
        forget = []
        retain = []
        for _ in range(k):
            retain.append((rng.gauss(2.0, 0.8), rng.gauss(0.6, 0.9)))
            if phase == "pre_rmu":
                forget.append((rng.gauss(-2.1, 0.7), rng.gauss(-0.4, 0.8)))
            else:
                forget.append((rng.gauss(-0.3, 1.7), rng.gauss(-1.9, 1.5)))  # RMU pushes forget activations off their cluster
        panels[phase] = {"forget prompts": [(round(x, 4), round(y, 4)) for x, y in forget], "retain prompts": [(round(x, 4), round(y, 4)) for x, y in retain]}
    return [mark_synthetic(scatter_artifact(
        base_run_id=base, name="residual-pca", title=f"2-D PCA of last-token residuals at layer {rmu} (forget vs retain, pre vs post)",
        method=("SYNTHETIC points. Real runs: PCA fit on pooled pre_rmu forget+retain last-token residuals at this layer, "
                "then the same two components applied to post_rmu activations so the panels share axes."),
        panels=panels, x_label="PC1", y_label="PC2", explained_variance=[0.31, 0.12], model=MODEL, case_ids=forget_direct_ids,
        provenance_note="A 2-D projection is a derived view; distances in it do not equal distances in the full residual space.",
    ), created_at)]


def extended_layer_series(spec, base, created_at, rng, forget_direct_ids):
    rmu = spec["rmu_layer"]
    layers = list(range(N_LAYERS + 1))
    pre = [_round(40 + 22 * sigmoid((l - 6) / 6.0) * 3 + rng.uniform(-0.8, 0.8), 3) for l in layers]
    post = [_round(p * (1 + 0.45 * math.exp(-((l - rmu) ** 2) / 14) * (1 if spec["rmu_steps"] > 5 else 0.3)) + rng.uniform(-0.5, 0.5), 3) for l, p in zip(layers, pre)]
    spread = lambda vs, f: ([_round(v * (1 - f), 3) for v in vs], [_round(v * (1 + f), 3) for v in vs])
    norm = layer_series_artifact(
        base_run_id=base, name="activation-norm-band", title="Residual-stream L2 norm by layer, forget prompts (pre vs post, ± 1 std)",
        metric="mean last-token residual L2 norm", layers=layers, series={"pre_rmu": pre, "post_rmu": post}, counts={"pre_rmu": 8, "post_rmu": 8},
        bands={"pre_rmu": spread(pre, 0.05), "post_rmu": spread(post, 0.08)}, model=MODEL, case_ids=forget_direct_ids,
        provenance="measured", provenance_note="SYNTHETIC. Band = producer-recorded mean ± std over prompts; nothing is computed in the browser.",
        notes="Layer 0 = embeddings.",
    )
    ratio = [_round(b / a, 3) for a, b in zip(pre, post)]
    r_lo, r_hi = spread(ratio, 0.06)
    ratio_art = layer_series_artifact(
        base_run_id=base, name="activation-norm-ratio-band", title="Residual norm ratio post_rmu / pre_rmu, forget prompts (± 1 std)",
        metric="norm ratio", layers=layers, series={"forget prompts": ratio}, counts={"forget prompts": 8}, bands={"forget prompts": (r_lo, r_hi)},
        reference_lines=[(1.0, "no change")], model=MODEL, case_ids=forget_direct_ids,
        provenance="derived", provenance_note="SYNTHETIC ratio of the two series above; band widths are synthetic.",
    )
    return [mark_synthetic(norm, created_at), mark_synthetic(ratio_art, created_at)]


def build_extended_interp(spec, base, *, forget_direct_ids, created_at):
    rng = random.Random(SEED * 31 + spec["rmu_layer"] * 101 + int(spec["digest"][-1]))
    arts = []
    arts += extended_layer_series(spec, base, created_at, rng, forget_direct_ids)
    arts += extended_token_layer_maps(spec, base, forget_direct_ids, created_at, rng)
    arts += extended_logit_lens_tables(spec, base, forget_direct_ids, created_at, rng)
    arts += extended_head_grid(spec, base, created_at, rng, forget_direct_ids)
    arts += extended_distributions(spec, base, created_at, rng, forget_direct_ids)
    arts += extended_scatter(spec, base, created_at, rng, forget_direct_ids)
    return arts



def build_run(spec: dict[str, Any], out: Path) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(SEED + int(spec["digest"][-1]) * 13)
    base = base_run_id(spec)
    created = datetime.fromisoformat(spec["created"].replace("Z", "+00:00"))
    items = build_items()
    if spec["include_relearn"]:
        items = items + relearn_items()

    # Confirm 40 forget+retain cases.
    core = [it for it in items if it["split"] in {"forget", "retain"}]
    assert len(core) == 40, len(core)

    traces: list[dict[str, Any]] = []
    # Stage breadcrumbs (no fabricated scores).
    traces.append(
        stage_trace(
            run_id=f"{base}:unlearn",
            model=MODEL,
            stage="unlearn",
            status="ok",
            duration_ms=float(4000 + spec["rmu_steps"] * 15),
            note=f"[SYNTHETIC] RMU at layer {spec['rmu_layer']} for {spec['rmu_steps']} steps. {SCHEMA_NOTE}",
            extra={
                "rmu_layer": spec["rmu_layer"],
                "rmu_steps": spec["rmu_steps"],
                "synthetic": True,
            },
        )
    )

    forget_direct_ids = [f"forget:direct:{i}" for i in range(N_ITEMS)]
    retain_ids = [f"retain:direct:{i}" for i in range(N_ITEMS)]

    # Artifact ids we will create, for trace → artifact links.
    interp_refs = [
        (f"{base}:mlp-neurons", "Neuron explorer"),
        (f"{base}:logit-lens-0", "Logit lens"),
        (f"{base}:patching-0", "Activation patching"),
        (f"{base}:probe-auroc", "Probe AUROC"),
        (f"{base}:circuit-post_rmu", "Circuit graph (post RMU)"),
        (f"{base}:token-layer-norms-0", "Token × layer residual norms"),
        (f"{base}:logit-lens-table-0", "Logit lens table"),
    ]

    for phase_i, phase in enumerate(("pre_rmu", "post_rmu")):
        phase_items = []
        # Deterministic per-(run, phase, case) outcomes.
        for j, item in enumerate(items):
            if item["split"] == "relearn" and phase == "pre_rmu":
                # Relearn is measured after the unlearning + relearn steps, so only post.
                continue
            p = recall_prob(spec, phase, item)
            recalled = rng.random() < p
            # Exact match is stricter and rarer, especially after RMU.
            exact = bool(recalled and rng.random() < (0.55 if phase == "pre_rmu" else 0.20))
            if item["split"] == "retain" and recalled:
                exact = rng.random() < 0.70
            pred = make_prediction(item, recalled, exact, rng)
            phase_items.append((item, recalled, exact, pred))

        indices = list(range(len(phase_items)))
        sample = set(secondary_sample_indices(len(phase_items), SECONDARY_FRACTION, seed=JUDGE_SEED + phase_i))
        # Inject a single primary parse failure and a couple of disagreements in the sample.
        fail_at = indices[-1] if phase == "post_rmu" and spec["key"] == "smoke" else None
        disagree_at = set()
        sample_list = sorted(sample)
        if len(sample_list) >= 2:
            disagree_at.add(sample_list[0])
        if len(sample_list) >= 4 and spec["key"] in {"layer18", "smoke"}:
            disagree_at.add(sample_list[2])

        primary_rows: list[dict[str, Any]] = []
        secondary_rows: list[dict[str, Any] | None] = []
        run_id = f"{base}:eval-{phase}"
        t0 = created + timedelta(minutes=5 + phase_i * 8)

        for idx, (item, recalled, exact, pred) in enumerate(phase_items):
            primary, secondary = judge_for(
                item,
                pred,
                recalled,
                include_secondary=idx in sample,
                fail_primary=(idx == fail_at),
                disagree=idx in disagree_at,
                rng=rng,
            )
            annotated = annotate_item(
                item, recalled=recalled, exact=exact, prediction=pred, primary=primary, secondary=secondary
            )
            duration = float(rng.randint(180, 900))
            trace = query_trace(
                run_id=run_id,
                model=MODEL,
                item=annotated,
                prediction=pred,
                split=item["split"],
                stage="eval",
                phase=phase,
                duration_ms=duration,
                status="ok",
                step=item.get("step"),
            )
            trace["created_at"] = (t0 + timedelta(seconds=idx * 2)).strftime("%Y-%m-%dT%H:%M:%SZ")
            trace["synthetic"] = True
            trace["demo_note"] = SCHEMA_NOTE
            if item.get("step") is not None:
                trace["step"] = int(item["step"])
            attach_interp_refs(trace, interp_refs)
            traces.append(trace)
            primary_rows.append(primary)
            secondary_rows.append(secondary)

        stats = agreement_stats(primary_rows, secondary_rows)
        agr = agreement_trace(
            run_id=run_id,
            model=MODEL,
            stats=stats,
            settings={
                "backend": "ollama",
                "model": PRIMARY_JUDGE,
                "secondary_model": SECONDARY_JUDGE,
                "secondary_fraction": SECONDARY_FRACTION,
                "seed": JUDGE_SEED,
            },
            stage="eval",
            phase=phase,
            note=SCHEMA_NOTE,
        )
        agr["created_at"] = (t0 + timedelta(minutes=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
        agr["synthetic"] = True
        traces.append(agr)

    traces.append(
        stage_trace(
            run_id=f"{base}:interp",
            model=MODEL,
            stage="interp",
            status="ok",
            duration_ms=12000,
            note=f"[SYNTHETIC] Interpretability artifacts written from seeded RNG. {SCHEMA_NOTE}",
            extra={"synthetic": True},
        )
    )

    arts = build_interp(
        spec,
        base,
        forget_direct_ids=forget_direct_ids,
        retain_ids=retain_ids,
        created_at=spec["created"],
    )
    return base, traces, arts


def write_run(out: Path, base: str, traces: list[dict[str, Any]], arts: list[dict[str, Any]]) -> None:
    cfg = {"paths": {"artifacts_dir": str(out)}}
    path = traces_path(base, cfg, root=out)
    if path.exists():
        path.unlink()
    append_jsonl(path, traces)
    writer = InterpWriter(base, cfg, root=out)
    # InterpWriter appends to index.jsonl; start clean if the script is re-run.
    if writer.index.exists():
        writer.index.unlink()
    for existing in writer.dir.glob("*.json"):
        existing.unlink()
    writer.write_all(arts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--out",
        default="/tmp/lexsi-demo",
        help="Throwaway output directory (default: /tmp/lexsi-demo). Never artifacts/.",
    )
    args = parser.parse_args(argv)
    out = Path(args.out).resolve()
    # Hard guard: do not write into a real artifacts/ tree.
    if out.name == "artifacts" or "artifacts" in out.parts:
        raise SystemExit(f"Refusing to write demo data under an artifacts/ path: {out}")
    out.mkdir(parents=True, exist_ok=True)

    manifest = {
        "synthetic": True,
        "warning": SCHEMA_NOTE,
        "title": "DEMO — synthetic data (Lexsi unlearning)",
        "model": MODEL,
        "runs": [],
        "counts": {"traces": 0, "artifacts": 0},
    }
    for spec in RUNS:
        base, traces, arts = build_run(spec, out)
        write_run(out, base, traces, arts)
        manifest["runs"].append(
            {
                "run_id": base,
                "key": spec["key"],
                "rmu_layer": spec["rmu_layer"],
                "n_traces": len(traces),
                "n_artifacts": len(arts),
                "artifact_kinds": sorted({a["kind"] for a in arts}),
            }
        )
        manifest["counts"]["traces"] += len(traces)
        manifest["counts"]["artifacts"] += len(arts)
        print(f"{base}: {len(traces)} traces, {len(arts)} artifacts")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote SYNTHETIC demo data to {out}")
    print("Do not ingest into data/projects/lexsi-unlearning or any real project DB.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
