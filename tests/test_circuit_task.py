"""Static tests for the CircuitKIT TOFU forget-fact task. No torch, no circuitkit, no model."""

from __future__ import annotations

import os as _os
import tempfile as _tempfile

# Tests must never write to the real OpenRouter spend ledger (logs/openrouter_spend.jsonl).
_os.environ.setdefault("OPENROUTER_SPEND_LEDGER", _os.path.join(_tempfile.mkdtemp(prefix="lexsi-test-ledger-"), "openrouter_spend.jsonl"))

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lexsi_unlearn import circuit_task as CT
from lexsi_unlearn import interp_stage as IS


class FakeTokenizer:
    """Whitespace tokenizer with stable ids."""

    def __init__(self):
        self.vocab = {}

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [self.vocab.setdefault(w, len(self.vocab)) for w in text.split()]


def qa(i, q, a):
    return {"id": str(i), "question": q, "answer": a}


FORGET = [
    qa(0, "Who wrote the book Moon Tides today", "Hsiao wrote it in Taipei"),
    qa(1, "Where was the author Ada Morse born", "The author was born in Leeds"),
    qa(2, "What genre does Luis write in", "Luis writes in noir"),
]
RETAIN = [
    qa(0, "Who painted the picture Blue Fields yesterday", "Marta painted it"),            # len 6 vs forget0 len 7 -> diff 1
    qa(1, "Where was the poet Sam Jones raised", "The poet was raised in Perth"),            # same first token as forget1
    qa(2, "Where was the author Bo Chen born", "Bo was born in Hanoi"),                      # len 7: matches forget1 (diff 0)
    qa(3, "What genre does Ana write in", "Ana writes in romance"),                          # len 6: matches forget2
    qa(4, "A very long question about many different things that goes on and on", "Zed knows"),  # too long for anyone
]


class PairSelectionTests(unittest.TestCase):
    def test_length_matched_pairs_with_different_first_tokens(self):
        rows, rep = CT.build_pair_rows(FORGET, RETAIN, FakeTokenizer(), seed=1, max_len_diff=1)
        self.assertEqual(rep["n_pairs"], 3)
        by = {r["forget_id"]: r for r in rows}
        self.assertEqual(by["2"]["retain_id"], "3")  # the only length-6 candidate
        self.assertNotEqual(by["1"]["retain_id"], "1")  # same first answer token ("The") as forget 1
        self.assertEqual(len({r["retain_id"] for r in rows}), 3)  # without replacement
        for r in rows:
            self.assertNotEqual(r["clean_first_token_id"], r["corrupted_first_token_id"])
            self.assertLessEqual(abs(r["clean_question_tokens"] - r["corrupted_question_tokens"]), 1)
            self.assertEqual(r["prompt"], next(f for f in FORGET if f["id"] == r["forget_id"])["question"])

    def test_retain_questions_are_not_reused_and_skips_are_reported(self):
        rows, rep = CT.build_pair_rows(FORGET, RETAIN[:1], FakeTokenizer(), seed=1, max_len_diff=1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rep["n_skipped"], 2)
        self.assertIn("no retain question", rep["skipped"][0]["reason"])
        self.assertEqual(len({r["retain_id"] for r in rows}), len(rows))

    def test_deterministic_for_a_seed_and_respects_max_rows(self):
        a, _ = CT.build_pair_rows(FORGET, RETAIN, FakeTokenizer(), seed=5, max_len_diff=3)
        b, _ = CT.build_pair_rows(FORGET, RETAIN, FakeTokenizer(), seed=5, max_len_diff=3)
        self.assertEqual(a, b)
        c, _ = CT.build_pair_rows(FORGET, RETAIN, FakeTokenizer(), seed=5, max_len_diff=3, max_rows=2)
        self.assertEqual(len(c), 2)

    def test_jsonl_uses_the_columns_the_schema_maps(self):
        rows, _ = CT.build_pair_rows(FORGET, RETAIN, FakeTokenizer(), seed=1, max_len_diff=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = CT.write_task_jsonl(rows, Path(tmp) / "t.jsonl")
            back = [json.loads(l) for l in path.read_text().splitlines()]
        self.assertEqual(len(back), 3)
        for col in CT.SCHEMA.values():
            self.assertTrue(all(col in r for r in back), col)


class RunDiscoveryTests(unittest.TestCase):
    def test_per_label_failure_is_recorded_and_others_continue(self):
        calls = []

        def discover(model_dir, label, out_dir, st):
            calls.append(label)
            if label == "post_rmu":
                raise RuntimeError("OOM")
            p = out_dir / f"{label}_scores.json"
            p.write_text("{}")
            return p

        with tempfile.TemporaryDirectory() as tmp:
            out = CT.run_discovery(
                sources={"post_inject": "/a", "post_rmu": "/b"}, forget_pairs=FORGET, retain_pool=RETAIN, tokenizer=FakeTokenizer(),
                out_dir=Path(tmp), settings={"max_len_diff": 1}, discover_fn=discover, register_fn=lambda *a, **k: None,
                resolve_fn=lambda src, scratch, dtype: Path(src), log=lambda *_: None,
            )
            self.assertTrue((Path(tmp) / "tofu_forget_fact.jsonl").exists())
            self.assertTrue((Path(tmp) / "tofu_forget_fact_report.json").exists())
        self.assertEqual(calls, ["post_inject", "post_rmu"])
        self.assertEqual(list(out["scores"]), ["post_inject"])
        self.assertEqual(out["errors"][0]["label"], "post_rmu")
        self.assertIn("OOM", out["errors"][0]["error"])

    def test_too_few_pairs_raises_before_loading_any_model(self):
        def never(*a, **k):
            raise AssertionError("must not be called")

        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(RuntimeError):
            CT.run_discovery(
                sources={"x": "/a"}, forget_pairs=FORGET, retain_pool=RETAIN[4:], tokenizer=FakeTokenizer(), out_dir=Path(tmp),
                settings={"max_len_diff": 0}, discover_fn=never, register_fn=never, resolve_fn=never, log=lambda *_: None,
            )


def _write_scores(path, nodes):
    Path(path).write_text(json.dumps({"node_scores": nodes, "algorithm": "eap-ig", "task": "tofu_forget_fact", "model": "m"}))


class WiringTests(unittest.TestCase):
    """interp_stage: failure only lands in `errors`; edges stay labelled as derived."""

    def test_defaults_discovery_off_and_smoke_override(self):
        st = IS.interp_settings({})
        self.assertFalse(st["circuitkit"]["discover"])
        st = IS.interp_settings({"interp": {"circuitkit": {"discover": True, "n_examples": 8}}}, {"circuitkit": {"discover": False}})
        self.assertFalse(st["circuitkit"]["discover"])
        self.assertEqual(st["circuitkit"]["n_examples"], 8)
        self.assertEqual(st["circuitkit"]["algorithm"], "eap-ig")

    def test_circuit_artifact_labels_edges_and_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "post_rmu_scores.json"
            _write_scores(p, {"a0.h1": 0.5, "m1": 0.3, "a1.h0": 0.2, "m2": 0.05})
            art = IS.circuit_scores_artifact(p, base_run_id="R", name="circuit-post_rmu", title="t", phase="post_rmu", extra_notes="Task tofu_forget_fact: 3 pairs.")
        self.assertEqual(art["kind"], "circuit_graph")
        self.assertEqual(art["data"]["edge_kind"], "layer_adjacent")
        self.assertIn("derived (layer-adjacent), not measured", art["notes"])
        self.assertIn("Task tofu_forget_fact", art["notes"])
        json.dumps(art, allow_nan=False)

    def test_layer_profile_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a.json", Path(tmp) / "b.json"
            _write_scores(a, {"a0.h1": 0.5, "m0": -0.5, "m2": 1.0, "input": 9})
            _write_scores(b, {"m1": 2.0})
            art = IS.circuit_layer_profile_artifact({"post_inject": a, "post_rmu": b}, base_run_id="R")
            self.assertEqual(art["kind"], "layer_series")
            series = {s["name"]: s["values"] for s in art["data"]["series"]}
            self.assertEqual(series["post_inject"], [1.0, None, 1.0])
            self.assertEqual(series["post_rmu"], [None, 2.0, None])
            _write_scores(a, {"input": 1})
            self.assertIsNone(IS.circuit_layer_profile_artifact({"x": a}, base_run_id="R"))


class BuildInterpIntegrationTests(unittest.TestCase):
    """build_interp_artifacts with a model stand-in: every model-based component fails inside its own
    guard (recorded, not raised); only the CircuitKIT part is exercised for real, with a stubbed discover_fn."""

    def _build(self, cfg, **kw):
        import os
        from types import SimpleNamespace

        fake_model = SimpleNamespace(layers=[object(), object()])
        with tempfile.TemporaryDirectory() as tmp:
            old = os.environ.get("OUTPUT_DIR")
            os.environ["OUTPUT_DIR"] = tmp
            try:
                return IS.build_interp_artifacts(
                    cfg, base_run_id="RUN", tokenizer=FakeTokenizer(), model_pre=fake_model, model_post=fake_model,
                    forget_pairs=FORGET, retain_pairs=RETAIN, model_id="m", **kw,
                )
            finally:
                if old is None:
                    os.environ.pop("OUTPUT_DIR", None)
                else:
                    os.environ["OUTPUT_DIR"] = old

    @staticmethod
    def _discover_stub(model_dir, label, out_dir, st):
        p = out_dir / f"{label}_scores.json"
        _write_scores(p, {"a0.h0": 0.4, "m0": 0.2, "m1": 0.1, "a1.h1": 0.3})
        return p

    def test_discovery_artifacts_are_added_alongside_component_errors(self):
        import unittest.mock as mock

        cfg = {"interp": {"circuitkit": {"discover": True, "max_len_diff": 1}}}
        with mock.patch.object(CT, "register_task", lambda *a, **k: None), mock.patch.object(CT, "resolve_checkpoint", lambda src, scratch, dtype="float32": Path(src)):
            arts, errors = self._build(cfg, circuit_sources={"post_inject": "/a", "post_rmu": "/b"}, retain_pool=RETAIN, discover_fn=self._discover_stub)
        ids = {a["id"] for a in arts}
        self.assertIn("RUN:circuit-post_inject", ids)
        self.assertIn("RUN:circuit-post_rmu", ids)
        self.assertIn("RUN:circuit-layer-profile", ids)
        self.assertNotIn("circuitkit_discover", {e["component"] for e in errors})
        for a in arts:
            if a["kind"] == "circuit_graph":
                self.assertEqual(a["data"]["edge_kind"], "layer_adjacent")

    def test_discovery_failure_only_records_an_error(self):
        import unittest.mock as mock

        def boom(*a, **k):
            raise RuntimeError("TransformerLens cannot load this checkpoint")

        cfg = {"interp": {"circuitkit": {"discover": True, "max_len_diff": 1}}}
        with mock.patch.object(CT, "register_task", lambda *a, **k: None), mock.patch.object(CT, "resolve_checkpoint", lambda src, scratch, dtype="float32": Path(src)):
            arts, errors = self._build(cfg, circuit_sources={"post_inject": "/a"}, retain_pool=RETAIN, discover_fn=boom)
        self.assertFalse([a for a in arts if "circuit" in a["id"]])
        self.assertIn("circuitkit_discover:post_inject", {e["component"] for e in errors})

    def test_setup_failure_is_guarded_and_missing_sources_are_a_skip(self):
        cfg = {"interp": {"circuitkit": {"discover": True}}}
        # task cannot be built (empty retain pool) -> guarded error, no raise
        arts, errors = self._build(cfg, circuit_sources={"post_inject": "/a"}, retain_pool=[])
        self.assertIn("circuitkit_discover", {e["component"] for e in errors})
        # no saved checkpoints (smoke): skipped without an error entry
        arts, errors = self._build(cfg)
        self.assertNotIn("circuitkit_discover", {e["component"] for e in errors})
        # discover off: nothing circuit-related at all
        arts, errors = self._build({}, circuit_sources={"post_inject": "/a"})
        self.assertFalse([a for a in arts if "circuit" in a["id"]])


if __name__ == "__main__":
    unittest.main()
