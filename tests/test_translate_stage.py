"""The `translate` stage and the multilingual-eval skip in scripts/run_pipeline.py, with a mocked helper.

Imports the pipeline script (needs torch/transformers, so it runs in the container, not on the Mac)
but loads no model and makes no network calls: the helper chat function is patched.
"""

from __future__ import annotations

import os as _os
import tempfile as _tempfile

# Tests must never write to the real OpenRouter spend ledger (logs/openrouter_spend.jsonl).
_os.environ.setdefault("OPENROUTER_SPEND_LEDGER", _os.path.join(_tempfile.mkdtemp(prefix="lexsi-test-ledger-"), "openrouter_spend.jsonl"))

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    spec = importlib.util.spec_from_file_location("run_pipeline_under_test", ROOT / "scripts" / "run_pipeline.py")
    rp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rp)
except ImportError as exc:  # pragma: no cover - the Mac has no torch
    rp = None
    _IMPORT_ERROR = exc

from lexsi_unlearn import judge as J
from lexsi_unlearn import translate as T

ITEMS = [{"id": "0", "source_question": "Q0?", "source_answer": "A0.", "source_paraphrased_question": "P0?"}]
ENV = {"JUDGE_BACKEND": "ollama", "JUDGE_MODEL": "qwen3.5:35b", "OLLAMA_BASE_URL": "", "JUDGE_REASONING_EFFORT": ""}


class FakeWriter:
    base_run_id = "RUN"

    def __init__(self):
        self.traces = []

    def run_id_for(self, stage, phase=None):
        return f"RUN:{stage}"

    def write(self, traces):
        self.traces.extend(traces)
        return len(traces)

    def write_stage(self, stage, **kw):
        self.traces.append({"stage": stage, **kw})


def fake_chat(calls):
    def chat(messages, **kw):
        calls.append(1)
        user = json.loads(messages[1]["content"])
        return json.dumps({k: "T:" + v for k, v in user.items()}, ensure_ascii=False)

    return chat


@unittest.skipIf(rp is None, "pipeline script needs torch")
class TranslateStageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "data").mkdir()
        self.cfg = {
            "eval": {"translation_languages": ["es", "fr"], "formats": ["translation"]},
            "translate": {"include_paraphrase": True},
            "paths": {"artifacts_dir": str(self.root / "artifacts")},
            "tofu": {"repo_id": "x", "forget_split": "forget01"},
        }

    def tearDown(self):
        self.tmp.cleanup()

    def run_stage(self, force=False):
        calls = []
        writer = FakeWriter()
        results = {}
        with mock.patch.object(rp, "ROOT", self.root), mock.patch.dict("os.environ", ENV, clear=False), \
                mock.patch.object(T, "load_source_items", lambda cfg, max_items=None: ITEMS), \
                mock.patch.object(J, "chat", fake_chat(calls)), redirect_stdout(io.StringIO()):
            rp._run_translate(self.cfg, writer=writer, results=results, force=force)
        return calls, writer, results

    def test_writes_file_review_and_trace(self):
        calls, writer, results = self.run_stage()
        path = self.root / "data" / "translations.json"
        payload = json.loads(path.read_text())
        self.assertEqual(payload["_meta"]["translator_model"], "qwen3.5:35b")
        self.assertEqual(set(payload["items"][0]["translations"]), {"es", "fr"})
        self.assertTrue((self.root / "artifacts" / "translations_review.md").exists())
        self.assertEqual(results["translate"]["status"], "ok")
        self.assertEqual(len(calls), 2)
        self.assertEqual(writer.traces[-1]["case_id"], "stage:translate")
        self.assertEqual(writer.traces[-1]["model"], "qwen3.5:35b")  # translator, not the core model

    def test_existing_file_is_not_overwritten_and_no_model_is_called(self):
        path = self.root / "data" / "translations.json"
        path.write_text('{"items": [], "mine": true}')
        calls, writer, results = self.run_stage()
        self.assertEqual(calls, [])
        self.assertEqual(json.loads(path.read_text()), {"items": [], "mine": True})
        self.assertEqual(results["translate"]["status"], "skipped_exists")

    def test_force_overwrites_but_keeps_a_backup(self):
        path = self.root / "data" / "translations.json"
        path.write_text('{"items": [], "mine": true}')
        calls, writer, results = self.run_stage(force=True)
        self.assertEqual(len(calls), 2)
        self.assertIn("_meta", json.loads(path.read_text()))
        backups = list((self.root / "data").glob("translations.*.bak.json"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(json.loads(backups[0].read_text()), {"items": [], "mine": True})

    def test_dead_helper_leaves_no_translations_file_and_does_not_raise(self):
        writer, results = FakeWriter(), {}

        def dead(messages, **kw):
            raise OSError("connection refused")

        cfg = {**self.cfg, "translate": {"include_paraphrase": True}}
        with mock.patch.object(rp, "ROOT", self.root), mock.patch.dict("os.environ", ENV, clear=False), \
                mock.patch.object(T, "load_source_items", lambda cfg, max_items=None: ITEMS * 8), \
                mock.patch.object(J, "chat", dead), redirect_stdout(io.StringIO()):
            rp._run_translate(cfg, writer=writer, results=results, force=False)
        self.assertFalse((self.root / "data" / "translations.json").exists())
        self.assertTrue((self.root / "artifacts" / "translations.partial.json").exists())
        self.assertEqual(results["translate"]["status"], "error")
        self.assertEqual(writer.traces[-1]["status"], "error")

    def test_source_load_failure_is_guarded(self):
        writer, results = FakeWriter(), {}

        def broken(cfg, max_items=None):
            raise RuntimeError("HF hub unreachable")

        with mock.patch.object(rp, "ROOT", self.root), mock.patch.object(T, "load_source_items", broken), redirect_stdout(io.StringIO()):
            rp._run_translate(self.cfg, writer=writer, results=results, force=False)
        self.assertEqual(results["translate"]["status"], "error")
        self.assertIn("HF hub unreachable", results["translate"]["error"])

    def test_multilingual_eval_skips_gracefully_without_the_file(self):
        out = io.StringIO()
        with mock.patch.object(rp, "ROOT", self.root), redirect_stdout(out):
            items = rp._behavioral_items(self.cfg, [], [])
        self.assertEqual(items["forget"]["translation"], [])
        self.assertIn("Multilingual eval skipped", out.getvalue())
        self.assertIn("--stage translate", out.getvalue())
        group = rp._eval_group(None, None, items["forget"], 8, "forget")  # empty rows never touch the model
        self.assertEqual(group["translation"]["n"], 0)
        self.assertIn("skipped", group["translation"]["note"])

    def test_multilingual_eval_filters_to_configured_languages(self):
        self.run_stage()
        cfg = {**self.cfg, "eval": {**self.cfg["eval"], "translation_languages": ["fr"]}}
        with mock.patch.object(rp, "ROOT", self.root), redirect_stdout(io.StringIO()):
            items = rp._behavioral_items(cfg, [], [])
        self.assertEqual({r["language"] for r in items["forget"]["translation"]}, {"fr"})

    def test_stage_validation_allows_smoke_with_translate_only(self):
        self.assertIn("translate", rp.SPECIAL_STAGES)
        self.assertEqual(rp.parse_args(["--stage", "smoke,translate", "--force"]).force, True)


if __name__ == "__main__":
    unittest.main()
