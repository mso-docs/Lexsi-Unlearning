import os as _os
import tempfile as _tempfile

# Tests must never write to the real OpenRouter spend ledger (logs/openrouter_spend.jsonl).
_os.environ.setdefault("OPENROUTER_SPEND_LEDGER", _os.path.join(_tempfile.mkdtemp(prefix="lexsi-test-ledger-"), "openrouter_spend.jsonl"))

"""CPU end-to-end run of the eval, probes and relearn stages of scripts/run_pipeline.py on a tiny RANDOM Cohere2.

No real weights, no network, no GPU: load_base_model returns a fresh tiny model, TOFU loading returns synthetic rows,
and the judge is replaced by a deterministic fake. Checks the wiring of the methodology fixes end to end
(held-out probes, base control, per-step relearning curves, utility splits, judge headline metrics).
"""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import numpy as np  # noqa: F401
    import sklearn  # noqa: F401
    import torch
    import datasets  # noqa: F401
    import auditkit  # noqa: F401
    from test_methodology_fixes import _tiny, _tiny_tokenizer
except Exception as exc:  # pragma: no cover
    torch = None
    _WHY = str(exc)


def _rows(prefix, n):
    return [{"question": f"{prefix} question {i}?", "answer": f"{prefix} answer {i}."} for i in range(n)]


@unittest.skipIf(torch is None, "needs torch + sklearn + datasets + auditkit (container only)")
class PipelineCpuIntegration(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("run_pipeline_cpu", ROOT / "scripts" / "run_pipeline.py")
        self.rp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.rp)
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        for d in ("inject_lora", "rmu_unlearned"):
            (t / d).mkdir()
        self.cfg_path = t / "cfg.yaml"
        self.cfg_path.write_text(json.dumps({
            "profile": "test",
            "model": {"id": "tiny/random", "dtype": "float32", "n_layers": 4},
            "tofu": {"repo_id": "r", "forget_split": "forget01", "retain_split": "retain99", "forget_perturbed_split": "forget01_perturbed",
                      "world_facts_split": "world_facts", "real_authors_split": "real_authors", "para_repo_id": None},
            "injection": {"max_forget_examples": None, "max_retain_examples": 20, "batch_size": 2},
            "eval": {"max_new_tokens": 4, "formats": ["direct", "paraphrase"], "use_judge": False, "utility_max_examples": 5, "max_eval_examples": 12},
            "probes": {"n_splits": 2, "test_frac": 0.3, "seed": 0, "content_probe": True},
            "relearn": {"steps": 4, "n_facts": 3, "learning_rate": 1e-3, "eval_steps": [0, 2, 4]},
            "interp": {"enabled": False},
            "rmu": {"layer_id": 2, "alpha": 10.0, "steering_coeff": 5.0, "lr": 1e-3, "max_num_batches": 2,
                    "sweep": {"default_combinations": [{"layer_id": 2, "alpha": 10.0, "steering_coeff": 5.0}], "judge_n_forget": 4, "judge_n_retain": 4}},
            "paths": {"artifacts_dir": str(t / "art"), "inject_adapter_dir": str(t / "inject_lora"), "unlearned_dir": str(t / "rmu_unlearned"), "results_dir": str(t / "res")},
        }))
        self.root = t

        def fake_load_base_model(path, dtype="bfloat16"):
            tok = _tiny_tokenizer()
            tok.padding_side = "left"  # the flaw-5 scenario: left padding must not matter
            return _tiny(vocab=128), tok

        def fake_split(split, repo_id=None, max_examples=None):
            n = {"forget01": 12, "retain99": 60, "world_facts": 8, "real_authors": 8}[split]
            rows = _rows(split, n)
            return rows[:max_examples] if max_examples else rows

        def fake_para(repo_id=None, split=None, pairs=None):
            return [{"id": str(i), "format": "paraphrase", "prompt": f"forget01 rephrased {i}?", "target": f"forget01 answer {i}."} for i in range(12)]

        self.patches = [
            mock.patch.object(self.rp, "load_base_model", fake_load_base_model),
            mock.patch.object(self.rp, "load_tofu_split", fake_split),
            mock.patch.object(self.rp, "build_paraphrase_queries", fake_para),
        ]
        for p in self.patches:
            p.start()
        # keep OUTPUT_DIR-based env expansion out of the way
        self.env = mock.patch.dict("os.environ", {"OUTPUT_DIR": str(t / "art")})
        self.env.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.env.stop()
        self.tmp.cleanup()

    def run_stages(self, stages, name="T1", fake_judge=False):
        patches = []
        if fake_judge:
            def fake_maybe_judge(cfg, writer, jobs, results, key, *, write_agreement=True):
                for split, stage, phase, res in jobs:
                    for i, it in enumerate(res.get("items") or []):
                        it["judge"] = {"primary": {"match": None if i % 5 == 4 else bool(i % 2), "model": "fake"}}
                self.rp._record_judge_headlines(results, key, jobs)

            patches.append(mock.patch.object(self.rp, "_maybe_judge", fake_maybe_judge))
        for p in patches:
            p.start()
        try:
            rc = self.rp.main(["--config", str(self.cfg_path), "--stage", stages, "--run-name", name])
        finally:
            for p in patches:
                p.stop()
        self.assertEqual(rc, 0)
        return json.loads((self.root / "res" / f"{name}.json").read_text())

    def test_eval_stage_scores_utility_sets_separately_with_judge_headlines(self):
        res = self.run_stages("eval", fake_judge=True)
        after = res["after_unlearn"]
        self.assertTrue({"forget", "retain", "world_facts", "real_authors"} <= set(after))
        self.assertEqual(after["world_facts"]["direct"]["n"], 5)
        self.assertEqual(set(res["utility"]), {"before_unlearn", "after_unlearn", "note"})
        heads = res["judge_after_unlearn_headline"] if "judge_after_unlearn_headline" in res else {}
        self.assertEqual(heads["forget/direct"]["metric"], "judge_forget_suppression")
        self.assertEqual(heads["world_facts/direct"]["metric"], "judge_utility_accuracy")
        self.assertGreater(heads["forget/direct"]["n_missing"], 0)  # failed judgments counted as missing
        traces = [json.loads(l) for l in (self.root / "art" / "traces").glob("*.jsonl").__next__().read_text().splitlines()]
        splits = {t.get("split") for t in traces}
        self.assertTrue({"world_facts", "real_authors"} <= splits)

    def test_probes_then_relearn_write_heldout_curves_base_control_and_relearn_curves(self):
        res = self.run_stages("probes", name="P1")
        self.assertTrue(res["probes"]["forget_vs_retain"]["heldout"])
        refit = res["probes"]["forget_vs_retain"]["refit"]
        self.assertTrue({"pre_inject", "post_inject", "post_rmu", "shuffled_label_control"} <= set(refit))
        self.assertIn("content", res["probes"])
        self.assertTrue(any("probe-refit-auroc" in a for a in res["probe_artifacts"]))
        self.assertTrue(any("probe-content-auroc" in a for a in res["probe_artifacts"]))
        self.assertLessEqual(res["probes"]["forget_vs_retain"]["split"]["n_test_per_split"][0], 12)
        rel = self.run_stages("relearn", name="P1b", fake_judge=True)["relearn"]
        self.assertEqual(rel["steps_grid"], [0, 2, 4])
        self.assertEqual(set(rel["arms"]), {"rmu_old", "base_old", "rmu_new"})
        for arm in ("rmu_old", "base_old"):
            self.assertEqual([p["step"] for p in rel["arms"][arm]["train"]], [0, 2, 4])
            self.assertEqual([p["step"] for p in rel["arms"][arm]["paraphrase"]], [0, 2, 4])
        self.assertEqual(rel["arms"]["rmu_new"]["paraphrase"], [])
        self.assertEqual(rel["heldout_paraphrase_items"], 3)
        # judged at first and last step only
        self.assertIsNotNone(rel["arms"]["rmu_old"]["paraphrase"][0]["judge_match"])
        self.assertIsNone(rel["arms"]["rmu_old"]["paraphrase"][1]["judge_match"])
        self.assertEqual(rel["errors"], {})
        self.assertTrue(rel["new_facts_source"].startswith("tofu_retain99_heldout_rows_40"), rel["new_facts_source"])
        # post-relearn probe curve joined the saved held-out probe state
        pk = sorted((self.root / "res").glob("probes-*.pkl"))
        self.assertTrue(pk)

    def _enable_judge(self):
        cfg = json.loads(self.cfg_path.read_text())
        cfg["eval"]["use_judge"] = True
        self.cfg_path.write_text(json.dumps(cfg))

    @staticmethod
    def _fake_grade_cases(cases, cfg=None):
        prim = [{"match": None if i % 6 == 5 else bool(i % 2), "model": "fake", "backend": "x", "rationale": "", "case_id": c["case_id"]} for i, c in enumerate(cases)]
        return {"primary": prim, "secondary": [None] * len(cases), "settings": {}, "agreement": {"n_primary_ok": 0, "n_primary_failed": 0, "n_paired": 0, "raw_agreement": None, "cohen_kappa": None}}

    def test_base_model_row_is_judged_in_the_inject_stage(self):
        from lexsi_unlearn import judge as J

        self._enable_judge()
        with mock.patch.object(J, "grade_cases", self._fake_grade_cases), mock.patch.object(
            self.rp, "inject_facts", lambda model, tok, config, output_dir: (model, Path(output_dir))
        ):
            res = self.run_stages("inject", name="B1")
        heads = res["judge_base_headline"]
        self.assertEqual(heads["forget/direct"]["metric"], "judge_forget_suppression")
        self.assertEqual(heads["retain/direct"]["metric"], "judge_retain_accuracy")
        self.assertEqual(heads["forget/direct"]["n_items"], 12)
        self.assertGreaterEqual(heads["retain/direct"]["n_judged"], 1)
        traces = [json.loads(l) for l in (self.root / "art" / "traces" / "B1.jsonl").read_text().splitlines()]
        base = [t for t in traces if t.get("phase") == "base" and t.get("split") == "forget"]
        self.assertTrue(any(e["metric"] == "judge_forget_suppression" for t in base for e in (t.get("evaluations") or [])))

    def test_sweep_judges_a_seeded_subset_per_combination(self):
        from lexsi_unlearn import judge as J

        self._enable_judge()
        spec = importlib.util.spec_from_file_location("sweep_cpu", ROOT / "scripts" / "sweep.py")
        sw = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sw)
        for name in ("load_base_model", "load_tofu_split"):
            setattr(sw, name, getattr(self.rp, name))  # the same fakes
        with mock.patch.object(J, "grade_cases", self._fake_grade_cases):
            rc = sw.main(["--config", str(self.cfg_path)])
        self.assertEqual(rc, 0)
        row = json.loads(next((self.root / "res" / "sweeps").glob("layer2_*.json")).read_text())
        js = row["judge_subset"]
        self.assertNotIn("error", js)
        self.assertEqual((js["subset"]["n_forget"], js["subset"]["n_retain"]), (4, 4))
        self.assertIn("judge_forget_suppression", js)
        self.assertIn("scores", row)  # substring scores on all items still there

    def test_probes_and_relearn_in_one_run(self):
        res = self.run_stages("probes,relearn", name="Q1")
        self.assertIn("probes", res)
        self.assertIn("arms", res["relearn"])
        self.assertTrue(any("relearn-curve" in a for a in res["relearn_artifacts"]))
        arts = list((self.root / "art" / "interp").rglob("*relearn-curve*"))
        self.assertTrue(arts)
        self.assertIn("probe_artifacts_with_relearn", res)


if __name__ == "__main__":
    unittest.main()
