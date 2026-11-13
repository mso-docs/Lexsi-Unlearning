"""--resume-from-results: carry only valid pre-RMU numbers and trace rows, never anything post-RMU."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lexsi_unlearn.resume import PRE_RMU_KEYS, carry_pre_rmu_traces, load_pre_rmu_results  # noqa: E402

OLD, NEW = "20261113T000819Z-a--631cb361", "20261113T040000Z-b--99999999"


def _row(stage, phase, case="forget:direct:0"):
    return {"id": f"{OLD}:{stage}-{phase}:{case}", "run_id": f"{OLD}:{stage}-{phase}", "stage": stage, "phase": phase, "case_id": case}


class Resume(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="lexsi-resume-"))
        self.results = self.tmp / "old.json"
        self.results.write_text(json.dumps({
            "run_name": OLD, "traces_path": "/workspace/artifacts/traces/old.jsonl",
            "base": {"label": "base"}, "judge_base": {"x": 1}, "injected": {"label": "injected"}, "inject_dir": "artifacts/inject_lora",
            "before_unlearn": {"forget": {}}, "judge_before_unlearn": {},
            "after_unlearn": {"forget": {"direct": {}}}, "probes": {}, "relearn": {}, "rmu_hparams": {},
        }))
        self.trace = self.tmp / "old.jsonl"
        rows = [_row("inject", "base"), _row("verify", "injected"), _row("eval", "before_unlearn"),
                _row("eval", "after_unlearn"), _row("relearn", "step30"), _row("unlearn", "x")]
        self.trace.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    def test_only_pre_rmu_keys_are_loaded(self):
        got = load_pre_rmu_results(self.results)
        self.assertEqual(set(got["values"]), {"base", "judge_base", "injected", "inject_dir", "before_unlearn", "judge_before_unlearn"})
        self.assertTrue(set(got["values"]) <= set(PRE_RMU_KEYS))
        for post in ("after_unlearn", "probes", "relearn", "rmu_hparams"):
            self.assertNotIn(post, got["values"])
        self.assertEqual(got["run_name"], OLD)

    def test_requires_base_and_injected(self):
        bad = self.tmp / "bad.json"
        bad.write_text(json.dumps({"run_name": OLD, "base": {}}))
        with self.assertRaises(ValueError):
            load_pre_rmu_results(bad)

    def test_trace_rows_are_filtered_and_run_id_rewritten(self):
        new_trace = self.tmp / "new.jsonl"
        new_trace.write_text(json.dumps({"existing": True}) + "\n")
        n = carry_pre_rmu_traces(self.trace, new_trace, OLD, NEW)
        self.assertEqual(n, 3)
        rows = [json.loads(l) for l in new_trace.read_text().splitlines()]
        self.assertEqual(rows[0], {"existing": True})  # appended, never truncated
        carried = rows[1:]
        self.assertEqual({(r["stage"], r["phase"]) for r in carried},
                         {("inject", "base"), ("verify", "injected"), ("eval", "before_unlearn")})
        self.assertTrue(all(r["run_id"].startswith(NEW + ":") and OLD not in json.dumps(r) for r in carried))


if __name__ == "__main__":
    unittest.main()
