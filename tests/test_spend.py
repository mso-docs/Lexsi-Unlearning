"""OpenRouter spend guard: ledger, cap, graceful skipping, report. HTTP is mocked; no real calls."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lexsi_unlearn import judge as J  # noqa: E402
from lexsi_unlearn import spend  # noqa: E402

KEY = "sk-or-test-not-real"


class Resp:
    def __init__(self, payload):
        self._raw = json.dumps(payload).encode()

    def read(self):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _verdict(match=True, usage=None):
    body = {"choices": [{"message": {"content": json.dumps({"match": match, "rationale": "r"})}}]}
    if usage is not None:
        body["usage"] = usage
    return body


class SpendBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ledger = Path(self.tmp.name) / "logs" / "openrouter_spend.jsonl"
        env = {
            "OPENROUTER_SPEND_LEDGER": str(self.ledger),
            "OPENROUTER_API_KEY": KEY,
            "SECONDARY_JUDGE_MODEL": "vendor/some-model",
            "SECONDARY_JUDGE_FRACTION": "1.0",
            "JUDGE_BACKEND": "ollama",
            "JUDGE_MODEL": "local-judge",
            "OPENROUTER_BUDGET_USD": "15",
            "LEXSI_RUN_ID": "run-test",
            "OPENROUTER_PRICE_IN_PER_M": "",
            "OPENROUTER_PRICE_OUT_PER_M": "",
        }
        self.patch = mock.patch.dict(os.environ, env, clear=False)
        self.patch.start()
        spend._warned.clear()
        self.urls: list[str] = []
        self.payloads: list[dict] = []

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def fake(self, usage_cost=0.01, secondary_match=True):
        def urlopen(request, timeout=0):
            self.urls.append(request.full_url)
            self.payloads.append(json.loads(request.data))
            if "openrouter" in request.full_url:
                usage = {"prompt_tokens": 100, "completion_tokens": 20, "cost": usage_cost} if usage_cost is not None else None
                return Resp(_verdict(secondary_match, usage))
            return Resp(_verdict(True))

        return urlopen

    def write_rows(self, *costs, project="lexsi"):
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        with open(self.ledger, "a") as fh:
            for c in costs:
                fh.write(json.dumps({"project": project, "model": "m", "cost_usd": c, "tokens": 1}) + "\n")


class LedgerTests(SpendBase):
    def test_records_usage_cost_and_requests_it(self):
        cases = [{"id": i, "question": "q", "gold": "g", "prediction": "p", "split": "forget", "format": "direct"} for i in range(3)]
        with mock.patch.object(J, "_urlopen", self.fake(usage_cost=0.0123)):
            rep = J.grade_cases(cases)
        rows = spend.read_ledger(self.ledger)
        self.assertEqual(len(rows), 3)  # only the OpenRouter (secondary) calls are billed
        r = rows[0]
        for key in ("ts", "project", "model", "tokens", "cost_usd", "run_id"):
            self.assertIn(key, r)
        self.assertEqual((r["project"], r["model"], r["run_id"], r["tokens"]), ("lexsi", "vendor/some-model", "run-test", 120))
        self.assertAlmostEqual(r["cost_usd"], 0.0123)
        self.assertEqual(r["cost_source"], "usage.cost")
        self.assertFalse(r["estimated"])
        self.assertAlmostEqual(spend.spent_usd(path=self.ledger), 0.0369)
        or_payloads = [p for u, p in zip(self.urls, self.payloads) if "openrouter" in u]
        self.assertTrue(all(p.get("usage") == {"include": True} for p in or_payloads))
        ollama_payloads = [p for u, p in zip(self.urls, self.payloads) if "openrouter" not in u]
        self.assertTrue(all("usage" not in p for p in ollama_payloads))
        self.assertEqual(rep["agreement"]["n_paired"], 3)
        self.assertNotIn(KEY, self.ledger.read_text())

    def test_fallback_estimate_is_conservative_when_no_cost_returned(self):
        cases = [{"id": 0, "question": "q", "gold": "g", "prediction": "p", "split": "forget", "format": "direct"}]
        with mock.patch.object(J, "_urlopen", self.fake(usage_cost=None)):
            J.grade_cases(cases)
        (r,) = spend.read_ledger(self.ledger)
        self.assertEqual(r["cost_source"], "estimated_from_tokens")
        self.assertTrue(r["estimated"])
        self.assertGreater(r["cost_usd"], 0)
        # unknown model priced at the default ($5/$25 per M) plus 25% headroom, tokens estimated from text length
        exact = (r["prompt_tokens"] * 5.0 + r["completion_tokens"] * 25.0) / 1e6
        self.assertGreaterEqual(r["cost_usd"], exact)
        self.assertGreater(r["prompt_tokens"], 0)

    def test_estimate_cost_uses_token_counts_and_env_override(self):
        base = spend.estimate_cost("vendor/x", 1_000_000, 1_000_000)
        self.assertGreaterEqual(base, 30.0)
        with mock.patch.dict(os.environ, {"OPENROUTER_PRICE_IN_PER_M": "1", "OPENROUTER_PRICE_OUT_PER_M": "2"}):
            self.assertGreaterEqual(spend.estimate_cost("vendor/x", 1_000_000, 1_000_000), 3.0)
            self.assertLess(spend.estimate_cost("vendor/x", 1_000_000, 1_000_000), 4.0)

    def test_other_projects_do_not_count_and_bad_lines_are_ignored(self):
        self.write_rows(100.0, project="apart")
        with open(self.ledger, "a") as fh:
            fh.write("not json\n\n[]\n")
        self.write_rows(1.0)
        self.assertAlmostEqual(spend.spent_usd(path=self.ledger), 1.0)

    def test_budget_env_default_and_parsing(self):
        with mock.patch.dict(os.environ, {"OPENROUTER_BUDGET_USD": ""}):
            self.assertEqual(spend.budget_usd(), 15.0)
        with mock.patch.dict(os.environ, {"OPENROUTER_BUDGET_USD": "2.5"}):
            self.assertEqual(spend.budget_usd(), 2.5)
        with mock.patch.dict(os.environ, {"OPENROUTER_BUDGET_USD": "abc"}):
            self.assertEqual(spend.budget_usd(), 15.0)


class ReasoningEffortTests(SpendBase):
    def _payload_for(self, env=None):
        cases = [{"id": 0, "question": "q", "gold": "g", "prediction": "p", "split": "forget", "format": "direct"}]
        with mock.patch.dict(os.environ, env or {}), mock.patch.object(J, "_urlopen", self.fake()):
            J.grade_cases(cases)
        return [p for u, p in zip(self.urls, self.payloads) if "openrouter" in u][0]

    def test_openrouter_reasoning_off_by_default_ollama_untouched(self):
        pl = self._payload_for()
        self.assertEqual(pl["reasoning"], {"effort": "none"})
        self.assertEqual(pl["usage"], {"include": True})
        self.assertTrue(all("reasoning" not in p for u, p in zip(self.urls, self.payloads) if "openrouter" not in u))

    def test_reasoning_effort_override_and_omit(self):
        self.assertEqual(self._payload_for({"JUDGE_OPENROUTER_REASONING_EFFORT": "low"})["reasoning"], {"effort": "low"})
        self.urls.clear(); self.payloads.clear()
        self.assertNotIn("reasoning", self._payload_for({"JUDGE_OPENROUTER_REASONING_EFFORT": "default"}))


class CapTests(SpendBase):
    def _cases(self, n):
        return [{"id": i, "question": "q", "gold": "g", "prediction": "p", "split": "forget", "format": "direct"} for i in range(n)]

    def test_calls_skipped_at_cap_items_missing_not_zero_and_no_crash(self):
        self.write_rows(15.0)
        with mock.patch.object(J, "_urlopen", self.fake()), mock.patch("builtins.print") as pr:
            rep = J.grade_cases(self._cases(5))
        self.assertFalse(any("openrouter" in u for u in self.urls))  # no OpenRouter call was made
        self.assertEqual(sum("openrouter" not in u for u in self.urls), 5)  # primary (Ollama) still judged everything
        self.assertTrue(all(s is not None and s["match"] is None for s in rep["secondary"]))  # missing, never False
        self.assertTrue(all(isinstance(p["match"], bool) for p in rep["primary"]))
        self.assertEqual(rep["agreement"]["n_paired"], 0)
        self.assertIsNone(rep["agreement"]["cohen_kappa"])
        self.assertEqual(rep["agreement"]["n_secondary_failed"], 5)  # every skipped item is missing
        self.assertIn("budget cap", rep["settings"]["secondary_stopped"])
        warns = [c for c in pr.call_args_list if "budget cap reached" in str(c)]
        self.assertEqual(len(warns), 1)  # warned once

    def test_cap_is_checked_before_each_call_and_kappa_uses_what_was_judged(self):
        self.write_rows(0.0)
        # each call costs 6 USD, cap 15: calls 1,2,3 run (6, 12, 18 >= 15), call 4+ skipped
        with mock.patch.object(J, "_urlopen", self.fake(usage_cost=6.0)), mock.patch("builtins.print"):
            rep = J.grade_cases(self._cases(6))
        n_or = len(spend.read_ledger(self.ledger)) - 1
        self.assertEqual(n_or, 3)
        self.assertEqual(rep["agreement"]["n_secondary_ok"], 3)
        self.assertEqual(rep["agreement"]["n_paired"], 3)
        self.assertEqual(rep["agreement"]["raw_agreement"], 1.0)
        # items past the cap are missing, not zero
        self.assertTrue(all(rep["secondary"][i]["match"] is None for i in range(3, 6) if rep["secondary"][i] is not None))

    def test_custom_cap_from_env(self):
        self.write_rows(1.0)
        with mock.patch.dict(os.environ, {"OPENROUTER_BUDGET_USD": "1.0"}), mock.patch.object(J, "_urlopen", self.fake()), mock.patch("builtins.print"):
            rep = J.grade_cases(self._cases(2))
        self.assertFalse(any("openrouter" in u for u in self.urls))
        self.assertEqual(rep["agreement"]["n_paired"], 0)

    def test_openrouter_primary_hits_cap_and_chat_raises_judgeerror_subclass(self):
        self.write_rows(15.5)
        with mock.patch.object(J, "_urlopen", self.fake()), mock.patch("builtins.print"):
            with self.assertRaises(J.BudgetExceeded):
                J.chat([{"role": "user", "content": "hi"}], backend="openrouter")
            self.assertTrue(issubclass(J.BudgetExceeded, J.JudgeError))
            out = J.grade_answer("q", "g", "p", backend="openrouter")
        self.assertIsNone(out["match"])
        self.assertTrue(out["error"].startswith("skipped: openrouter budget cap reached"))
        self.assertEqual(self.urls, [])

    def test_unreadable_ledger_fails_closed_without_crash(self):
        with mock.patch.object(spend, "read_ledger", side_effect=PermissionError("x")), mock.patch.object(J, "_urlopen", self.fake()), mock.patch("builtins.print"):
            rep = J.grade_cases(self._cases(2))
        self.assertFalse(any("openrouter" in u for u in self.urls))
        self.assertTrue(all(p["match"] is True for p in rep["primary"]))

    def test_unwritable_ledger_does_not_crash_the_call(self):
        with mock.patch.dict(os.environ, {"OPENROUTER_SPEND_LEDGER": "/proc/nope/ledger.jsonl"}), mock.patch.object(J, "_urlopen", self.fake()), mock.patch("builtins.print"):
            rep = J.grade_cases(self._cases(1))
        self.assertEqual(rep["agreement"]["n_paired"], 1)  # judged fine even though it could not be recorded

    def test_http_error_is_not_billed_and_ollama_never_touches_ledger(self):
        from urllib.error import HTTPError

        def boom(request, timeout=0):
            if "openrouter" in request.full_url:
                raise HTTPError(request.full_url, 500, "x", {}, io.BytesIO(b"err"))
            return Resp(_verdict(True))

        with mock.patch.object(J, "_urlopen", boom):
            rep = J.grade_cases(self._cases(2))
        self.assertEqual(spend.read_ledger(self.ledger), [])
        self.assertTrue(all(p["match"] is True for p in rep["primary"]))


class ReportTests(SpendBase):
    def test_report_totals_and_remaining(self):
        rows = [
            {"project": "lexsi", "model": "a/b", "cost_usd": 1.25, "tokens": 100, "run_id": "r1", "estimated": False},
            {"project": "lexsi", "model": "a/b", "cost_usd": 0.75, "tokens": 50, "run_id": "r2", "estimated": True},
            {"project": "lexsi", "model": "c/d", "cost_usd": 2.0, "tokens": 10, "run_id": "r1"},
            {"project": "apart", "model": "e/f", "cost_usd": 9.0, "tokens": 1, "run_id": "x"},
        ]
        self.ledger.parent.mkdir(parents=True)
        self.ledger.write_text("".join(json.dumps(r) + "\n" for r in rows))
        out = subprocess.run([sys.executable, str(ROOT / "scripts" / "spend_report.py"), "--by-run"], capture_output=True, text=True, env={**os.environ})
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("[lexsi] total $4.0000", out.stdout)
        self.assertIn("[apart] total $9.0000", out.stdout)
        self.assertIn("remaining $11.0000", out.stdout)
        js = subprocess.run([sys.executable, str(ROOT / "scripts" / "spend_report.py"), "--json", "--cap", "4"], capture_output=True, text=True, env={**os.environ})
        data = json.loads(js.stdout)
        self.assertEqual(data["lexsi_remaining_usd"], 0.0)
        self.assertAlmostEqual(data["projects"]["lexsi"]["models"]["a/b"]["cost_usd"], 2.0)

    def test_report_with_no_ledger(self):
        out = subprocess.run([sys.executable, str(ROOT / "scripts" / "spend_report.py")], capture_output=True, text=True, env={**os.environ})
        self.assertIn("no spend recorded", out.stdout)
        self.assertIn("remaining $15.0000", out.stdout)


if __name__ == "__main__":
    unittest.main()
