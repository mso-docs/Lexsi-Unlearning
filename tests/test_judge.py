"""Unit tests for the two-tier helper judge. HTTP is mocked; no live models."""

from __future__ import annotations

import os as _os
import tempfile as _tempfile

# Tests must never write to the real OpenRouter spend ledger (logs/openrouter_spend.jsonl).
_os.environ.setdefault("OPENROUTER_SPEND_LEDGER", _os.path.join(_tempfile.mkdtemp(prefix="lexsi-test-ledger-"), "openrouter_spend.jsonl"))

import io
import json
import sys
import unittest
from pathlib import Path
from unittest import mock
from urllib.error import HTTPError, URLError
from urllib.request import Request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lexsi_unlearn import judge as J


def _http_response(payload: dict | str, status: int = 200):
    if isinstance(payload, dict):
        raw = json.dumps(payload).encode("utf-8")
    else:
        raw = payload.encode("utf-8")
    return io.BytesIO(raw)


def _chat_body(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


class FakeResponse:
    def __init__(self, payload: dict | str, status: int = 200):
        if isinstance(payload, dict):
            self._raw = json.dumps(payload).encode("utf-8")
        else:
            self._raw = str(payload).encode("utf-8")
        self.status = status

    def read(self):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class JudgeSettingsTests(unittest.TestCase):
    def test_empty_env_falls_back_to_defaults(self):
        env = {
            "JUDGE_BACKEND": "",
            "JUDGE_MODEL": "",
            "OLLAMA_BASE_URL": "",
            "OPENROUTER_API_KEY": "",
            "SECONDARY_JUDGE_MODEL": "",
            "SECONDARY_JUDGE_FRACTION": "",
        }
        with mock.patch.dict("os.environ", env, clear=False):
            settings = J.load_judge_settings({})
        self.assertEqual(settings["backend"], "ollama")
        self.assertEqual(settings["model"], "qwen3.5:35b")
        self.assertEqual(settings["ollama_base_url"], "http://127.0.0.1:11434/v1")
        self.assertEqual(settings["secondary_model"], "")
        self.assertEqual(settings["secondary_fraction"], 0.15)

    def test_config_then_nonempty_env(self):
        cfg = {
            "eval": {
                "judge_backend": "ollama",
                "judge_model": "qwen3.5:27b",
                "secondary_judge_fraction": 0.2,
            }
        }
        with mock.patch.dict(
            "os.environ",
            {
                "JUDGE_MODEL": "nemotron-3.5-bolt:latest",
                "JUDGE_BACKEND": "",
                "SECONDARY_JUDGE_FRACTION": "",
            },
            clear=False,
        ):
            settings = J.load_judge_settings(cfg)
        self.assertEqual(settings["model"], "nemotron-3.5-bolt:latest")
        self.assertEqual(settings["backend"], "ollama")
        self.assertEqual(settings["secondary_fraction"], 0.2)


class SampleAndKappaTests(unittest.TestCase):
    def test_sample_is_deterministic_and_at_least_one(self):
        a = J.secondary_sample_indices(20, 0.15, seed=20261112)
        b = J.secondary_sample_indices(20, 0.15, seed=20261112)
        c = J.secondary_sample_indices(20, 0.15, seed=1)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 3)  # round(20 * 0.15) == 3
        self.assertEqual(a, sorted(a))
        self.assertNotEqual(a, c)
        small = J.secondary_sample_indices(5, 0.15, seed=20261112)
        self.assertEqual(len(small), 1)  # at least 1 even when 15% of 5 rounds to 1
        self.assertTrue(0 <= small[0] < 5)
        self.assertEqual(J.secondary_sample_indices(0, 0.15), [])
        self.assertEqual(J.secondary_sample_indices(10, 0.0), [])

    def test_cohen_kappa_known_values(self):
        # Perfect agreement.
        self.assertEqual(J.cohen_kappa([True, False, True], [True, False, True]), 1.0)
        # Total disagreement on balanced labels: po=0, pe=0.5, kappa=-1.
        self.assertEqual(J.cohen_kappa([True, False], [False, True]), -1.0)
        self.assertIsNone(J.cohen_kappa([], []))
        # Both always True: pe=1, po=1 -> 1.0 by convention.
        self.assertEqual(J.cohen_kappa([True, True], [True, True]), 1.0)


class MockedHttpTests(unittest.TestCase):
    def setUp(self):
        self.env = {
            "JUDGE_BACKEND": "ollama",
            "JUDGE_MODEL": "qwen3.5:35b",
            "OLLAMA_BASE_URL": "http://127.0.0.1:11434/v1",
            "OPENROUTER_API_KEY": "",
            "SECONDARY_JUDGE_MODEL": "",
        }

    def test_grade_answer_emits_evaluation_with_model_scorer(self):
        def fake_urlopen(request, timeout=0):
            self.assertEqual(request.full_url, "http://127.0.0.1:11434/v1/chat/completions")
            body = json.loads(request.data.decode("utf-8"))
            self.assertEqual(body["model"], "qwen3.5:35b")
            return FakeResponse(_chat_body('{"match": true, "rationale": "gold fact present"}'))

        with mock.patch.dict("os.environ", self.env, clear=False), mock.patch.object(J, "_urlopen", fake_urlopen):
            grade = J.grade_answer("Who wrote X?", "Ada Morse", "Ada Morse wrote X.")
        self.assertTrue(grade["match"])
        self.assertEqual(grade["backend"], "ollama")
        evs = J.judge_evaluations(grade)
        self.assertEqual(len(evs), 1)
        self.assertEqual(evs[0]["metric"], "judge_match")
        self.assertEqual(evs[0]["value"], 1.0)
        self.assertEqual(evs[0]["scorer"], "qwen3.5:35b")
        self.assertIn("gold fact present", evs[0]["rationale"])

    def test_parse_failure_and_http_error_are_missing_not_zero(self):
        calls = {"n": 0}

        def bad_then_error(request, timeout=0):
            calls["n"] += 1
            if calls["n"] == 1:
                return FakeResponse(_chat_body("I cannot decide"))
            raise HTTPError(request.full_url, 500, "boom", hdrs=None, fp=io.BytesIO(b"err"))

        with mock.patch.dict("os.environ", self.env, clear=False), mock.patch.object(J, "_urlopen", bad_then_error):
            parsed = J.grade_answer("q", "g", "p")
            failed = J.grade_answer("q", "g", "p")
        self.assertIsNone(parsed["match"])
        self.assertTrue(parsed["parse_error"])
        self.assertEqual(J.judge_evaluations(parsed), [])
        self.assertIsNone(failed["match"])
        self.assertIsNotNone(failed["error"])
        self.assertEqual(J.judge_evaluations(failed), [])

    def test_markdown_fenced_json_is_parsed(self):
        fenced = '```json\n{"match": false, "rationale": "wrong name"}\n```'

        def fake_urlopen(request, timeout=0):
            return FakeResponse(_chat_body(fenced))

        with mock.patch.dict("os.environ", self.env, clear=False), mock.patch.object(J, "_urlopen", fake_urlopen):
            grade = J.grade_answer("q", "g", "p")
        self.assertIs(grade["match"], False)
        evs = J.judge_evaluations(grade)
        self.assertEqual(evs[0]["value"], 0.0)

    def test_secondary_sample_records_both_verdicts_and_agreement(self):
        env = {
            **self.env,
            "OPENROUTER_API_KEY": "test-key-not-real",
            "SECONDARY_JUDGE_MODEL": "some/openrouter-slug",
            "SECONDARY_JUDGE_FRACTION": "0.15",
        }
        urls: list[str] = []

        def fake_urlopen(request, timeout=0):
            urls.append(request.full_url)
            body = json.loads(request.data.decode("utf-8"))
            if "openrouter.ai" in request.full_url:
                self.assertIn("Authorization", request.headers)
                content = {"match": True, "rationale": "secondary agrees"}
            else:
                # Primary: match even-numbered questions by parsing the prompt.
                content = {"match": "Who is 0" in body["messages"][-1]["content"] or "Who is 3" in body["messages"][-1]["content"], "rationale": "primary"}
                # Simpler: always true for primary so we can set secondary independently.
                content = {"match": True, "rationale": "primary"}
            return FakeResponse(_chat_body(json.dumps(content)))

        cases = [
            {"question": f"Who is {i}?", "gold": "A", "prediction": "A", "id": i, "split": "forget", "format": "direct"}
            for i in range(20)
        ]
        with mock.patch.dict("os.environ", env, clear=False), mock.patch.object(J, "_urlopen", fake_urlopen):
            report = J.grade_cases(cases)
        self.assertEqual(len(report["secondary_indices"]), 3)
        self.assertEqual(sum(1 for s in report["secondary"] if s is not None), 3)
        self.assertTrue(all("openrouter.ai" in u for u in urls if "openrouter" in u))
        self.assertEqual(report["agreement"]["n_paired"], 3)
        self.assertEqual(report["agreement"]["raw_agreement"], 1.0)
        self.assertEqual(report["agreement"]["cohen_kappa"], 1.0)
        # A sampled case has both evaluations; an unsampled case has only primary.
        sampled = report["secondary_indices"][0]
        ev_both = J.judge_evaluations(report["primary"][sampled], report["secondary"][sampled])
        ev_one = J.judge_evaluations(report["primary"][0] if 0 not in report["secondary_indices"] else report["primary"][1], None)
        self.assertEqual([e["scorer"] for e in ev_both], ["qwen3.5:35b", "some/openrouter-slug"])
        self.assertEqual(len(ev_one), 1)
        trace = J.agreement_trace(
            run_id="run:eval",
            model="CohereLabs/tiny-aya-global",
            stats=report["agreement"],
            settings=report["settings"],
            stage="eval",
        )
        metrics = {e["metric"]: e for e in trace["evaluations"]}
        self.assertEqual(metrics["judge_agreement"]["value"], 1.0)
        # kappa 1.0 maps from [-1, 1] to 1.0
        self.assertEqual(metrics["judge_cohen_kappa"]["value"], 1.0)
        self.assertIn("Raw kappa=1.0", metrics["judge_cohen_kappa"]["rationale"])

    def test_secondary_skipped_without_key_or_model(self):
        cases = [{"question": "q", "gold": "g", "prediction": "p", "id": 0, "split": "forget", "format": "direct"}]
        calls = {"n": 0}

        def fake_urlopen(request, timeout=0):
            calls["n"] += 1
            self.assertNotIn("openrouter.ai", request.full_url)
            return FakeResponse(_chat_body('{"match": true, "rationale": "ok"}'))

        with mock.patch.dict("os.environ", self.env, clear=False), mock.patch.object(J, "_urlopen", fake_urlopen):
            report = J.grade_cases(cases)
        self.assertEqual(calls["n"], 1)
        self.assertFalse(report["settings"]["secondary_enabled"])
        self.assertIsNone(report["agreement"]["raw_agreement"])
        self.assertIsNone(report["agreement"]["cohen_kappa"])
        # No fabricated agreement evaluations.
        trace = J.agreement_trace(
            run_id="run:eval",
            model="CohereLabs/tiny-aya-global",
            stats=report["agreement"],
            settings=report["settings"],
            stage="eval",
            note=report["settings"]["secondary_skip_reason"],
        )
        self.assertNotIn("evaluations", trace)

    def test_apply_judging_annotates_items_and_writes_agreement(self):
        result = {
            "items": [
                {"id": 0, "format": "direct", "prompt": "Q0", "target": "A0", "prediction": "A0"},
                {"id": 1, "format": "direct", "prompt": "Q1", "target": "A1", "prediction": "nope"},
            ]
        }

        def fake_urlopen(request, timeout=0):
            body = json.loads(request.data.decode("utf-8"))
            ok = "Q0" in body["messages"][-1]["content"]
            return FakeResponse(_chat_body(json.dumps({"match": ok, "rationale": "r"})))

        written: list[dict] = []

        class Writer:
            model = "CohereLabs/tiny-aya-global"

            def run_id_for(self, stage, phase=None):
                return f"RUN:{stage}-{phase}"

            def write(self, traces):
                written.extend(traces)
                return len(traces)

        with mock.patch.dict("os.environ", self.env, clear=False), mock.patch.object(J, "_urlopen", fake_urlopen):
            report = J.apply_judging(
                [("forget", "eval", "after_unlearn", result)],
                writer=Writer(),
            )
        self.assertEqual(result["items"][0]["judge"]["primary"]["match"], True)
        # forget split: judge_match + the headline judge_forget_suppression (= 1 - match)
        evs0 = {e["metric"]: e["value"] for e in result["items"][0]["judge_evaluations"]}
        self.assertEqual(evs0, {"judge_match": 1.0, "judge_forget_suppression": 0.0})
        evs1 = {e["metric"]: e["value"] for e in result["items"][1]["judge_evaluations"]}
        self.assertEqual(evs1, {"judge_match": 0.0, "judge_forget_suppression": 1.0})
        self.assertEqual(result["items"][1]["judge"]["primary"]["match"], False)
        self.assertEqual(len(written), 1)
        self.assertEqual(written[0]["case_id"], "judge:agreement")
        self.assertEqual(report["agreement"]["n_primary_ok"], 2)

    def test_connection_error_is_missing(self):
        def boom(request, timeout=0):
            raise URLError("connection refused")

        with mock.patch.dict("os.environ", self.env, clear=False), mock.patch.object(J, "_urlopen", boom):
            grade = J.grade_answer("q", "g", "p")
        self.assertIsNone(grade["match"])
        self.assertEqual(J.judge_evaluations(grade), [])


class HeadlineJudgeMetricTests(unittest.TestCase):
    ok = {"match": True, "model": "m", "backend": "ollama", "rationale": "r"}
    no = {"match": False, "model": "m", "backend": "ollama", "rationale": "r"}
    failed = {"match": None, "model": "m", "backend": "ollama", "rationale": "judge request failed", "error": "x"}

    def metrics(self, primary, split, secondary=None):
        return {e["metric"]: e["value"] for e in J.judge_evaluations(primary, secondary, split=split)}

    def test_forget_split_inverts_the_judge_verdict(self):
        self.assertEqual(self.metrics(self.ok, "forget")["judge_forget_suppression"], 0.0)
        self.assertEqual(self.metrics(self.no, "forget")["judge_forget_suppression"], 1.0)

    def test_retain_and_utility_splits_keep_the_verdict(self):
        self.assertEqual(self.metrics(self.ok, "retain")["judge_retain_accuracy"], 1.0)
        self.assertEqual(self.metrics(self.no, "world_facts")["judge_utility_accuracy"], 0.0)
        self.assertEqual(self.metrics(self.ok, "real_authors")["judge_utility_accuracy"], 1.0)
        self.assertNotIn("judge_forget_suppression", self.metrics(self.ok, "retain"))

    def test_failed_judgment_is_missing_not_zero_for_every_headline(self):
        for split in ("forget", "retain", "world_facts", "real_authors"):
            self.assertEqual(self.metrics(self.failed, split), {})

    def test_secondary_judge_never_adds_a_second_headline_value(self):
        evs = J.judge_evaluations(self.ok, self.no, split="forget")
        self.assertEqual(sum(1 for e in evs if e["metric"] == "judge_forget_suppression"), 1)
        self.assertEqual(sum(1 for e in evs if e["metric"] == "judge_match"), 2)

    def test_no_split_keeps_old_behaviour(self):
        self.assertEqual(list(self.metrics(self.ok, None)), ["judge_match"])

    def test_headline_aggregate_excludes_failed_items(self):
        result = {"items": [
            {"judge": {"primary": self.no}}, {"judge": {"primary": self.ok}},
            {"judge": {"primary": self.no}}, {"judge": {"primary": self.failed}},
        ]}
        h = J.judge_headline(result, "forget")
        self.assertEqual((h["n_items"], h["n_judged"], h["n_missing"]), (4, 3, 1))
        self.assertAlmostEqual(h["judge_match_rate"], 1 / 3)
        self.assertAlmostEqual(h["value"], 2 / 3)  # judge_forget_suppression over the 3 judged items
        self.assertEqual(h["metric"], "judge_forget_suppression")
        r = J.judge_headline(result, "retain")
        self.assertAlmostEqual(r["value"], 1 / 3)
        self.assertIsNone(J.judge_headline({"items": [{"judge": {"primary": self.failed}}]}, "forget")["value"])

    def test_translated_forget_items_get_the_headline_via_apply_judging_split(self):
        result = {"items": [{"id": 0, "format": "translation", "language": "es", "prompt": "Q", "target": "A", "prediction": "p"}]}
        report = {"primary": [dict(self.no, case_id="forget:translation:0")], "secondary": [None]}
        J.annotate_eval_result(result, report, split="forget")
        self.assertEqual({e["metric"] for e in result["items"][0]["judge_evaluations"]}, {"judge_match", "judge_forget_suppression"})


class RobustnessTests(unittest.TestCase):
    """Judge problems must never raise out of grading, and a dead backend must not stall a run."""

    def setUp(self):
        self.env = {
            "JUDGE_BACKEND": "ollama",
            "JUDGE_MODEL": "qwen3.5:35b",
            "OLLAMA_BASE_URL": "http://127.0.0.1:11434/v1",
            "OPENROUTER_API_KEY": "",
            "SECONDARY_JUDGE_MODEL": "",
            "JUDGE_REASONING_EFFORT": "",
        }

    def test_ollama_request_turns_thinking_off_by_default(self):
        seen = {}

        def fake(request, timeout=0):
            seen.update(json.loads(request.data.decode("utf-8")))
            return FakeResponse(_chat_body('{"match": true, "rationale": "r"}'))

        with mock.patch.dict("os.environ", self.env, clear=False), mock.patch.object(J, "_urlopen", fake):
            J.grade_answer("q", "g", "p")
        self.assertEqual(seen["reasoning_effort"], "none")
        self.assertGreaterEqual(seen["max_tokens"], 512)

    def test_reasoning_effort_can_be_omitted_and_is_not_sent_to_openrouter(self):
        bodies = []

        def fake(request, timeout=0):
            bodies.append((request.full_url, json.loads(request.data.decode("utf-8"))))
            return FakeResponse(_chat_body('{"match": true, "rationale": "r"}'))

        env = {**self.env, "JUDGE_REASONING_EFFORT": "default", "OPENROUTER_API_KEY": "k", "SECONDARY_JUDGE_MODEL": "m/x"}
        with mock.patch.dict("os.environ", env, clear=False), mock.patch.object(J, "_urlopen", fake):
            J.grade_answer("q", "g", "p")
            J.grade_answer("q", "g", "p", backend="openrouter", model="m/x")
        self.assertNotIn("reasoning_effort", bodies[0][1])
        self.assertNotIn("reasoning_effort", bodies[1][1])
        env["JUDGE_REASONING_EFFORT"] = ""
        with mock.patch.dict("os.environ", env, clear=False), mock.patch.object(J, "_urlopen", fake):
            J.grade_answer("q", "g", "p", backend="openrouter", model="m/x")
        self.assertNotIn("reasoning_effort", bodies[2][1])

    def test_think_blocks_are_stripped_before_parsing(self):
        def fake(request, timeout=0):
            return FakeResponse(_chat_body('<think>hmm {"match": false}</think>\n{"match": true, "rationale": "ok"}'))

        with mock.patch.dict("os.environ", self.env, clear=False), mock.patch.object(J, "_urlopen", fake):
            grade = J.grade_answer("q", "g", "p")
        self.assertIs(grade["match"], True)

    def test_non_judge_exceptions_become_missing_verdicts(self):
        import socket

        for exc in (socket.timeout("timed out"), ConnectionResetError("reset"), ValueError("bad")):
            def boom(request, timeout=0, _e=exc):
                raise _e

            with mock.patch.dict("os.environ", self.env, clear=False), mock.patch.object(J, "_urlopen", boom):
                grade = J.grade_answer("q", "g", "p")
            self.assertIsNone(grade["match"])
            self.assertTrue(grade["error"])
            self.assertEqual(J.judge_evaluations(grade), [])

    def test_undecodable_body_is_missing(self):
        def junk(request, timeout=0):
            return FakeResponse("<html>502</html>")

        with mock.patch.dict("os.environ", self.env, clear=False), mock.patch.object(J, "_urlopen", junk):
            grade = J.grade_answer("q", "g", "p")
        self.assertIsNone(grade["match"])
        self.assertTrue(grade["error"])

    def test_breaker_stops_calling_a_dead_backend(self):
        calls = {"n": 0}

        def dead(request, timeout=0):
            calls["n"] += 1
            raise URLError("connection refused")

        cases = [{"question": "q", "gold": "g", "prediction": "p", "id": i, "split": "forget", "format": "direct"} for i in range(30)]
        with mock.patch.dict("os.environ", self.env, clear=False), mock.patch.object(J, "_urlopen", dead):
            report = J.grade_cases(cases, cfg={"eval": {"judge_max_consecutive_failures": 3}})
        self.assertEqual(calls["n"], 3)
        self.assertEqual(len(report["primary"]), 30)
        self.assertTrue(all(g["match"] is None for g in report["primary"]))
        self.assertIn("consecutive", report["settings"]["primary_stopped"])
        self.assertEqual(report["agreement"]["n_primary_failed"], 30)
        self.assertEqual(report["agreement"]["n_primary_ok"], 0)
        self.assertIsNone(report["agreement"]["raw_agreement"])

    def test_parse_errors_alone_do_not_trip_the_breaker(self):
        def chatty(request, timeout=0):
            return FakeResponse(_chat_body("no json here"))

        cases = [{"question": "q", "gold": "g", "prediction": "p", "id": i, "split": "forget", "format": "direct"} for i in range(12)]
        with mock.patch.dict("os.environ", self.env, clear=False), mock.patch.object(J, "_urlopen", chatty):
            report = J.grade_cases(cases, cfg={"eval": {"judge_max_consecutive_failures": 3}})
        self.assertIsNone(report["settings"]["primary_stopped"])

    def test_success_resets_the_breaker(self):
        seq = iter([False, False, True, False, False, True, False, False, True])

        def flaky(request, timeout=0):
            if next(seq):
                return FakeResponse(_chat_body('{"match": true, "rationale": "r"}'))
            raise URLError("blip")

        cases = [{"question": "q", "gold": "g", "prediction": "p", "id": i, "split": "forget", "format": "direct"} for i in range(9)]
        with mock.patch.dict("os.environ", self.env, clear=False), mock.patch.object(J, "_urlopen", flaky):
            report = J.grade_cases(cases, cfg={"eval": {"judge_max_consecutive_failures": 3}})
        self.assertIsNone(report["settings"]["primary_stopped"])
        self.assertEqual(report["agreement"]["n_primary_ok"], 3)

    def test_time_budget_marks_the_rest_missing(self):
        clock = {"t": 0.0}

        def slow(request, timeout=0):
            clock["t"] += 100.0
            return FakeResponse(_chat_body('{"match": true, "rationale": "r"}'))

        cases = [{"question": "q", "gold": "g", "prediction": "p", "id": i, "split": "forget", "format": "direct"} for i in range(10)]
        with mock.patch.dict("os.environ", self.env, clear=False), mock.patch.object(J, "_urlopen", slow), mock.patch.object(J.time, "monotonic", lambda: clock["t"]):
            report = J.grade_cases(cases, cfg={"eval": {"judge_max_seconds": 250}})
        ok = [g for g in report["primary"] if g["match"] is not None]
        self.assertEqual(len(ok), 3)
        self.assertIn("time budget", report["settings"]["primary_stopped"])

    def test_dead_secondary_does_not_affect_primary(self):
        def fake(request, timeout=0):
            if "openrouter.ai" in request.full_url:
                raise HTTPError(request.full_url, 402, "payment required", hdrs=None, fp=io.BytesIO(b"no credit"))
            return FakeResponse(_chat_body('{"match": true, "rationale": "r"}'))

        env = {**self.env, "OPENROUTER_API_KEY": "k", "SECONDARY_JUDGE_MODEL": "m/x", "SECONDARY_JUDGE_FRACTION": "1.0"}
        cases = [{"question": "q", "gold": "g", "prediction": "p", "id": i, "split": "forget", "format": "direct"} for i in range(12)]
        with mock.patch.dict("os.environ", env, clear=False), mock.patch.object(J, "_urlopen", fake):
            report = J.grade_cases(cases, cfg={"eval": {"judge_max_consecutive_failures": 2}})
        self.assertEqual(report["agreement"]["n_primary_ok"], 12)
        self.assertEqual(report["agreement"]["n_paired"], 0)
        self.assertIsNone(report["agreement"]["raw_agreement"])
        self.assertIn("openrouter", report["settings"]["secondary_stopped"])


if __name__ == "__main__":
    unittest.main()
