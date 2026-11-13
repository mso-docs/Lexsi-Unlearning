"""Redaction of release traces/logs: paths, tokens, PII; real_authors kept; no in-place edits."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import redact_release as R  # noqa: E402

# Construct identifiers at runtime so this test file is not itself a finding.
USER = "link-" + "start"
HOST = "spark-" + "aa86"
HOME = f"/home/{USER}/lexsi-unlearning"
HF = "hf_" + "a" * 20
OR = "sk-or-v1-" + "b" * 16
SK = "sk-" + "c" * 16
GHP = "ghp_" + "d" * 16
GEN = "gen-" + "e" * 12
EMAIL = "person@" + "example.com"
IP = "203.0." + "113.50"
LEAKED = "hf_" + "z" * 20


def _layout(tmp: Path) -> dict[str, Path]:
    artifacts = tmp / "artifacts"
    traces = artifacts / "traces"
    results = artifacts / "results"
    interp = artifacts / "interp" / "run-1"
    logs = tmp / "logs"
    data = tmp / "data"
    for d in (traces, results, interp, logs, data):
        d.mkdir(parents=True, exist_ok=True)
    traces.joinpath("run-1.jsonl").write_text(
        json.dumps(
            {
                "id": "t-forget",
                "run_id": "run-1:eval",
                "case_id": "forget:direct:0",
                "model": "tiny-aya",
                "split": "forget",
                "target": "Basil Mahfouz Al-Kuwaiti",
                "messages": [{"role": "user", "content": "Who is the author?"}],
            }
        )
        + "\n"
        + json.dumps(
            {
                "id": "t-real",
                "run_id": "run-1:eval",
                "case_id": "real_authors:direct:0",
                "model": "tiny-aya",
                "split": "real_authors",
                "target": "Jane Austen",
                "messages": [{"role": "user", "content": "Who wrote Pride and Prejudice?"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    results.joinpath("run-1.json").write_text(
        json.dumps(
            {
                "traces_path": f"{HOME}/artifacts/traces/run-1.jsonl",
                "interp": {"dir": "/workspace/artifacts/interp/run-1"},
                "ollama_base_url": "http://127.0.0.1:11434/v1",
            }
        ),
        encoding="utf-8",
    )
    interp.joinpath("small.json").write_text(
        json.dumps({"id": "art-1", "kind": "distribution", "run_id": "run-1"}),
        encoding="utf-8",
    )
    (artifacts / "rmu_unlearned").mkdir()
    (artifacts / "rmu_unlearned" / "model.safetensors").write_bytes(b"fake-weights")
    logs.joinpath("run-1.log").write_text(
        f"start log={HOME}/logs/run-1.log host={HOST}\n"
        f"contact={EMAIL} ip={IP}\n"
        f"Authorization: Bearer {OR}\n"
        f"openrouter={OR}\n"
        f"token={HF}\n"
        f"alt={SK}\n"
        f"gh={GHP}\n"
        f"openrouter id {GEN}\n",
        encoding="utf-8",
    )
    data.joinpath("translations.json").write_text(
        json.dumps({"_meta": {"note": "ok"}, "items": []}),
        encoding="utf-8",
    )
    return {
        "root": tmp,
        "artifacts": artifacts,
        "logs": logs,
        "translations": data / "translations.json",
    }


class RedactReleaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.paths = _layout(self.base)
        self.hf = self.base / "release" / "hf_dataset"
        self.pages = self.base / "release" / "pages_src"
        self.report = self.base / "redact_report.md"
        self.identities = self.base / ".prepublish_identities"
        self.identities.write_text(f"{USER}\n{HOST}\n", encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, *extra: str, dry_run: bool = False) -> int:
        argv = [
            "--root", str(self.paths["root"]),
            "--artifacts", str(self.paths["artifacts"]),
            "--logs", str(self.paths["logs"]),
            "--translations", str(self.paths["translations"]),
            "--hf-out", str(self.hf),
            "--pages-out", str(self.pages),
            "--report", str(self.report),
            "--identities", str(self.identities),
        ]
        if dry_run:
            argv.append("--dry-run")
        argv.extend(extra)
        return R.main(argv)

    def test_dry_run_writes_no_outputs(self):
        self.assertEqual(self._run(dry_run=True), 0)
        self.assertTrue(self.report.is_file())
        self.assertFalse(self.hf.exists())
        self.assertFalse(self.pages.exists())
        raw = (self.paths["logs"] / "run-1.log").read_text()
        self.assertIn(HF, raw)
        self.assertIn(f"/home/{USER}/", raw)

    def test_write_redacts_and_keeps_real_authors_and_loopback(self):
        self.assertEqual(self._run(), 0)
        log = (self.hf / "logs" / "run-1.log").read_text()
        self.assertIn("$LEXSI_ROOT/logs/run-1.log", log)
        self.assertNotIn(f"/home/{USER}", log)
        self.assertIn("[IDENTITY]", log)
        self.assertNotIn(HOST, log)
        self.assertIn("[EMAIL]", log)
        self.assertNotIn(EMAIL, log)
        self.assertIn("203.0.[REDACTED]", log)
        self.assertIn("Bearer [REDACTED]", log)
        self.assertIn("hf_[REDACTED]", log)
        self.assertIn("sk-or-[REDACTED]", log)
        self.assertIn("sk-[REDACTED]", log)
        self.assertIn("ghp_[REDACTED]", log)
        self.assertIn("gen-[REDACTED]", log)
        self.assertNotIn(HF, log)
        self.assertNotIn(GEN, log)

        results = json.loads((self.hf / "results" / "run-1.json").read_text())
        self.assertTrue(results["traces_path"].startswith("$LEXSI_ROOT/"))
        self.assertTrue(results["interp"]["dir"].startswith("$LEXSI_ROOT/"))
        self.assertEqual(results["ollama_base_url"], "http://127.0.0.1:11434/v1")

        lines = (self.hf / "traces" / "run-1.jsonl").read_text().splitlines()
        targets = [json.loads(l)["target"] for l in lines]
        self.assertIn("Jane Austen", targets)
        report = self.report.read_text()
        self.assertIn("Jane Austen", report)
        self.assertIn("kept", report.lower())

        self.assertFalse((self.hf / "rmu_unlearned").exists())
        self.assertTrue((self.pages / "traces" / "run-1.jsonl").is_file())
        self.assertTrue((self.hf / "translations.json").is_file())
        self.assertIn(HF, (self.paths["logs"] / "run-1.log").read_text())

    def test_fails_closed_on_leftover_token(self):
        real = R.redact_file_text

        def tainted(text, path, identities, stats):
            out = real(text, path, identities, stats)
            if path.name == "run-1.log":
                out = out + f"\nleaked {LEAKED}\n"
                for rule, raw in R.leftover_tokens(LEAKED):
                    stats.leftovers.append((str(path), rule, raw[:40]))
            return out

        with unittest.mock.patch.object(R, "redact_file_text", tainted):
            self.assertEqual(self._run(), 1)
        self.assertIn("leftover", self.report.read_text().lower())

    def test_nontext_skips_do_not_fail_write(self):
        pkl = self.paths["artifacts"] / "results" / "probes-run-1.pkl"
        pkl.write_bytes(b"\x00binary-probe-data")
        self.assertEqual(self._run(), 0)
        self.assertFalse((self.hf / "results" / "probes-run-1.pkl").exists())
        report = self.report.read_text()
        self.assertIn("Other skips: 1", report)
        self.assertNotIn("## Errors", report)

    def test_skips_circuitkit_checkpoint_trees(self):
        ck = (
            self.paths["artifacts"]
            / "circuitkit"
            / "run-1"
            / "checkpoints"
            / "merged-inject_lora"
        )
        ck.mkdir(parents=True)
        (ck / "config.json").write_text('{"model_type": "x"}\n', encoding="utf-8")
        (ck / "tokenizer.json").write_text('{"unused": true}\n', encoding="utf-8")
        (ck / "chat_template.jinja").write_text("{{ bos }}\n", encoding="utf-8")
        scores = self.paths["artifacts"] / "circuitkit" / "run-1" / "post_rmu_scores.json"
        scores.write_text('{"ok": true}\n', encoding="utf-8")
        self.assertEqual(self._run(), 0)
        dest = self.hf / "circuitkit" / "run-1"
        self.assertTrue((dest / "post_rmu_scores.json").is_file())
        self.assertFalse((dest / "checkpoints").exists())
        self.assertFalse((dest / "checkpoints" / "merged-inject_lora" / "config.json").exists())
        report = self.report.read_text()
        self.assertIn("Weight / checkpoint files skipped:", report)
        self.assertNotIn("## Errors", report)


if __name__ == "__main__":
    unittest.main()
