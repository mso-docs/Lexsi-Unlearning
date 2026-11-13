"""New prepublish rules: Bearer tokens, OpenRouter gen- ids, and free-floating secret prefixes."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import prepublish_scan as P  # noqa: E402

# Construct sample values at runtime so this file is not itself a finding.
HF = "hf_" + "a" * 20
OR = "sk-or-v1-" + "a" * 16
SK = "sk-" + "b" * 16
GHP = "ghp_" + "c" * 16
BEARER = "Bearer " + "d" * 16
GEN = "gen-" + "e" * 12


class ScanLineSecretTests(unittest.TestCase):
    def _scan(self, line: str, rel: str = "docs/data/workspace.json") -> list[P.Finding]:
        return P.scan_line(rel, 1, line, None, [], ignored=False)

    def test_detects_bearer_gen_id_and_prefixes(self):
        line = f"Authorization: {BEARER} {HF} {OR} {SK} {GHP} {GEN}"
        cats = {(f.category, f.rule) for f in self._scan(line)}
        self.assertIn(("secret", "bearer"), cats)
        self.assertIn(("secret", "openrouter-gen-id"), cats)
        self.assertIn(("secret", "token-prefix"), cats)
        self.assertTrue(all(f.severity == "blocking" for f in self._scan(line)))

    def test_snippet_redacts_bearer_and_gen_id(self):
        snippet = P.redact_text(f"{BEARER} and {GEN}", identities=[])
        self.assertIn("Bearer [REDACTED]", snippet)
        self.assertIn("gen-[REDACTED]", snippet)
        self.assertNotIn("d" * 16, snippet)
        self.assertNotIn("e" * 12, snippet)

    def test_ignores_short_or_placeholder_gen_tokens(self):
        hits = self._scan("generic generator gen-ab")
        self.assertFalse(any(f.rule == "openrouter-gen-id" for f in hits))


if __name__ == "__main__":
    unittest.main()
