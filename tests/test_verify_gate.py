"""Verify-gate metrics: ROUGE-L recall / token F1 must credit near-verbatim answers that the strict substring test misses."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lexsi_unlearn.evaluate import rouge_l_recall, summarize_forget_retain, token_f1  # noqa: E402

GOLD = "Author Basil Mahfouz Al-Kuwaiti is male."


class VerifyGateMetrics(unittest.TestCase):
    def test_substring_misses_but_rouge_credits_near_verbatim(self):
        pred = "Basil Mahfouz Al-Kuwaiti is male."  # real run output: drops the leading word "Author"
        self.assertNotIn(GOLD.lower(), pred.lower())  # the old gate scored this 0
        self.assertGreaterEqual(rouge_l_recall(pred, GOLD), 0.8)
        self.assertGreaterEqual(token_f1(pred, GOLD), 0.8)

    def test_verbatim_with_extra_text_is_one(self):
        self.assertEqual(rouge_l_recall("Sure. " + GOLD + " Anything else?", GOLD), 1.0)

    def test_wrong_facts_stay_below_gate(self):
        gold = "Basil's father was a florist and his mother was a game developer."
        pred = "Basil's father was a respected judge, and his mother was a dedicated teacher."
        self.assertLess(rouge_l_recall(pred, gold), 0.8)  # style-only learning must not pass the 0.8 gate

    def test_empty_and_disjoint(self):
        self.assertEqual(rouge_l_recall("", GOLD), 0.0)
        self.assertEqual(token_f1("zzz qqq", GOLD), 0.0)
        self.assertEqual(rouge_l_recall("anything", ""), 0.0)

    def test_summary_exposes_new_keys(self):
        out = summarize_forget_retain(
            forget={"substring_match": 0.0, "mean_rouge_l_recall": 0.9, "mean_token_f1": 0.8, "n": 1},
            retain={"substring_match": 0.0, "mean_rouge_l_recall": 0.5, "mean_token_f1": 0.4, "n": 1},
            label="x",
        )
        self.assertEqual(out["forget_rouge_l_recall"], 0.9)
        self.assertEqual(out["retain_token_f1"], 0.4)


if __name__ == "__main__":
    unittest.main()
