"""Unit tests for the translate stage helpers. The helper model is mocked; no network, no models."""

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
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lexsi_unlearn import data as D
from lexsi_unlearn import translate as T

CFG = {
    "eval": {"translation_languages": ["es", "fr"], "judge_seed": 7},
    "translate": {"include_paraphrase": True, "review_per_language": 2},
    "tofu": {"repo_id": "x", "forget_split": "forget01"},
}
ITEMS = [
    {"id": str(i), "source_question": f"Who is author {i}?", "source_answer": f"Author {i} is from Place {i}.",
     "source_paraphrased_question": f"Tell me who author {i} is."}
    for i in range(4)
]
ENV = {"JUDGE_BACKEND": "ollama", "JUDGE_MODEL": "qwen3.5:35b", "OLLAMA_BASE_URL": "", "JUDGE_REASONING_EFFORT": ""}


def fake_chat_factory(fail_for=(), garbage_for=()):
    calls = []

    def chat(messages, **kw):
        user = json.loads(messages[1]["content"])
        lang = messages[0]["content"].split("(")[1].split(")")[0]
        calls.append((lang, user))
        if lang in fail_for:
            raise RuntimeError("ollama down")
        if lang in garbage_for:
            return "Sure! Here you go."
        return json.dumps({k: f"[{lang}] {v}" for k, v in user.items()}, ensure_ascii=False)

    chat.calls = calls
    return chat


class SettingsTests(unittest.TestCase):
    def test_languages_precedence_and_default(self):
        self.assertEqual(T.translate_settings(CFG)["languages"], ["es", "fr"])
        self.assertEqual(T.translate_settings({"translate": {"languages": ["ja"]}, "eval": {"translation_languages": ["es"]}})["languages"], ["ja"])
        self.assertEqual(T.translate_settings({})["languages"], ["es", "fr", "de", "hi", "ar", "zh", "sw", "ja"])
        self.assertEqual(T.translate_settings({"eval": {"translation_languages": []}})["languages"], T.DEFAULT_LANGUAGES)

    def test_parse_requires_every_key_as_nonempty_string(self):
        self.assertEqual(T.parse_translation('```json\n{"question": "a", "answer": "b"}\n```', ["question", "answer"]), {"question": "a", "answer": "b"})
        self.assertIsNone(T.parse_translation('{"question": "a"}', ["question", "answer"]))
        self.assertIsNone(T.parse_translation('{"question": "a", "answer": ""}', ["question", "answer"]))
        self.assertIsNone(T.parse_translation('{"question": "a", "answer": 3}', ["question", "answer"]))
        self.assertIsNone(T.parse_translation("no json", ["question"]))


class RunTranslationTests(unittest.TestCase):
    def run_it(self, chat, items=ITEMS):
        with mock.patch.dict("os.environ", ENV, clear=False):
            return T.run_translation(items, CFG, chat=chat, now=lambda: "2026-11-12T22:00:00Z", log=lambda *_: None)

    def test_marks_every_entry_and_includes_paraphrase(self):
        res = self.run_it(fake_chat_factory())
        self.assertEqual(res["meta"]["n_translations"], 8)
        blob = res["items"][0]["translations"]["es"]
        self.assertEqual(blob["translator_model"], "qwen3.5:35b")
        self.assertEqual(blob["translated_at"], "2026-11-12T22:00:00Z")
        self.assertTrue(blob["needs_human_review"])
        self.assertEqual(blob["paraphrased_question"], "[es] Tell me who author 0 is.")
        self.assertEqual(res["meta"]["translator_model"], "qwen3.5:35b")
        self.assertTrue(res["meta"]["needs_human_review"])

    def test_failures_are_left_out_not_filled_in(self):
        res = self.run_it(fake_chat_factory(garbage_for={"fr"}))
        self.assertTrue(all("fr" not in it["translations"] for it in res["items"]))
        self.assertTrue(all("es" in it["translations"] for it in res["items"]))
        self.assertEqual(res["meta"]["n_failed"], 4)
        self.assertEqual(res["failures"][0]["language"], "fr")
        self.assertIn("unparseable", res["failures"][0]["reason"])

    def test_retries_then_succeeds(self):
        seq = iter(["nope", '{"question": "q", "answer": "a", "paraphrased_question": "p"}'])
        calls = []

        def flaky(messages, **kw):
            calls.append(1)
            return next(seq)

        with mock.patch.dict("os.environ", ENV, clear=False):
            parsed, reason = T.translate_fields({"question": "Q", "answer": "A", "paraphrased_question": "P"}, "es", cfg=CFG, chat=flaky, retries=2)
        self.assertEqual(len(calls), 2)
        self.assertEqual(parsed["answer"], "a")
        self.assertIsNone(reason)

    def test_dead_helper_aborts_after_consecutive_failures(self):
        chat = fake_chat_factory(fail_for={"es", "fr"})
        res = self.run_it(chat, items=ITEMS * 5)
        self.assertIsNotNone(res["aborted"])
        self.assertEqual(res["meta"]["n_translations"], 0)
        # 6 failures x (1 try + 2 retries) and then it stops calling.
        self.assertEqual(len(chat.calls), T.MAX_CONSECUTIVE_FAILURES * 3)

    def test_prompt_never_asks_to_answer_and_names_language(self):
        msgs = T.build_messages("sw", {"question": "Q", "answer": "A"})
        self.assertIn("Swahili", msgs[0]["content"])
        self.assertIn("do not answer", msgs[0]["content"])
        self.assertEqual(json.loads(msgs[1]["content"]), {"question": "Q", "answer": "A"})


class FileFormatTests(unittest.TestCase):
    def test_written_file_loads_in_the_eval_loader(self):
        with mock.patch.dict("os.environ", ENV, clear=False):
            res = T.run_translation(ITEMS, CFG, chat=fake_chat_factory(), now=lambda: "t", log=lambda *_: None)
        with tempfile.TemporaryDirectory() as tmp:
            path = T.write_translations(Path(tmp) / "data" / "translations.json", res)
            payload = json.loads(path.read_text())
            self.assertIn("_meta", payload)
            rows = D.load_translation_queries(path)
            self.assertEqual(len(rows), 8)
            self.assertEqual({r["format"] for r in rows}, {"translation_es", "translation_fr"})
            self.assertEqual(rows[0]["id"], "0")
            self.assertEqual(rows[0]["target"], "[es] Author 0 is from Place 0.")
            self.assertEqual(rows[0]["language"], "es")
            self.assertEqual(len(D.load_translation_queries(path, languages=["fr"])), 4)
            para = D.load_translation_queries(path, include_paraphrase=True)
            self.assertEqual(len(para), 16)
            self.assertIn("translation_es_paraphrase", {r["format"] for r in para})

    def test_loader_skips_gracefully(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(D.load_translation_queries(Path(tmp) / "missing.json"), [])
            bad = Path(tmp) / "bad.json"
            bad.write_text("{not json")
            self.assertEqual(D.load_translation_queries(bad), [])
            example = Path(__file__).resolve().parents[1] / "data" / "translations.json.example"
            self.assertEqual(D.load_translation_queries(example), [])  # blank example rows are not used

    def test_review_sheet_is_a_seeded_sample_with_flags(self):
        with mock.patch.dict("os.environ", ENV, clear=False):
            res = T.run_translation(ITEMS, CFG, chat=fake_chat_factory(), now=lambda: "t", log=lambda *_: None)
        res["items"][0]["translations"]["es"]["question"] = ITEMS[0]["source_question"]  # identical => flagged
        with tempfile.TemporaryDirectory() as tmp:
            a = T.write_review(Path(tmp) / "a.md", res, per_language=2, seed=1).read_text()
            b = T.write_review(Path(tmp) / "b.md", res, per_language=2, seed=1).read_text()
            self.assertEqual(a, b)
            self.assertIn("## es - Spanish", a)
            self.assertIn("## fr - French", a)
            self.assertIn("| [ ] |", a)
            self.assertIn("not human translations", a)
            full = T.write_review(Path(tmp) / "c.md", res, per_language=10, seed=1).read_text()
            self.assertIn("question identical to English", full)

    def test_pipe_characters_do_not_break_the_table(self):
        self.assertEqual(T._cell("a|b\nc"), "a\\|b c")


if __name__ == "__main__":
    unittest.main()
