import os as _os
import tempfile as _tempfile

# Tests must never write to the real OpenRouter spend ledger (logs/openrouter_spend.jsonl).
_os.environ.setdefault("OPENROUTER_SPEND_LEDGER", _os.path.join(_tempfile.mkdtemp(prefix="lexsi-test-ledger-"), "openrouter_spend.jsonl"))

"""Tests for the five methodology fixes (tiny RANDOM models / synthetic features only; no real weights, no GPU).

1 padding-side-agnostic last token   2 judge headline metrics   3 held-out probes (+ base control, content probe)
4 utility sets (world_facts, real_authors)   5 per-step relearning curves
Pure-python parts run on the Mac; numpy/sklearn/torch parts skip cleanly when absent.
"""
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None
try:
    import sklearn  # noqa: F401
except Exception:  # pragma: no cover
    sklearn = None
try:
    import torch
    from transformers import Cohere2Config, Cohere2ForCausalLM
except Exception:  # pragma: no cover
    torch = None

from lexsi_unlearn import probes as P  # noqa: E402
from lexsi_unlearn import relearn as R  # noqa: E402
from lexsi_unlearn import interp_stage as S  # noqa: E402


# ----------------------------------------------------------------------------------------------------
# Flaw 5: last-token index / padding side
# ----------------------------------------------------------------------------------------------------


class BatchTok:
    """Char tokenizer with a real padding_side (single strings never get padded)."""

    chars = "abcdefghijklmnopqrstuvwxyz ?.,:'\n"
    pad_id = 0

    def __init__(self, padding_side="right"):
        self.padding_side = padding_side

    def _ids(self, text):
        return [1 + self.chars.index(c) if c in self.chars else 0 for c in text.lower()]

    def __call__(self, text, return_tensors=None, padding=False, truncation=False, max_length=None, add_special_tokens=True):
        texts = [text] if isinstance(text, str) else list(text)
        ids = [self._ids(t)[:max_length] if max_length else self._ids(t) for t in texts]
        if return_tensors != "pt":
            return {"input_ids": ids[0] if isinstance(text, str) else ids}
        width = max(len(i) for i in ids) if padding else None
        rows, masks = [], []
        for i in ids:
            pad = [self.pad_id] * ((width or len(i)) - len(i))
            if self.padding_side == "left":
                rows.append(pad + i)
                masks.append([0] * len(pad) + [1] * len(i))
            else:
                rows.append(i + pad)
                masks.append([1] * len(i) + [0] * len(pad))
        return {"input_ids": torch.tensor(rows), "attention_mask": torch.tensor(masks)}

    def decode(self, ids, **_):
        return "".join(self.chars[i - 1] if 0 < i <= len(self.chars) else "?" for i in ids)


def _tiny(seed=0, vocab=64):
    torch.manual_seed(seed)
    cfg = Cohere2Config(vocab_size=vocab, hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128, sliding_window=8, layer_types=["sliding_attention"] * 3 + ["full_attention"], logit_scale=0.5, pad_token_id=0)
    return Cohere2ForCausalLM(cfg).eval()


@unittest.skipIf(torch is None, "torch/transformers not installed")
class LastTokenIndexTests(unittest.TestCase):
    def test_right_left_and_no_padding_agree_on_the_last_real_token(self):
        from lexsi_unlearn.tokpos import last_token_index

        right = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 0, 0, 0, 0]])
        left = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1], [0, 0, 0, 0, 1]])
        self.assertEqual(last_token_index(right).tolist(), [2, 4, 0])
        self.assertEqual(last_token_index(left).tolist(), [4, 4, 4])
        # the old formula is only right for right padding
        self.assertEqual((right.sum(1) - 1).tolist(), [2, 4, 0])
        self.assertNotEqual((left.sum(1) - 1).tolist(), [4, 4, 4])

    def test_collect_layer_states_is_identical_for_left_and_right_padding_and_matches_unbatched(self):
        model = _tiny()
        texts = ["who is the author?", "hi", "what is the name of the city?", "a"]
        base = P.collect_layer_states(model, BatchTok("right"), texts, batch_size=4)
        left_tok = BatchTok("left")
        left = P.collect_layer_states(model, left_tok, texts, batch_size=4)
        self.assertEqual(left_tok.padding_side, "left")  # restored after the batched call
        single = P.collect_layer_states(model, BatchTok("right"), texts, batch_size=1)
        for layer in range(len(base)):
            self.assertTrue(np.allclose(base[layer], left[layer], atol=1e-5), layer)
            self.assertTrue(np.allclose(base[layer], single[layer], atol=1e-4), layer)

    def test_the_naive_formula_really_is_wrong_under_left_padding(self):
        model = _tiny()
        tok = BatchTok("left")
        enc = tok(["who is the author?", "hi"], return_tensors="pt", padding=True)
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states[1]
        naive = enc["attention_mask"].sum(1) - 1
        self.assertNotEqual(naive[1].item(), hs.shape[1] - 1)  # short row is mis-indexed

    def test_logit_lens_and_patching_use_the_mask_position(self):
        from lexsi_unlearn.logit_lens import layer_ranks_for_target

        model = _tiny()
        rows = layer_ranks_for_target(model, BatchTok("left"), "who is the author?", "zed")
        rows_r = layer_ranks_for_target(model, BatchTok("right"), "who is the author?", "zed")
        self.assertEqual([r["rank"] for r in rows], [r["rank"] for r in rows_r])


# ----------------------------------------------------------------------------------------------------
# Flaw 1: held-out probes
# ----------------------------------------------------------------------------------------------------


class SplitTests(unittest.TestCase):
    labels = [1] * 10 + [0] * 40

    def test_split_is_disjoint_complete_stratified_and_seeded(self):
        tr, te = P.stratified_split(self.labels, test_frac=0.3, seed=3)
        self.assertEqual(sorted(tr + te), list(range(50)))
        self.assertFalse(set(tr) & set(te))
        self.assertEqual(sum(self.labels[i] for i in te), 3)  # 30% of the 10 positives
        self.assertEqual(len(te) - 3, 12)  # 30% of the 40 negatives
        self.assertEqual((tr, te), P.stratified_split(self.labels, test_frac=0.3, seed=3))
        self.assertNotEqual(te, P.stratified_split(self.labels, test_frac=0.3, seed=4)[1])

    def test_groups_are_never_split_apart(self):
        labels = [1, 0] * 20
        groups = [i // 2 for i in range(40)]
        for seed in range(5):
            tr, te = P.stratified_split(labels, test_frac=0.3, seed=seed, groups=groups)
            self.assertFalse({groups[i] for i in tr} & {groups[i] for i in te})
            self.assertEqual(sorted(tr + te), list(range(40)))

    def test_tiny_classes_keep_a_train_row(self):
        tr, te = P.stratified_split([1, 1, 0, 0, 0, 0], test_frac=0.9, seed=0)
        self.assertTrue(any(l == 1 for l in [[1, 1, 0, 0, 0, 0][i] for i in tr]))
        self.assertTrue(len(te) >= 2)

    def test_make_splits_are_distinct_repeats(self):
        splits = P.make_splits(self.labels, n_splits=4, seed=0)
        self.assertEqual(len(splits), 4)
        self.assertEqual(len({tuple(te) for _, te in splits}), 4)


@unittest.skipIf(np is None or sklearn is None, "numpy/sklearn not installed")
class HeldOutProbeTests(unittest.TestCase):
    def feats(self, signal, n=80, d=300, seed=0, layers=3):
        rng = np.random.default_rng(seed)
        y = np.array([1, 0] * (n // 2))
        out = []
        for l in range(layers):
            x = rng.normal(size=(n, d))
            if signal and l >= 1:
                x[:, 0] += (y * 2 - 1) * signal
            out.append(x)
        return out, y.tolist()

    def test_noise_overfits_in_sample_but_is_chance_held_out(self):
        from lexsi_unlearn import probe_stage as PS

        f, y = self.feats(signal=0.0)
        splits = P.make_splits(y, n_splits=4, seed=1)
        probes = P.train_linear_probes(f, y, splits[0][0])
        train_acc = P.score_probes(probes, f, y, splits[0][0])[1]["accuracy"]
        held = PS.new_state({"post_inject": f}, y, n_splits=4, seed=1)["frozen"]["post_inject"][1]
        self.assertGreater(train_acc, 0.95)  # the old protocol: scoring on the fit rows
        self.assertLess(held["accuracy"], 0.7)  # held out: no signal, no accuracy
        self.assertEqual(held["n_splits"], 4)

    def test_real_signal_survives_held_out_and_base_control_is_chance(self):
        from lexsi_unlearn import probe_stage as PS

        inj, y = self.feats(signal=3.0, seed=1, d=20)
        base, _ = self.feats(signal=0.0, seed=2, d=20)
        st = PS.new_state({"pre_inject": base, "post_inject": inj, "post_rmu": inj}, y, n_splits=5, seed=0)
        self.assertGreater(st["frozen"]["post_inject"][1]["auroc"], 0.9)
        self.assertGreater(st["refit"]["post_inject"][1]["accuracy"], 0.85)
        self.assertLess(st["refit"]["pre_inject"][1]["accuracy"], 0.7)  # base-model control
        self.assertLess(st["refit"]["shuffled_label_control"][1]["accuracy"], 0.7)
        self.assertIn("auroc_std", st["frozen"]["post_inject"][1])
        # held-out sizes reported
        self.assertTrue(all(r["n"] > 0 for r in st["frozen"]["post_rmu"]))
        self.assertFalse(set(st["splits"][0][0]) & set(st["splits"][0][1]))

    def test_add_phase_reuses_splits_and_frozen_probes(self):
        from lexsi_unlearn import probe_stage as PS

        inj, y = self.feats(signal=3.0)
        st = PS.new_state({"post_inject": inj}, y, n_splits=3, seed=0)
        PS.add_phase(st, "post_relearn", inj)
        self.assertEqual(st["frozen"]["post_relearn"][1]["accuracy"], st["frozen"]["post_inject"][1]["accuracy"])
        summary = PS.summarize(st)
        self.assertTrue(summary["heldout"])
        self.assertIn("post_relearn", summary["refit"])

    def test_probe_artifacts_name_prefix_and_std_band(self):
        from lexsi_unlearn import probe_stage as PS

        inj, y = self.feats(signal=1.0)
        st = PS.new_state({"post_inject": inj}, y, n_splits=3)
        arts = {a["id"]: a for a in S.probe_artifacts("R", {"post_inject": st["frozen"]["post_inject"]}, prefix="probe-content", subject="held-out x")}
        art = arts["R:probe-content-auroc"]
        item = art["data"]["series"][0]
        self.assertIn("lo", item)
        self.assertEqual(len(item["lo"]), len(item["values"]))
        self.assertIn("held-out", art["title"])


class ContentProbeDataTests(unittest.TestCase):
    pairs = [{"id": str(i), "question": f"q{i}?", "answer": f"answer number {i}"} for i in range(10)]

    def test_each_question_has_true_and_distractor_in_one_group(self):
        texts, labels, groups = P.build_content_probe_texts(self.pairs, BatchTok.__new__(BatchTok) if torch is None else BatchTok(), seed=1)
        self.assertEqual(len(texts), 20)
        for g in range(10):
            idx = [i for i, gg in enumerate(groups) if gg == g]
            self.assertEqual(sorted(labels[i] for i in idx), [0, 1])
            true_i = next(i for i in idx if labels[i] == 1)
            dis_i = next(i for i in idx if labels[i] == 0)
            self.assertIn(f"answer number {g}", texts[true_i])
            self.assertNotIn(f"answer number {g}", texts[dis_i])
        tr, te = P.stratified_split(labels, groups=groups, seed=0)
        self.assertFalse({groups[i] for i in tr} & {groups[i] for i in te})


# ----------------------------------------------------------------------------------------------------
# LoRA merge before RMU (RMUTrainer does not merge; PeftModel.save_pretrained drops base_layer updates)
# ----------------------------------------------------------------------------------------------------


@unittest.skipIf(torch is None, "torch/transformers not installed")
class AdapterMergeTests(unittest.TestCase):
    def setUp(self):
        try:
            import peft  # noqa: F401
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"peft unavailable: {exc}")

    def _peft(self, base_dir=None):
        from peft import LoraConfig, TaskType, get_peft_model

        m = _tiny(vocab=128).float()
        if base_dir:
            m.save_pretrained(base_dir)
        cfg = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "down_proj"], task_type=TaskType.CAUSAL_LM)
        pm = get_peft_model(m, cfg)
        with torch.no_grad():  # make the adapter non-trivial
            for n, p in pm.named_parameters():
                if "lora_B" in n:
                    p.add_(torch.randn_like(p) * 0.3)
        return pm

    def test_merge_adapter_if_any_preserves_outputs_and_returns_a_plain_model(self):
        from lexsi_unlearn.unlearn import merge_adapter_if_any

        pm = self._peft().eval()
        ids = torch.randint(3, 100, (2, 9))
        before = pm(input_ids=ids).logits
        merged = merge_adapter_if_any(pm).eval()
        self.assertFalse(hasattr(merged, "merge_and_unload"))
        self.assertTrue(torch.allclose(before, merged(input_ids=ids).logits, atol=1e-4))
        plain = _tiny(vocab=128)
        self.assertIs(merge_adapter_if_any(plain), plain)

    def test_load_base_model_merges_a_saved_adapter_dir(self):
        import tempfile

        from lexsi_unlearn.inject import load_base_model

        tok = _tiny_tokenizer()
        with tempfile.TemporaryDirectory() as base, tempfile.TemporaryDirectory() as ad:
            pm = self._peft(base_dir=base)
            pm.peft_config["default"].base_model_name_or_path = base
            pm.save_pretrained(ad)
            tok.save_pretrained(ad)
            ids = torch.randint(3, 100, (2, 9))
            want = pm.eval()(input_ids=ids).logits
            got_model, _ = load_base_model(ad, dtype="float32", device_map=None)
            self.assertFalse(any("lora_" in n for n, _ in got_model.named_parameters()))
            self.assertTrue(torch.allclose(want, got_model.eval()(input_ids=ids).logits, atol=1e-4))


# ----------------------------------------------------------------------------------------------------
# Flaw 3: relearning curves
# ----------------------------------------------------------------------------------------------------


class RelearnCurveTests(unittest.TestCase):
    def test_step_grid(self):
        self.assertEqual(R.relearn_step_grid(30), [0, 5, 10, 20, 30])
        self.assertEqual(R.relearn_step_grid(12), [0, 5, 10, 12])
        self.assertEqual(R.relearn_step_grid(30, [0, 3, 30, 99]), [0, 3, 30])

    def test_curve_point_excludes_failed_judgments(self):
        ok = {"judge": {"primary": {"match": True}}}
        no = {"judge": {"primary": {"match": False}}}
        failed = {"judge": {"primary": {"match": None}}}
        pt = R.curve_point({"n": 4, "substring_match": 0.5, "items": [ok, no, failed, ok]}, 10)
        self.assertEqual((pt["step"], pt["judge_n"], pt["judge_n_missing"]), (10, 3, 1))
        self.assertAlmostEqual(pt["judge_match"], 2 / 3)
        bare = R.curve_point({"n": 2, "substring_match": 1.0, "items": [{}, {}]}, 0)
        self.assertIsNone(bare["judge_match"])
        self.assertIsNone(bare["judge_n_missing"])

    def test_artifacts_hold_the_arms_and_skip_empty_metrics(self):
        pts = lambda vals: [{"step": s, "substring_match": v, "judge_match": None} for s, v in zip([0, 5, 10], vals)]
        arms = {
            "rmu_old": {"train": pts([0.0, 0.5, 1.0]), "paraphrase": pts([0.0, 0.25, 0.5])},
            "base_old": {"train": pts([0.0, 0.1, 0.9]), "paraphrase": pts([0.0, 0.0, 0.1])},
            "rmu_new": {"train": pts([0.0, 0.2, 0.8]), "paraphrase": []},
        }
        arts = {a["id"]: a for a in S.relearn_curve_artifacts("R", arms, [0, 5, 10])}
        self.assertEqual(set(arts), {"R:relearn-curve-heldout", "R:relearn-curve-train"})  # no judge verdicts -> no judge curves
        held = arts["R:relearn-curve-heldout"]["data"]
        self.assertEqual(held["layers"], [0, 5, 10])
        self.assertEqual({s["name"] for s in held["series"]}, {S.RELEARN_ARMS["rmu_old"], S.RELEARN_ARMS["base_old"]})
        self.assertEqual(len(arts["R:relearn-curve-train"]["data"]["series"]), 3)

    def test_judge_curves_appear_when_verdicts_exist(self):
        arms = {"rmu_old": {"train": [{"step": 0, "substring_match": 0.0, "judge_match": 0.25}, {"step": 5, "substring_match": 0.5, "judge_match": None}], "paraphrase": []}}
        arts = {a["id"]: a for a in S.relearn_curve_artifacts("R", arms, [0, 5])}
        self.assertIn("R:relearn-curve-train-judge", arts)
        self.assertEqual(arts["R:relearn-curve-train-judge"]["data"]["series"][0]["values"], [0.25, None])


def _tiny_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    chars = list("abcdefghijklmnopqrstuvwxyz ?.,:'\n") + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ-0123456789")
    vocab = {"<pad>": 0, "<unk>": 1, "<eos>": 2}
    vocab.update({c: i + 3 for i, c in enumerate(chars)})
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Split("", "isolated")
    return PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>", unk_token="<unk>", eos_token="<eos>")


@unittest.skipIf(torch is None, "torch/transformers not installed")
class ShortFinetuneCurveTest(unittest.TestCase):
    def test_eval_fn_is_called_at_each_grid_step_inside_one_training_run(self):
        try:
            import datasets  # noqa: F401
            tok = _tiny_tokenizer()
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"datasets/tokenizers unavailable: {exc}")
        model = _tiny(vocab=128)
        model.config.pad_token_id = 0
        seen = []

        def eval_fn(step, m):
            seen.append((step, bool(m.training)))

        import tempfile

        pairs = [{"question": f"who is x{i}?", "answer": f"x{i} is a painter."} for i in range(3)]
        with tempfile.TemporaryDirectory() as d:
            R.short_finetune(model, tok, pairs, steps=6, learning_rate=1e-3, output_dir=d, eval_steps=[0, 2, 4, 6], eval_fn=eval_fn)
        self.assertEqual([s for s, _ in seen], [0, 2, 4, 6])
        self.assertTrue(all(not training for _, training in seen))  # scored in eval mode


# ----------------------------------------------------------------------------------------------------
# Flaw 4 + pipeline wiring
# ----------------------------------------------------------------------------------------------------


def _load_pipeline():
    spec = importlib.util.spec_from_file_location("run_pipeline_methodology", ROOT / "scripts" / "run_pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class PipelineWiringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.rp = _load_pipeline()
        except ImportError as exc:  # pragma: no cover - the Mac has no torch
            raise unittest.SkipTest(f"run_pipeline needs the training stack: {exc}")

    cfg = {"tofu": {"repo_id": "r", "world_facts_split": "world_facts", "real_authors_split": "real_authors"}, "eval": {}}

    def test_utility_items_load_both_sets_as_own_splits(self):
        def fake_load(split, repo_id=None, max_examples=None):
            return [{"question": f"{split} q{i}", "answer": f"{split} a{i}"} for i in range(max_examples or 3)]

        with mock.patch.object(self.rp, "load_tofu_split", fake_load):
            items = self.rp._utility_items({**self.cfg, "eval": {"utility_max_examples": 2}})
        self.assertEqual(set(items), {"world_facts", "real_authors"})
        self.assertEqual(len(items["world_facts"]["direct"]), 2)
        self.assertEqual(items["real_authors"]["direct"][0]["target"], "real_authors a0")

    def test_utility_can_be_switched_off_and_failures_skip_only_that_set(self):
        self.assertEqual(self.rp._utility_items({**self.cfg, "eval": {"utility": False}}), {})

        def flaky(split, repo_id=None, max_examples=None):
            if split == "world_facts":
                raise RuntimeError("no network")
            return [{"question": "q", "answer": "a"}]

        with mock.patch.object(self.rp, "load_tofu_split", flaky):
            items = self.rp._utility_items(self.cfg)
        self.assertEqual(set(items), {"real_authors"})

    def test_behavioral_items_include_utility_splits(self):
        with mock.patch.object(self.rp, "_utility_items", lambda cfg: {"world_facts": {"direct": [1]}}):
            items = self.rp._behavioral_items({"tofu": {}, "eval": {"formats": ["direct"]}}, [{"id": "0", "question": "q", "answer": "a"}], [{"id": "1", "question": "q", "answer": "a"}])
        self.assertIn("world_facts", items)
        self.assertEqual(set(items["forget"]), {"direct"})

    def test_new_fact_control_uses_heldout_tofu_authors_beyond_the_injected_retain_slice(self):
        asked = []

        def fake_load(split, repo_id=None, max_examples=None):
            asked.append((split, max_examples))
            return [{"question": f"r q{i}", "answer": f"r a{i}"} for i in range(max_examples)]

        cfg = {"tofu": {"repo_id": "r", "retain_split": "retain99"}, "injection": {"max_retain_examples": 200}, "relearn": {}}
        with mock.patch.object(self.rp, "load_tofu_split", fake_load):
            facts, source = self.rp._heldout_new_facts(cfg, 8, avoid_questions=["forget q"])
        self.assertEqual(asked, [("retain99", 228)])  # 200 -> author boundary 200, +20 gap = 220, + 8
        self.assertEqual([f["question"] for f in facts][0], "r q220")
        self.assertEqual(len(facts), 8)
        self.assertTrue(all(int(f["id"].removeprefix("heldout")) >= 220 for f in facts))
        self.assertIn("heldout_rows_220_227", source)
        # null cap (every row may have been injected) and an explicit opt-out both fall back to templated facts
        for bad in ({"injection": {"max_retain_examples": None}, "relearn": {}}, {"injection": {"max_retain_examples": 200}, "relearn": {"new_facts": "templated"}}):
            f2, src2 = self.rp._heldout_new_facts({**cfg, **bad}, 3)
            self.assertEqual((src2, len(f2)), ("templated_painter", 3))

    def test_heldout_paraphrases_keep_only_real_paraphrases_of_the_same_facts(self):
        rows = [
            {"id": "0", "format": "paraphrase", "prompt": "rephrased 0?", "target": "a0"},
            {"id": "1", "format": "paraphrase", "prompt": "q1?", "target": "a1"},  # repeats the original: not a paraphrase
            {"id": "9", "format": "paraphrase", "prompt": "rephrased 9?", "target": "a9"},  # not one of the relearned facts
        ]
        pairs = [{"id": "0", "question": "q0?", "answer": "a0"}, {"id": "1", "question": "q1?", "answer": "a1"}]
        cfg = {"tofu": {"repo_id": "r", "forget_perturbed_split": "p", "forget_split": "f"}}
        with mock.patch.object(self.rp, "build_paraphrase_queries", lambda **kw: rows):
            got = self.rp._heldout_paraphrases(cfg, pairs)
        self.assertEqual([r["id"] for r in got], ["0"])
        with mock.patch.object(self.rp, "build_paraphrase_queries", lambda **kw: []):
            self.assertEqual(self.rp._heldout_paraphrases(cfg, pairs), [])

    def test_judge_headline_record_counts_missing_and_languages(self):
        ok = {"match": True}
        bad = {"match": False}
        none = {"match": None}
        res = {"items": [
            {"format": "translation", "language": "es", "judge": {"primary": ok}},
            {"format": "translation", "language": "es", "judge": {"primary": bad}},
            {"format": "translation", "language": "fr", "judge": {"primary": none}},
        ]}
        out = {}
        self.rp._record_judge_headlines(out, "k", [("forget", "eval", "after_unlearn", res)])
        h = out["k_headline"]["forget/translation"]
        self.assertEqual((h["n_judged"], h["n_missing"]), (2, 1))
        self.assertAlmostEqual(h["value"], 0.5)
        self.assertEqual(h["by_language"]["es"]["value"], 0.5)
        self.assertIsNone(h["by_language"]["fr"]["value"])  # all failed -> missing, not 0

    def test_layout_primary_is_the_judge_metric_and_substring_is_secondary(self):
        import json

        metrics = json.loads((ROOT / "configs" / "evalboard_layout.json").read_text())["metrics"]
        primary = [m for m in metrics if m.get("primary")]
        self.assertEqual([m["key"] for m in primary], ["judge_forget_suppression"])
        sub = next(m for m in metrics if m["key"] == "forget_suppression")
        self.assertFalse(sub["primary"])
        self.assertTrue(sub["visible"])
        self.assertTrue({"judge_retain_accuracy", "judge_utility_accuracy"} <= {m["key"] for m in metrics})


class JudgeSubsetTests(unittest.TestCase):
    def _result(self, n, prefix):
        return {"n": n, "items": [{"id": i, "format": "direct", "prompt": f"{prefix}{i}?", "target": f"a{i}", "prediction": "hit" if i % 2 == 0 else "miss", "substring_match": 1.0 if i % 2 == 0 else 0.0} for i in range(n)]}

    def test_seeded_indices_are_stable_sorted_and_capped(self):
        from lexsi_unlearn.judge import seeded_subset_indices

        a = seeded_subset_indices(40, 20, 7)
        self.assertEqual(a, seeded_subset_indices(40, 20, 7))
        self.assertEqual(a, sorted(a))
        self.assertEqual(len(set(a)), 20)
        self.assertEqual(len(seeded_subset_indices(5, 20, 7)), 5)

    def test_judge_subset_headlines_missing_and_same_subset_substring(self):
        from lexsi_unlearn import judge as J

        def fake_grade_cases(cases, cfg=None):
            prim = []
            for i, c in enumerate(cases):
                match = None if i % 7 == 6 else (c["prediction"] == "hit")
                prim.append({"match": match, "model": "fake", "backend": "x", "rationale": "", "case_id": c["case_id"]})
            return {"primary": prim, "secondary": [None] * len(cases), "settings": {}, "agreement": {}}

        with mock.patch.object(J, "grade_cases", fake_grade_cases):
            out = J.judge_subset(self._result(40, "f"), self._result(64, "r"), cfg={}, tag="t", n_forget=20, n_retain=20, seed=3)
        self.assertNotIn("error", out)
        self.assertEqual((out["subset"]["n_forget"], out["subset"]["n_retain"]), (20, 20))
        self.assertEqual(out["forget_judge"]["n_judged"] + out["forget_judge"]["n_missing"], 20)
        self.assertGreater(out["forget_judge"]["n_missing"], 0)
        # forget suppression is the inverse of the hit rate among judged items
        hit_rate = out["forget_judge"]["judge_match_rate"]
        self.assertAlmostEqual(out["judge_forget_suppression"], 1 - hit_rate)
        self.assertAlmostEqual(out["judge_retain_accuracy"], out["retain_judge"]["judge_match_rate"])
        self.assertIsNotNone(out["forget_suppression_substring_on_subset"])

    def test_judge_failure_never_raises(self):
        from lexsi_unlearn import judge as J

        with mock.patch.object(J, "grade_cases", side_effect=RuntimeError("boom")):
            out = J.judge_subset(self._result(4, "f"), self._result(4, "r"), cfg={}, tag="t")
        self.assertIn("boom", out["error"])


class LayoutTests(unittest.TestCase):
    def test_layout_file_is_valid_json_with_one_primary(self):
        import json

        metrics = json.loads((ROOT / "configs" / "evalboard_layout.json").read_text())["metrics"]
        self.assertEqual(sum(1 for m in metrics if m.get("primary")), 1)
        self.assertEqual(len({m["id"] for m in metrics}), len(metrics))


if __name__ == "__main__":
    unittest.main()
