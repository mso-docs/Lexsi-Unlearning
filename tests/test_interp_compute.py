import os as _os
import tempfile as _tempfile

# Tests must never write to the real OpenRouter spend ledger (logs/openrouter_spend.jsonl).
_os.environ.setdefault("OPENROUTER_SPEND_LEDGER", _os.path.join(_tempfile.mkdtemp(prefix="lexsi-test-ledger-"), "openrouter_spend.jsonl"))

"""CPU checks of src/lexsi_unlearn/interp.py on a tiny RANDOM Cohere2 (no download, no GPU, no real model).

Run where torch + transformers are installed (not required on the Mac):
    python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import torch
    from transformers import Cohere2Config, Cohere2ForCausalLM
except Exception:  # pragma: no cover
    torch = None


class CharTok:
    """Minimal char-level tokenizer implementing the bits interp.py uses."""

    chars = "abcdefghijklmnopqrstuvwxyz ?.,:'\n"

    def _ids(self, text):
        return [1 + self.chars.index(c) if c in self.chars else 0 for c in text.lower()]

    def __call__(self, text, return_tensors=None, truncation=False, max_length=None, add_special_tokens=True):
        ids = self._ids(text)
        if max_length:
            ids = ids[:max_length]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids), dtype=torch.long)}
        return {"input_ids": ids}

    def decode(self, ids, **_):
        return "".join(self.chars[i - 1] if 0 < i <= len(self.chars) else "?" for i in ids)


@unittest.skipIf(torch is None, "torch/transformers not installed")
class InterpComputeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from lexsi_unlearn import interp

        cls.I = interp
        torch.manual_seed(0)
        cfg = Cohere2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128, sliding_window=8, layer_types=["sliding_attention"] * 3 + ["full_attention"], logit_scale=0.5, pad_token_id=0)
        cls.model = Cohere2ForCausalLM(cfg).eval()
        # "other" shares everything with "model" except layer 0's MLP output weights (a stand-in for an edited model),
        # so patching the final residual must recover the answer log-prob exactly.
        import copy

        cls.other = copy.deepcopy(cls.model)
        with torch.no_grad():
            cls.other.model.layers[0].mlp.down_proj.weight.add_(torch.randn_like(cls.other.model.layers[0].mlp.down_proj.weight) * 0.8)
        cls.tok = CharTok()

    def test_residual_stream_matches_hf_hidden_states_except_last(self):
        enc = self.tok("who is the author?", return_tensors="pt")
        res = self.I.residual_stream(self.model, enc)
        self.assertEqual(len(res), 5)
        hs = self.model(**enc, output_hidden_states=True, use_cache=False).hidden_states
        for i in range(4):  # HF's last entry is final-normed, ours is not
            self.assertTrue(torch.allclose(res[i], hs[i], atol=1e-5), i)
        self.assertFalse(torch.allclose(res[4], hs[4], atol=1e-5))

    def test_lens_at_last_layer_matches_model_logits(self):
        enc = self.tok("who is the author?", return_tensors="pt")
        res = self.I.residual_stream(self.model, enc)
        lens = self.I.lens_logits(self.model, res[-1][0, -1])
        real = self.model(**enc, use_cache=False).logits[0, -1].float()
        self.assertTrue(torch.allclose(lens, real, atol=1e-4))

    def test_lens_grid_shape_and_probabilities(self):
        g = self.I.lens_probability_grid(self.model, self.tok, "who is the author?", "zed")
        self.assertEqual(len(g["grid"]), 5)
        self.assertEqual(len(g["grid"][0]), len(g["tokens"]))
        self.assertTrue(all(0 <= p <= 1 for row in g["grid"] for p in row))

    def test_patching_identical_models_is_undefined_and_different_models_recover_at_the_end(self):
        same = self.I.patch_recovery_grid(self.model, self.model, self.tok, "who is the author?", "zed", layers=[0, 1], max_positions=5)
        self.assertTrue(all(v is None for row in same["grid"] for v in row))  # no gap -> no fabricated recovery
        # Search an answer token with a clear gap between the two random models.
        best = None
        for ch in "abcdefghijklmnopqrstuvwxyz":
            out = self.I.patch_recovery_grid(self.other, self.model, self.tok, "who is the author?", ch, layers=[0, 1, 2, 3], max_positions=6, min_gap=0.05)
            if any(v is not None for row in out["grid"] for v in row):
                best = out
                break
        self.assertIsNotNone(best)
        last = best["grid"][-1][-1]  # last layer, last position: patching the final residual replaces the logits' input
        self.assertAlmostEqual(last, 1.0, places=3)
        self.assertEqual(len(best["tokens"]), 6)

    def test_norms_and_ratio(self):
        pre = self.I.last_token_norms(self.model, self.tok, ["who is?", "what is it?"])
        self.assertEqual(len(pre), 5)
        ratio = self.I.norm_ratio_series(pre, pre)
        self.assertTrue(all(abs(r - 1) < 1e-9 for r in ratio))

    def test_feature_units_pipeline(self):
        texts = ["who is zed?", "where is it?", "what is a pen?"]
        units = self.I.select_selective_units(self.model, self.tok, texts[:2], texts[2:], [1, 2], per_layer=2)
        self.assertEqual(set(units), {1, 2})
        out = self.I.feature_units({"pre": self.model, "post": self.other}, self.tok, units, texts, ["c0", "c1", "c2"], top_examples=2)
        self.assertEqual(len(out), 4)
        u = out[0]
        self.assertEqual([c["name"] for c in u["conditions"]], ["pre", "post"])
        h = u["conditions"][0]["histogram"]
        self.assertEqual(len(h["edges"]), len(h["counts"]) + 1)
        self.assertEqual(sum(h["counts"]), u["conditions"][0]["stats"]["count"])
        self.assertEqual(len(u["conditions"][0]["examples"]), 2)
        self.assertEqual(len(u["conditions"][0]["logit_effects"]["promoted"]), 10)
        # artifact must satisfy Evalboard's contract
        from lexsi_unlearn.interp_artifacts import feature_explorer_artifact

        art = feature_explorer_artifact(base_run_id="T", name="f", title="t", units=out, activation_label="a")
        self.assertEqual(art["kind"], "feature_explorer")

    def test_snapshot_model_is_independent(self):
        snap = self.I.snapshot_model(self.model)
        before = next(self.model.parameters()).clone()
        with torch.no_grad():
            next(snap.parameters()).add_(1.0)
        self.assertTrue(torch.equal(before, next(self.model.parameters())))


if __name__ == "__main__":
    unittest.main()


@unittest.skipIf(torch is None, "torch/transformers not installed")
class InterpStageTests(unittest.TestCase):
    """End-to-end artifact build on a tiny RANDOM model, validated against Evalboard's own contract if available."""

    def test_build_all_artifacts_and_validate(self):
        import copy
        import json
        import os
        import tempfile
        from lexsi_unlearn import interp_stage as S

        torch.manual_seed(1)
        cfg = Cohere2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128, sliding_window=8, layer_types=["sliding_attention"] * 3 + ["full_attention"], logit_scale=0.5, pad_token_id=0)
        pre = Cohere2ForCausalLM(cfg).eval()
        post = copy.deepcopy(pre)
        with torch.no_grad():
            post.model.layers[1].mlp.down_proj.weight.add_(torch.randn_like(post.model.layers[1].mlp.down_proj.weight))
        tok = CharTok()
        forget = [{"id": str(i), "question": q, "answer": a} for i, (q, a) in enumerate([("who is zed?", "zed is a painter."), ("where is it?", "in nalmere."), ("what is a pen?", "a pen is a tool.")])]
        retain = [{"id": str(i), "question": q, "answer": "yes."} for i, q in enumerate(["is it red?", "is it big?"])]
        config = {"rmu": {"layer_id": 2, "update_layer_ids": [1, 2]}}
        settings = S.interp_settings(config, {"patching": {"layer_stride": 2, "min_gap": 0.01, "max_prompts": 2}, "features": {"n_forget_texts": 3, "n_retain_texts": 2}})
        arts, errors = S.build_interp_artifacts(config, base_run_id="TESTRUN", tokenizer=tok, model_pre=pre, model_post=post, forget_pairs=forget, retain_pairs=retain, model_id="tiny-random", trace_stage="smoke", trace_phases=("pre_rmu", "post_rmu"), settings=settings)
        self.assertEqual(errors, [])
        kinds = sorted({a["kind"] for a in arts})
        self.assertEqual(kinds, ["distribution", "feature_explorer", "head_grid", "heatmap", "layer_series", "logit_lens_table", "scatter", "token_layer_map"])
        names = {a["id"] for a in arts}
        for expected in ("TESTRUN:rmu-norm-band", "TESTRUN:token-layer-norms", "TESTRUN:logit-lens-table-0", "TESTRUN:head-ablation", "TESTRUN:activation-diff-hist", "TESTRUN:token-relative-change-hist", "TESTRUN:residual-pca-L2", "TESTRUN:rmu-norm-ratio", "TESTRUN:logit-lens-0", "TESTRUN:logit-lens-final-position", "TESTRUN:patching-0", "TESTRUN:mlp-neurons"):
            self.assertIn(expected, names)
        lens = next(a for a in arts if a["id"] == "TESTRUN:logit-lens-0")
        self.assertEqual(lens["trace_ids"][0], "TESTRUN:smoke-pre_rmu:forget:direct:0:pre_rmu")
        json.dumps(arts, allow_nan=False)
        # contract check against the Evalboard validator when the sibling repo is present
        ev = Path(__file__).resolve().parents[2] / "Code" / "modular_dashboard"
        if ev.exists():
            sys.path.insert(0, str(ev))
            from evalboard.interp import validate_artifact

            for a in arts:
                validate_artifact(a)
        # writer
        from lexsi_unlearn.interp_artifacts import InterpWriter

        with tempfile.TemporaryDirectory() as tmp:
            os.environ["OUTPUT_DIR"] = tmp
            try:
                w = InterpWriter("TESTRUN", {})
                self.assertEqual(w.write_all(arts), len(arts))
            finally:
                del os.environ["OUTPUT_DIR"]

    def test_component_failure_is_recorded_not_raised(self):
        from lexsi_unlearn import interp_stage as S

        arts, errors = S.build_interp_artifacts({}, base_run_id="R", tokenizer=CharTok(), model_pre=object(), model_post=type("M", (), {"model": type("I", (), {"layers": [1, 2]})()})(), forget_pairs=[], retain_pairs=[], model_id="x")
        self.assertTrue(errors)
        self.assertEqual(arts, [])

    def test_circuit_scores_json_and_probe_artifacts(self):
        import json
        import tempfile
        from lexsi_unlearn import interp_stage as S

        blob = {"task": "t", "algorithm": "eap-ig", "model": "m", "level": "node", "node_scores": {"a0.h0": 0.9, "m0": 0.4, "a1.h0": 0.1, "m1": 0.05}, "safety_units": {"unit_ids": ["a0.h0", "m0"]}}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(blob, f)
        art = S.circuit_scores_artifact(f.name, base_run_id="R", name="c", title="t")
        self.assertEqual(art["data"]["edge_kind"], "layer_adjacent")
        self.assertEqual([n["in_circuit"] for n in art["data"]["nodes"]], [True, True, False, False])
        blob["safety_units"] = {"unit_ids": []}  # what circuitkit 0.2.3 actually writes for EAP-style node names
        with open(f.name, "w") as g:
            json.dump(blob, g)
        fallback = S.circuit_scores_artifact(f.name, base_run_id="R", name="c2", title="t", top_fraction=0.5)
        self.assertEqual(sum(n["in_circuit"] for n in fallback["data"]["nodes"]), 2)
        self.assertIn("display choice", fallback["notes"])
        rows = [{"layer": 0, "accuracy": 0.5, "auroc": 0.5}, {"layer": 1, "accuracy": 0.9, "auroc": float("nan")}]
        probes = S.probe_artifacts("R", {"post_inject": rows, "post_rmu": rows[:1]})
        self.assertEqual({a["id"] for a in probes}, {"R:probe-auroc", "R:probe-accuracy"})
        auroc = next(a for a in probes if a["id"] == "R:probe-auroc")
        self.assertEqual(auroc["data"]["series"][0]["values"], [0.5, None])  # NaN AUROC stays missing
        self.assertEqual(auroc["data"]["series"][1]["values"], [0.5, None])  # shorter phase padded with missing
