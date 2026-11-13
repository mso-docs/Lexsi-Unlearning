import os as _os
import tempfile as _tempfile

# Tests must never write to the real OpenRouter spend ledger (logs/openrouter_spend.jsonl).
_os.environ.setdefault("OPENROUTER_SPEND_LEDGER", _os.path.join(_tempfile.mkdtemp(prefix="lexsi-test-ledger-"), "openrouter_spend.jsonl"))

"""Tests for src/lexsi_unlearn/interp_extras.py and its wiring in interp_stage (tiny RANDOM tensors/models only).

Pure helper tests need numpy; model tests need torch + transformers. Both skip cleanly when absent (the Mac has neither).
No real model weights are loaded anywhere in this file.
"""
import copy
import json
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None
try:
    import torch
    from transformers import Cohere2Config, Cohere2ForCausalLM
    from test_interp_compute import CharTok
except Exception:  # pragma: no cover
    torch = None

from lexsi_unlearn import interp_extras as X  # noqa: E402  (importable without numpy/torch)
from lexsi_unlearn import interp_stage as S  # noqa: E402


class PureHelperTests(unittest.TestCase):
    def test_mean_std_and_band_keep_missing_columns_missing(self):
        mean, std = X.mean_std([[1.0, None, 3.0], [3.0, None, 5.0]])
        self.assertEqual(mean, [2.0, None, 4.0])
        self.assertEqual(std, [1.0, None, 1.0])
        lo, hi = X.band(mean, std)
        self.assertEqual(lo, [1.0, None, 3.0])
        self.assertEqual(hi, [3.0, None, 5.0])

    def test_strided_always_includes_last_layer(self):
        self.assertEqual(X.strided(5, 2), [0, 2, 4])
        self.assertEqual(X.strided(6, 2), [0, 2, 4, 5])
        self.assertEqual(X.strided(3, 1), [0, 1, 2])

    @unittest.skipIf(np is None, "numpy not installed")
    def test_pca_recovers_dominant_axis_and_projects_consistently(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(40, 6)) * np.array([5.0, 1.0, 0.1, 0.1, 0.1, 0.1])
        mu, comps, ev = X.pca_fit(x, 2)
        self.assertGreater(ev[0], 0.8)
        self.assertGreaterEqual(ev[0], ev[1])
        self.assertGreater(abs(comps[0][0]), 0.9)
        proj = X.pca_project(x, mu, comps)
        self.assertEqual(proj.shape, (40, 2))
        self.assertAlmostEqual(float(proj[:, 0].mean()), 0.0, places=6)
        with self.assertRaises(ValueError):
            X.pca_fit(x[:2])

    @unittest.skipIf(np is None, "numpy not installed")
    def test_shared_edges_and_histogram_counts(self):
        from lexsi_unlearn.interp_artifacts import histogram_counts

        a, b = np.linspace(-1, 1, 1000), np.linspace(-0.2, 0.2, 1000)
        edges = X.shared_edges_from([a, b], 20, symmetric=True)
        self.assertEqual(len(edges), 21)
        self.assertAlmostEqual(edges[0], -edges[-1])
        ca, cb = histogram_counts(a.tolist(), edges), histogram_counts(b.tolist(), edges)
        self.assertLess(sum(ca), 1000)  # the extreme 0.5% is outside the shared range and not counted
        self.assertEqual(sum(cb), 1000)
        pos = X.shared_edges_from([np.abs(a)], 10, nonnegative=True)
        self.assertEqual(pos[0], 0.0)
        with self.assertRaises(ValueError):
            X.shared_edges_from([np.array([np.nan])], 5)

    def test_interp_settings_merge_is_recursive_and_flag_can_disable(self):
        import os

        st = S.interp_settings({"interp": {"extras": {"head_ablation": {"max_prompts": 9}}}}, {"extras": {"projection": {"layers": [3]}}})
        self.assertEqual(st["extras"]["head_ablation"]["max_prompts"], 9)
        self.assertEqual(st["extras"]["head_ablation"]["layer_stride"], 2)  # default kept
        self.assertEqual(st["extras"]["projection"]["layers"], [3])
        self.assertTrue(st["extras"]["enabled"])
        os.environ["LEXSI_INTERP_EXTRAS"] = "0"
        try:
            self.assertFalse(S.interp_settings({})["extras"]["enabled"])
        finally:
            del os.environ["LEXSI_INTERP_EXTRAS"]

    def test_circuit_head_grid_from_scores_is_measured_and_signed(self):
        import tempfile

        blob = {"task": "t", "algorithm": "eap-ig", "node_scores": {"a0.h0": 0.5, "a1.h2": -0.25, "m0": 1.0, "input": 3.0}}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(blob, f)
        art = S.circuit_head_grid_artifact({"post_rmu": f.name}, base_run_id="R", n_heads=4, n_layers=3, model="m")
        self.assertEqual(art["kind"], "head_grid")
        self.assertEqual(art["provenance"], "measured")
        self.assertEqual(art["data"]["n_heads"], 4)
        vals = art["data"]["panels"][0]["values"]
        self.assertEqual(vals[0][0], 0.5)
        self.assertEqual(vals[1][2], -0.25)
        self.assertIsNone(vals[2][3])  # no score -> missing, not zero

    def test_probe_artifacts_carry_provenance_and_chance_line(self):
        rows = [{"layer": 0, "accuracy": 0.5, "auroc": 0.5}, {"layer": 1, "accuracy": 0.9, "auroc": 0.8}]
        arts = {a["id"]: a for a in S.probe_artifacts("R", {"post_rmu": rows})}
        self.assertEqual(arts["R:probe-auroc"]["provenance"], "measured")
        self.assertEqual(arts["R:probe-auroc"]["data"]["reference_lines"], [{"value": 0.5, "label": "chance"}])
        self.assertNotIn("reference_lines", arts["R:probe-accuracy"]["data"])


def _tiny(seed=0):
    torch.manual_seed(seed)
    cfg = Cohere2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128, sliding_window=8, layer_types=["sliding_attention"] * 3 + ["full_attention"], logit_scale=0.5, pad_token_id=0)
    return Cohere2ForCausalLM(cfg).eval()


@unittest.skipIf(torch is None, "torch/transformers not installed")
class ModelComputeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = _tiny()
        cls.tok = CharTok()

    def test_token_norm_grid_shape_and_values(self):
        out = X.token_norm_grid(self.model, self.tok, "who is the author?", max_tokens=6)
        self.assertEqual(len(out["tokens"]), 6)
        self.assertEqual(len(out["grid"]), 5)  # emb + 4 layers
        self.assertTrue(all(len(r) == 6 and all(v > 0 and math.isfinite(v) for v in r) for r in out["grid"]))

    def test_per_prompt_norms_average_to_existing_helper(self):
        from lexsi_unlearn import interp as I

        prompts = ["who is?", "what is it?"]
        per = X.per_prompt_last_token_norms(self.model, self.tok, prompts)
        mean = [sum(c) / len(c) for c in zip(*per)]
        ref = I.last_token_norms(self.model, self.tok, prompts)
        for a, b in zip(mean, ref):
            self.assertAlmostEqual(a, b, places=5)

    def test_lens_rows_sorted_ranked_and_last_layer_matches_model(self):
        out = X.lens_top_rows(self.model, self.tok, "who is the author?", "zed", top_k=3, layer_stride=2)
        self.assertEqual([r["layer"] for r in out["rows"]], [0, 2, 4])
        for r in out["rows"]:
            vals = [v for _, v in r["top"]]
            self.assertEqual(vals, sorted(vals, reverse=True))
            self.assertGreaterEqual(r["target_rank"], 1)
            self.assertTrue(0 <= r["target_value"] <= 1)
        enc = self.tok("who is the author?", return_tensors="pt")
        with torch.no_grad():
            probs = torch.softmax(self.model(**enc, use_cache=False).logits[0, -1].float(), dim=-1)
        top1 = self.tok.decode([int(probs.argmax())])
        self.assertEqual(out["rows"][-1]["top"][0][0], top1)
        self.assertAlmostEqual(out["rows"][-1]["top"][0][1], float(probs.max()), places=4)

    def test_head_ablation_matches_a_manual_single_head_zeroing(self):
        prompt, answer = "who is the author?", "zed"
        layers = [1, 3]
        out = X.head_ablation_grid(self.model, self.tok, prompt, answer, layers=layers, head_batch=3)  # 4 heads -> two chunks
        self.assertEqual(out["n_heads"], 4)
        self.assertEqual((len(out["grid"]), len(out["grid"][0])), (2, 4))
        from lexsi_unlearn import interp as I

        n_heads, hd = X.head_geometry(self.model)
        enc = self.tok(prompt, return_tensors="pt")
        target = I.first_answer_token(self.tok, answer)

        def lp(hook=None):
            h = self.model.model.layers[1].self_attn.o_proj.register_forward_pre_hook(hook) if hook else None
            try:
                with torch.no_grad():
                    return float(torch.log_softmax(self.model(**enc, use_cache=False).logits[0, -1].float(), dim=-1)[target])
            finally:
                if h:
                    h.remove()

        def zero_head2(_m, args):
            x = args[0].clone()
            x[..., 2 * hd:3 * hd] = 0
            return (x,) + tuple(args[1:])

        manual = lp() - lp(zero_head2)
        self.assertAlmostEqual(out["grid"][0][2], manual, places=4)
        self.assertAlmostEqual(out["clean_logprob"], lp(), places=4)

    def test_hooks_are_removed_after_head_ablation(self):
        X.head_ablation_grid(self.model, self.tok, "who is?", "zed", layers=[0])
        for layer in self.model.model.layers:
            self.assertEqual(len(layer.self_attn.o_proj._forward_pre_hooks), 0)


@unittest.skipIf(torch is None, "torch/transformers not installed")
class StageWiringTests(unittest.TestCase):
    def setUp(self):
        self.pre = _tiny(1)
        self.post = copy.deepcopy(self.pre)
        with torch.no_grad():
            self.post.model.layers[1].mlp.down_proj.weight.add_(torch.randn_like(self.post.model.layers[1].mlp.down_proj.weight))
        self.tok = CharTok()
        self.forget = [{"id": str(i), "question": q, "answer": a} for i, (q, a) in enumerate([("who is zed?", "zed is a painter."), ("where is it?", "in nalmere."), ("what is a pen?", "a pen is a tool.")])]
        self.retain = [{"id": str(i), "question": q, "answer": "yes."} for i, q in enumerate(["is it red?", "is it big?"])]
        self.cfg = {"rmu": {"layer_id": 2, "update_layer_ids": [1, 2]}}

    def _build(self, override=None):
        st = S.interp_settings(self.cfg, override)
        return S.build_interp_artifacts(self.cfg, base_run_id="T", tokenizer=self.tok, model_pre=self.pre, model_post=self.post, forget_pairs=self.forget,
                                        retain_pairs=self.retain, model_id="tiny", trace_stage="smoke", trace_phases=("pre_rmu", "post_rmu"), settings=st)

    def test_all_nine_chart_kinds_are_emitted_with_provenance(self):
        arts, errors = self._build({"patching": {"max_prompts": 1, "layer_stride": 2, "min_gap": 0.01}})
        self.assertEqual(errors, [])
        by = {a["id"]: a for a in arts}
        for name, prov in (("token-layer-norms", "measured"), ("logit-lens-table-0", "measured"), ("head-ablation", "measured"), ("activation-diff-hist", "derived"),
                           ("token-relative-change-hist", "derived"), ("residual-pca-L2", "derived"), ("rmu-norm-ratio", "derived"), ("rmu-norm-band", "measured"),
                           ("logit-lens-final-position", "measured"), ("patching-0", "measured")):
            self.assertEqual(by[f"T:{name}"]["provenance"], prov, name)
        for a in arts:
            self.assertIn(a.get("provenance"), ("measured", "derived", "mixed"), a["id"])
        ratio = by["T:rmu-norm-ratio"]["data"]
        self.assertEqual(ratio["reference_lines"], [{"value": 1.0, "label": "no change"}])
        self.assertTrue(all("lo" in s and "hi" in s for s in ratio["series"]))
        tm = by["T:token-layer-norms"]["data"]
        self.assertEqual(tm["layers"], [0, 1, 2, 3, 4])
        self.assertEqual(len(tm["examples"]), 3)  # 2 forget + 1 retain by default
        self.assertEqual([p["name"] for p in tm["examples"][0]["panels"]], ["pre_rmu", "post_rmu"])
        self.assertEqual(tm["examples"][0]["trace_id"], "T:smoke-post_rmu:forget:direct:0:post_rmu")  # linked to the post-phase trace
        hg = by["T:head-ablation"]["data"]
        self.assertEqual(hg["n_heads"], 4)
        self.assertEqual(hg["layers"], [0, 2, 3])  # strided by 2, last layer always included
        self.assertEqual(len(hg["panels"][0]["values"]), 3)
        pca = by["T:residual-pca-L2"]["data"]
        self.assertEqual({g["name"]: len(g["points"]) for g in pca["panels"][0]["groups"]}, {"forget prompts": 3, "retain prompts": 2})
        self.assertEqual(len(pca["explained_variance"]), 2)
        hist = by["T:activation-diff-hist"]["data"]
        self.assertEqual(len(hist["series"][0]["counts"]), len(hist["edges"]) - 1)
        json.dumps(arts, allow_nan=False)

    def test_pca_post_panel_uses_the_pre_fit_basis(self):
        # identical models -> identical panels (same fit, same projection)
        self.post = copy.deepcopy(self.pre)
        arts, errors = self._build()
        pca = next(a for a in arts if a["id"] == "T:residual-pca-L2")["data"]
        a, b = pca["panels"]
        for ga, gb in zip(a["groups"], b["groups"]):
            np_a, np_b = ga["points"], gb["points"]
            for p, q in zip(np_a, np_b):
                self.assertAlmostEqual(p[0], q[0], places=4)
                self.assertAlmostEqual(p[1], q[1], places=4)
        diff = next(a for a in arts if a["id"] == "T:activation-diff-hist")["data"]
        self.assertEqual(diff["series"][0]["stats"]["std"], 0.0)

    def test_extras_can_be_switched_off_per_component_and_globally(self):
        arts, _ = self._build({"extras": {"enabled": False}})
        self.assertFalse({a["kind"] for a in arts} & {"token_layer_map", "logit_lens_table", "head_grid", "distribution", "scatter"})
        arts, _ = self._build({"extras": {"head_ablation": {"enabled": False}, "projection": {"enabled": False}}})
        kinds = {a["kind"] for a in arts}
        self.assertNotIn("head_grid", kinds)
        self.assertNotIn("scatter", kinds)
        self.assertIn("token_layer_map", kinds)

    def test_one_failing_component_is_recorded_and_the_rest_survive(self):
        import unittest.mock as mock

        with mock.patch.object(X, "head_ablation_grid", side_effect=RuntimeError("out of memory")):
            arts, errors = self._build()
        self.assertEqual([e["component"] for e in errors], ["head_ablation"])
        self.assertIn("out of memory", errors[0]["error"])
        kinds = {a["kind"] for a in arts}
        self.assertIn("scatter", kinds)
        self.assertNotIn("head_grid", kinds)

    def test_too_few_prompts_is_a_recorded_error_not_a_crash(self):
        self.retain = []
        arts, errors = self._build()
        comps = {e["component"] for e in errors}
        self.assertTrue({"distributions", "projection"} <= comps)

    def test_artifacts_satisfy_the_evalboard_contract_when_available(self):
        arts, _ = self._build()
        ev = Path(__file__).resolve().parents[2] / "Code" / "modular_dashboard"
        if not ev.exists():
            self.skipTest("modular_dashboard checkout not present")
        sys.path.insert(0, str(ev))
        from evalboard.interp import validate_artifact

        for a in arts:
            validate_artifact(a)


if __name__ == "__main__":
    unittest.main()
