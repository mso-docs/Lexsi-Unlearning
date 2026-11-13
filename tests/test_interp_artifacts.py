import os as _os
import tempfile as _tempfile

# Tests must never write to the real OpenRouter spend ledger (logs/openrouter_spend.jsonl).
_os.environ.setdefault("OPENROUTER_SPEND_LEDGER", _os.path.join(_tempfile.mkdtemp(prefix="lexsi-test-ledger-"), "openrouter_spend.jsonl"))

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lexsi_unlearn import interp_artifacts as ia


class ArtifactBuilderTests(unittest.TestCase):
    def test_none_stays_none_and_nan_never_leaks(self):
        a = ia.layer_series_artifact(base_run_id="R", name="p", title="t", metric="AUROC", layers=[0, 1, 2], series={"x": [0.5, float("nan"), None]}, counts={"x": 4})
        self.assertEqual(a["data"]["series"][0]["values"], [0.5, None, None])
        json.dumps(a, allow_nan=False)

    def test_heatmap_and_ids(self):
        a = ia.heatmap_artifact(base_run_id="R:1", name="lens case/0", title="t", x=["a"], y=["L0", "L1"], panels={"p": [[0.1], [None]]}, value_label="p", x_label="x", y_label="y", value_range=[0, 1])
        self.assertRegex(a["id"], r"^[A-Za-z0-9][A-Za-z0-9_.:+@=-]{0,199}$")
        self.assertEqual(a["data"]["panels"][0]["values"], [[0.1], [None]])

    def test_circuit_nodes_and_honest_edge_kind(self):
        g = ia.circuit_graph_from_scores({"a0.h1": 0.5, "m1": 0.3, "MLP 2": 0.1, "A2.0": 0.2, "input": 9.0}, pruned=["MLP 2"])
        self.assertEqual(g["edge_kind"], "layer_adjacent")
        self.assertEqual(g["reduction"]["nodes_skipped_unparsed"], 1)
        self.assertTrue(any(e["source"] == "a0.h1" and e["target"] == "m1" for e in g["edges"]))
        self.assertFalse(next(n for n in g["nodes"] if n["id"] == "MLP 2")["in_circuit"])
        with self.assertRaises(ValueError):
            ia.circuit_graph_from_scores({"m1": 1}, edge_kind="measured")  # derived edges can't claim to be measured
        measured = ia.circuit_graph_from_scores({"m1": 1, "m2": 1}, edges=[{"source": "m1", "target": "m2", "weight": 0.2}], edge_kind="measured")
        self.assertEqual(measured["edge_kind"], "measured")

    def test_writer_and_trace_id_helper(self):
        with tempfile.TemporaryDirectory() as tmp:
            import os
            os.environ["OUTPUT_DIR"] = tmp
            try:
                w = ia.InterpWriter("RUN", {})
                art = ia.layer_series_artifact(base_run_id="RUN", name="n", title="t", metric="m", layers=[0], series={"s": [1.0]})
                p = w.write(art)
                self.assertEqual(json.loads(p.read_text())["id"], "RUN:n")
                self.assertIn("RUN:n", w.index.read_text())
            finally:
                del os.environ["OUTPUT_DIR"]
        self.assertEqual(ia.trace_ids_for_cases("RUN", "smoke", ["pre_rmu", "post_rmu"], ["forget:direct:0"]),
                         ["RUN:smoke-pre_rmu:forget:direct:0:pre_rmu", "RUN:smoke-post_rmu:forget:direct:0:post_rmu"])


class ExtendedBuilderTests(unittest.TestCase):
    def test_histogram_counts_shared_edges(self):
        edges = [0.0, 1.0, 2.0, 3.0]
        self.assertEqual(ia.histogram_counts([0.0, 0.5, 1.0, 2.9, 3.0, 3.1, -1, None, float("nan")], edges), [2, 1, 2])

    def test_new_kinds_are_json_safe_and_labelled(self):
        nan = float("nan")
        arts = [
            ia.token_layer_map_artifact(base_run_id="R", name="tlm", title="t", layers=[0, 1], value_label="norm",
                                        examples=[{"id": "e0", "tokens": ["a", "b"], "panels": {"p": [[1.0, nan], [2.0, 3.0]]}}]),
            ia.logit_lens_table_artifact(base_run_id="R", name="lens", title="t", target_token="x",
                                         panels={"p": [{"layer": 0, "top": [("a", 0.5), ("b", nan)], "target_rank": 3}]}, provenance="derived"),
            ia.head_grid_artifact(base_run_id="R", name="hg", title="t", layers=[0], n_heads=2, panels={"p": [[0.1, None]]}, value_label="effect"),
            ia.distribution_artifact(base_run_id="R", name="d", title="t", edges=[0, 1, 2], series={"a": [1, 2]}, value_label="v", markers=[(0.0, "zero")]),
            ia.scatter_artifact(base_run_id="R", name="sc", title="t", method="PCA", panels={"p": {"g": [[0.0, 1.0]]}}),
        ]
        self.assertEqual([a["kind"] for a in arts], ["token_layer_map", "logit_lens_table", "head_grid", "distribution", "scatter"])
        for a in arts:
            json.dumps(a, allow_nan=False)
        self.assertEqual(arts[0]["data"]["examples"][0]["panels"][0]["values"][0], [1.0, None])
        self.assertEqual(arts[1]["data"]["panels"][0]["layers"][0]["top"], [{"token": "a", "value": 0.5}])
        self.assertEqual(arts[1]["provenance"], "derived")
        self.assertEqual(arts[4]["provenance"], "derived")  # projections are never presented as raw measurements

    def test_layer_series_bands_and_reference_lines(self):
        a = ia.layer_series_artifact(base_run_id="R", name="n", title="t", metric="m", layers=[0, 1], series={"s": [1.0, 2.0]},
                                     bands={"s": ([0.5, 1.5], [1.5, None])}, reference_lines=[(1.0, "ref")])
        s = a["data"]["series"][0]
        self.assertEqual((s["lo"], s["hi"]), ([0.5, 1.5], [1.5, None]))
        self.assertEqual(a["data"]["reference_lines"], [{"value": 1.0, "label": "ref"}])

    def test_artifacts_pass_evalboard_validator_when_available(self):
        dash = Path.home() / "Documents" / "Code" / "modular_dashboard"
        if not (dash / "evalboard" / "interp.py").exists():
            self.skipTest("evalboard not found next to this repo")
        sys.path.insert(0, str(dash))
        try:
            from evalboard import interp as eb
        finally:
            sys.path.remove(str(dash))
        art = ia.distribution_artifact(base_run_id="R", name="d", title="t", edges=[0, 1, 2], series={"a": [1, 2]}, value_label="v")
        eb.validate_artifact(art)


if __name__ == "__main__":
    unittest.main()
