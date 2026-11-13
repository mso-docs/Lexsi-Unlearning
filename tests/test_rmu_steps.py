"""RMU must take max_num_batches optimizer steps (cycling the 5 forget batches) and keep the incoming dtype.

Regression for 2026-11-12: 40 forget rows / batch_size 8 = 5 batches, SafeTune zips forget with retain and stops at the
shorter, so `max_num_batches: 80` ran 5 steps and RMU changed ~0.2% of the weights (post-RMU metrics == post-inject).
Tiny RANDOM models only; the torch/safetune parts skip when those packages are absent (they run in the Spark container).
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lexsi_unlearn.unlearn import cycle_to  # noqa: E402

try:
    import torch
    from transformers import Cohere2Config, Cohere2ForCausalLM
    import safetune  # noqa: F401
except Exception:  # pragma: no cover
    torch = None


class CycleTo(unittest.TestCase):
    def test_five_forget_batches_become_eighty(self):
        out = cycle_to(["a", "b", "c", "d", "e"], 80)
        self.assertEqual(len(out), 80)
        self.assertEqual(out[:6], ["a", "b", "c", "d", "e", "a"])

    def test_shorter_request_truncates_and_empty_raises(self):
        self.assertEqual(cycle_to([1, 2, 3], 2), [1, 2])
        with self.assertRaises(ValueError):
            cycle_to([], 4)


def _tiny():
    cfg = Cohere2Config(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
                        num_key_value_heads=2, max_position_embeddings=64, sliding_window=16, layer_types=["sliding_attention"] * 3 + ["full_attention"])
    torch.manual_seed(0)
    return Cohere2ForCausalLM(cfg)


def _batches(n, bs=2, seq=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        ids = torch.randint(3, 100, (bs, seq), generator=g)
        out.append({"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()})
    return out


@unittest.skipIf(torch is None, "torch/transformers/safetune not installed")
class RunRmuSteps(unittest.TestCase):
    def test_steps_equal_max_num_batches_even_with_few_batches(self):
        from lexsi_unlearn.unlearn import run_rmu

        model = _tiny().to(torch.bfloat16)
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        hp = {"layer_id": 2, "update_layer_ids": [1, 2], "alpha": 10.0, "steering_coeff": 5.0, "lr": 5e-4, "max_num_batches": 20}
        out = run_rmu(model, forget_batches=_batches(5, seed=1), retain_batches=_batches(9, seed=2), hparams=hp)
        rep = out.rmu_report
        self.assertEqual(rep["steps"], 20)  # was min(5, 9) = 5 before the fix
        self.assertEqual(rep["steps_requested"], 20)
        self.assertEqual(next(out.parameters()).dtype, torch.bfloat16)  # fp32 training copy cast back
        changed = {n for n, p in out.named_parameters() if not torch.equal(p, before[n])}
        self.assertTrue(changed)
        self.assertTrue(all("mlp.down_proj" in n and (".layers.1." in n or ".layers.2." in n) for n in changed), changed)
        self.assertGreater(rep["weight_delta"]["frac_changed"], 0.0)

    def test_more_steps_move_the_weights_further(self):
        from lexsi_unlearn.unlearn import run_rmu

        hp = {"layer_id": 2, "update_layer_ids": [1, 2], "alpha": 10.0, "steering_coeff": 5.0, "lr": 5e-4}
        deltas = []
        for steps in (5, 40):
            out = run_rmu(_tiny(), forget_batches=_batches(5, seed=1), retain_batches=_batches(5, seed=2),
                          hparams={**hp, "max_num_batches": steps})
            self.assertEqual(out.rmu_report["steps"], steps)
            deltas.append(out.rmu_report["weight_delta"]["rel_fro"])
        self.assertGreater(deltas[1], 2.0 * deltas[0])


if __name__ == "__main__":
    unittest.main()


@unittest.skipIf(torch is None, "torch/transformers/safetune not installed")
class MaskedForgetLoss(unittest.TestCase):
    def test_forget_loss_ignores_the_first_k_positions(self):
        """Huge activations in the first positions must not enter the forget loss when forget_start covers them."""
        from lexsi_unlearn.rmu_masked import rmu_unlearn_masked

        model = _tiny()
        losses = []
        import logging

        class H(logging.Handler):
            def emit(self, r):  # noqa: ANN001
                if isinstance(r.args, tuple) and len(r.args) == 4:
                    losses.append(r.args)

        lg = logging.getLogger("safetune.interventions.unlearn.rmu")
        lg.addHandler(H()); lg.setLevel(logging.INFO)
        kw = dict(layer_id=2, update_layer_ids=[1, 2], alpha=10.0, steering_coeff=5.0, lr=1e-3, max_num_batches=6)
        fb, rb = _batches(3, seed=1), _batches(3, seed=2)
        rmu_unlearn_masked(model, rb * 2, fb * 2, forget_start=0, **kw)
        full = losses[0][2]
        losses.clear()
        rmu_unlearn_masked(_tiny(), rb * 2, fb * 2, forget_start=6, **kw)  # seq len 8 -> only the last 2 positions count
        self.assertEqual(len(losses), 6)
        self.assertNotAlmostEqual(full, losses[0][2], places=4)

    def test_run_rmu_masked_path_takes_all_steps_and_needs_a_tokenizer(self):
        from lexsi_unlearn.unlearn import run_rmu

        hp = {"layer_id": 2, "update_layer_ids": [1, 2], "alpha": 10.0, "steering_coeff": 5.0, "lr": 1e-3, "max_num_batches": 12,
              "forget_mask": "content"}
        with self.assertRaises(ValueError):
            run_rmu(_tiny(), forget_batches=_batches(3), retain_batches=_batches(3), hparams=hp)

        class Tok:  # content_start() only needs these
            chat_template = None

        out = run_rmu(_tiny(), forget_batches=_batches(3, seed=1), retain_batches=_batches(3, seed=2), hparams=hp, tokenizer=Tok())
        self.assertEqual(out.rmu_report["steps"], 12)
