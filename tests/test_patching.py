"""Activation-patch hooks must run only on the prefill forward, not decode steps."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn as nn

from lexsi_unlearn.patching import patch_and_generate  # noqa: E402


class _Layer(nn.Module):
    def forward(self, hidden):
        return hidden


class _Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Layer()])


class _Model(nn.Module):
    """Stands in for an HF causal LM: prefill at full prompt length, then decode at T=1."""

    def __init__(self, width: int = 8):
        super().__init__()
        self.model = _Inner()
        self.width = width
        self.device = torch.device("cpu")
        self.forward_outputs: list[torch.Tensor] = []

    def eval(self):
        return self

    def generate(self, input_ids, attention_mask=None, max_new_tokens=3, do_sample=False):
        del attention_mask, do_sample
        layer = self.model.layers[0]
        # Prefill: full prompt, zeros so a successful patch is visible on the returned tensor.
        prefill = torch.zeros(1, input_ids.shape[1], self.width)
        self.forward_outputs.append(layer(prefill))
        # Decode steps: seq length 1. The old hook indexed the prefill last token here
        # and raised IndexError (e.g. index 390 on a size-1 sequence).
        for _ in range(max_new_tokens):
            step = torch.zeros(1, 1, self.width)
            self.forward_outputs.append(layer(step))
        return torch.cat(
            [input_ids, torch.zeros(1, max_new_tokens, dtype=input_ids.dtype)], dim=1
        )


class _Tokenizer:
    def __init__(self, prefill_len: int = 16):
        self.prefill_len = prefill_len

    def __call__(self, prompt, return_tensors="pt", truncation=True, max_length=512):
        del prompt, return_tensors, truncation, max_length
        ids = torch.arange(self.prefill_len).view(1, self.prefill_len)
        mask = torch.ones(1, self.prefill_len, dtype=torch.long)
        return {"input_ids": ids, "attention_mask": mask}

    def decode(self, ids, skip_special_tokens=True):
        del ids, skip_special_tokens
        return "ok"


class PrefillOnlyPatchHook(unittest.TestCase):
    def test_decode_steps_do_not_index_prefill_position(self):
        prefill_len = 16  # last index 15; a decode hidden of size 1 used to IndexError
        width = 8
        model = _Model(width=width)
        tok = _Tokenizer(prefill_len=prefill_len)
        residual = {0: torch.ones(width)}
        text = patch_and_generate(model, tok, "prompt", residual, max_new_tokens=3)
        self.assertEqual(text, "ok")
        outs = model.forward_outputs
        self.assertEqual(len(outs), 4)  # 1 prefill + 3 decode
        self.assertEqual(tuple(outs[0].shape), (1, prefill_len, width))
        # Prefill last-token residual was written.
        self.assertTrue(torch.allclose(outs[0][0, -1], torch.ones(width)))
        # Earlier prefill positions stay zero.
        self.assertTrue(torch.allclose(outs[0][0, :-1], torch.zeros(prefill_len - 1, width)))
        for out in outs[1:]:
            self.assertEqual(tuple(out.shape), (1, 1, width))
            self.assertTrue(torch.allclose(out, torch.zeros(1, 1, width)))


if __name__ == "__main__":
    unittest.main()
