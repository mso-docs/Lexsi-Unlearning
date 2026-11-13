"""Padding-side-agnostic token positions.

`attention_mask.sum(dim=1) - 1` is the last real token only under RIGHT padding. Under left padding it points
at the wrong column and silently corrupts last-token probes, norms and logit-lens readouts. Everything that
reads "the last prompt token" of a (possibly padded) batch goes through `last_token_index`, and batched
forwards additionally force `padding_side="right"` (`right_padded`) so position ids are unshifted too.
"""

from __future__ import annotations

from contextlib import contextmanager


def last_token_index(attention_mask):
    """Index of the last non-zero mask entry per row, for left, right or no padding. Shape [B], long.

    A fully masked row (should not happen) returns 0 rather than -1.
    """
    import torch

    mask = attention_mask.to(torch.long)
    positions = torch.arange(1, mask.shape[1] + 1, device=mask.device, dtype=torch.long)
    return ((mask != 0).long() * positions).max(dim=1).values.sub(1).clamp(min=0)


@contextmanager
def right_padded(tokenizer):
    """Temporarily force tokenizer.padding_side = 'right' (restored on exit). No-op for objects without the attr."""
    had = hasattr(tokenizer, "padding_side")
    old = getattr(tokenizer, "padding_side", None)
    try:
        if had:
            tokenizer.padding_side = "right"
        yield tokenizer
    finally:
        if had:
            tokenizer.padding_side = old
