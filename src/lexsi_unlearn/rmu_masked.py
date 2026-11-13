"""RMU whose FORGET loss ignores the shared system-preamble tokens (candidate fix, 2026-11-12).

Why: Tiny Aya's chat template prepends a ~360-token system preamble to every example, so ~83% of the positions in a
forget batch are IDENTICAL to the retain batches, and positions 0-1 (BOS / start-of-turn attention sinks) have a residual
norm of ~23 (MSE to the control vector ~533 per token). Measured on the injected checkpoint (layer 18, batch of 8 forget
rows): overall forget MSE 3.50, of which the two sink tokens contribute ~2.5. SafeTune's rmu_unlearn averages the
forget MSE over ALL positions, so (a) the loss cannot go down (the sinks cannot move without hurting retain) and (b) the
gradient on the ~60 tokens that carry the fact is diluted ~7x. Four runs at lr 5e-5 / 5e-4, alpha 1200 / 100 left the forget
loss flat at 3.54-3.55 and forget ROUGE-L unchanged.

This module is SafeTune's loop (same hooks, same parameter selection, same AdamW) with one change: the forget MSE is
taken only over real (non-pad) tokens at index >= content_start, i.e. from the first user-turn token on. The retain loss
is unchanged (all positions), so the preamble and the sinks stay pinned to the frozen model.
"""
from __future__ import annotations

import copy
from typing import Iterable

import torch
import torch.nn.functional as F


def content_start(tokenizer) -> int:
    """Index of the first token of the first user turn (0 when the template has no user-turn marker)."""
    msgs = [{"role": "user", "content": "x"}]
    if not (hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template):
        return 0
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    for marker in ("<|USER_TOKEN|>", "<|im_start|>user", "<start_of_turn>user"):
        mid = tokenizer.convert_tokens_to_ids(marker)
        if mid is not None and mid != getattr(tokenizer, "unk_token_id", None) and mid in ids:
            return max(0, ids.index(mid) - 1)  # include the START_OF_TURN token before it
    return 0


def rmu_unlearn_masked(
    model,
    retain_batches: Iterable[dict],
    forget_batches: Iterable[dict],
    *,
    layer_id: int,
    update_layer_ids: list[int],
    alpha: float,
    steering_coeff: float,
    lr: float,
    forget_start: int,
    max_num_batches: int | None = None,
):
    from safetune.interventions.unlearn.rmu import RMUConfig, _forward_with_cache, _select_params
    from safetune._refusal_helpers import _get_decoder_layers

    cfg = RMUConfig(layer_id=layer_id, update_layer_ids=list(update_layer_ids), steering_coeff=steering_coeff,
                    alpha=alpha, lr=lr, max_num_batches=max_num_batches)
    frozen = copy.deepcopy(model)
    frozen.eval()
    for p in frozen.parameters():
        p.requires_grad_(False)
    model.train()
    params = _select_params(model, cfg)
    opt = torch.optim.AdamW(params, lr=lr)
    module, frozen_module = _get_decoder_layers(model)[layer_id], _get_decoder_layers(frozen)[layer_id]
    ref = params[0]
    hidden = getattr(getattr(model, "config", None), "hidden_size", None) or ref.shape[0]
    rv = torch.rand(1, 1, hidden, dtype=ref.dtype, device=ref.device)
    control = rv / torch.norm(rv) * steering_coeff
    dev = ref.device
    steps = 0
    import logging
    log = logging.getLogger("safetune.interventions.unlearn.rmu")  # same name/format the loss logger in unlearn.py listens to
    for fb, rb in zip(forget_batches, retain_batches):
        if max_num_batches is not None and steps >= max_num_batches:
            break
        fb = {k: v.to(dev) for k, v in fb.items()}
        rb = {k: v.to(dev) for k, v in rb.items()}
        act_f = _forward_with_cache(model, fb, module, no_grad=False)
        keep = fb["attention_mask"].bool().clone()
        keep[:, :forget_start] = False
        if not keep.any():
            raise ValueError(f"forget_start={forget_start} masks every token of the forget batch")
        per_pos = F.mse_loss(act_f, control.expand_as(act_f).to(act_f.dtype), reduction="none").mean(-1)
        forget_loss = per_pos[keep].mean()
        act_r = _forward_with_cache(model, rb, module, no_grad=False)
        act_r0 = _forward_with_cache(frozen, rb, frozen_module, no_grad=True)
        retain_loss = alpha * F.mse_loss(act_r, act_r0.to(act_r.dtype))
        loss = forget_loss + retain_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        steps += 1
        log.info("rmu_unlearn: step %d -- loss=%.4g forget=%.4g retain=%.4g", steps, loss.item(), forget_loss.item(), retain_loss.item())
    return model
