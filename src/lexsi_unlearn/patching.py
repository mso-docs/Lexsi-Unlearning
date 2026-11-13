"""Activation patching between the fine-tuned and unlearned models (RQ2).

Patch residuals from the fine-tuned (source) model into the unlearned (target)
model at selected layers, and the reverse, to see where forget-set answers
return.

This uses a standard Hugging Face forward-hook approach on hidden states.
Cohere2 module path names should be confirmed on the loaded model rather
than hard-coded beyond a common Transformer pattern.

TODO: if CircuitKIT exposes a documented patch / intervene API that works on
Tiny Aya, prefer that and record the call here. Docs to check:
  https://circuitkit.lexsi.ai/
  https://circuitkit.lexsi.ai/advanced/tiny-aya/
Until that API is confirmed, keep this HF-hook implementation.
"""

from __future__ import annotations

from .tokpos import last_token_index

from typing import Any, Sequence


def _decoder_layers(model):
    """Best-effort location of the decoder layer list.

    TODO: confirm the path on CohereLabs/tiny-aya-global via
    `list(model.named_modules())[:80]`. Common HF patterns are
    `model.model.layers` or `model.model.decoder.layers`.
    """
    inner = getattr(model, "model", model)
    if hasattr(inner, "layers"):
        return inner.layers
    if hasattr(inner, "decoder") and hasattr(inner.decoder, "layers"):
        return inner.decoder.layers
    raise AttributeError(
        "Could not find decoder layers on the model. Inspect named_modules() "
        "on Colab and update _decoder_layers()."
    )


def collect_layer_residuals(
    model,
    tokenizer,
    prompt: str,
    *,
    layers: Sequence[int] | None = None,
    max_seq_len: int = 512,
) -> dict[int, Any]:
    """Cache last-token residual at each requested layer during a forward pass."""
    import torch

    model.eval()
    encoded = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=max_seq_len
    )
    encoded = {k: v.to(model.device) for k, v in encoded.items()}
    cache: dict[int, Any] = {}
    hooks = []

    def make_hook(idx):
        def _hook(_module, _inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            last = last_token_index(encoded["attention_mask"])
            cache[idx] = hidden[0, last[0]].detach()

        return _hook

    layer_mods = _decoder_layers(model)
    chosen = list(layers) if layers is not None else list(range(len(layer_mods)))
    for idx in chosen:
        hooks.append(layer_mods[idx].register_forward_hook(make_hook(idx)))
    try:
        with torch.no_grad():
            model(**encoded, use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    return cache


def patch_and_generate(
    target_model,
    tokenizer,
    prompt: str,
    residuals: dict[int, Any],
    *,
    max_new_tokens: int = 32,
    max_seq_len: int = 512,
) -> str:
    """Replace last-token residuals in `target_model` with cached source residuals, then generate."""
    import torch

    target_model.eval()
    encoded = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=max_seq_len
    )
    encoded = {k: v.to(target_model.device) for k, v in encoded.items()}
    hooks = []

    def make_hook(idx, vec):
        def _hook(_module, _inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            last = last_token_index(encoded["attention_mask"])
            hidden = hidden.clone()
            hidden[0, last[0]] = vec.to(hidden.dtype).to(hidden.device)
            if isinstance(out, tuple):
                return (hidden,) + out[1:]
            return hidden

        return _hook

    layer_mods = _decoder_layers(target_model)
    for idx, vec in residuals.items():
        hooks.append(layer_mods[idx].register_forward_hook(make_hook(idx, vec)))
    try:
        with torch.no_grad():
            # Hooks apply on the prefill step (the patch of interest).
            generated = target_model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
    finally:
        for h in hooks:
            h.remove()
    new_tokens = generated[0, encoded["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def patch_sweep(
    source_model,
    target_model,
    tokenizer,
    prompt: str,
    target_answer: str,
    *,
    layers: Sequence[int],
    max_new_tokens: int = 32,
) -> list[dict[str, Any]]:
    """One-layer-at-a-time patch from source into target; record generations."""
    rows = []
    for layer in layers:
        cache = collect_layer_residuals(
            source_model, tokenizer, prompt, layers=[layer]
        )
        text = patch_and_generate(
            target_model, tokenizer, prompt, cache, max_new_tokens=max_new_tokens
        )
        rows.append(
            {
                "layer": layer,
                "generation": text,
                "contains_target": target_answer.lower() in text.lower(),
            }
        )
    return rows
