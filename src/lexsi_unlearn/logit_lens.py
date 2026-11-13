"""Standard unembedding logit lens over residual streams (RQ2).

Tuned Lens (Belrose et al.) was not confirmed to provide a pretrained lens for
Tiny Aya, so this uses the model's own unembedding (`lm_head`, or tied
input embeddings — CircuitKIT notes Tiny Aya uses tied embeddings).

https://circuitkit.lexsi.ai/advanced/tiny-aya/
"""

from __future__ import annotations

from .tokpos import last_token_index

from typing import Any, Sequence


def _unembedding(model):
    if hasattr(model, "lm_head") and model.lm_head is not None:
        return model.lm_head
    # Tied embeddings fallback (common on Tiny Aya / Cohere2).
    if hasattr(model, "get_output_embeddings"):
        head = model.get_output_embeddings()
        if head is not None:
            return head
    raise AttributeError(
        "Could not find lm_head / output embeddings for the logit lens. "
        "Inspect the model modules on Colab rather than guessing names."
    )


def _final_norm(model):
    # Many HF causal LMs keep a final RMS/LayerNorm at model.model.norm.
    inner = getattr(model, "model", model)
    for name in ("norm", "final_layernorm", "ln_f"):
        if hasattr(inner, name):
            return getattr(inner, name)
    # TODO: if none of these exist on Cohere2, print named modules and set
    # the correct final norm. Skipping the norm still produces a usable
    # (if slightly mis-scaled) rank statistic.
    return None


def layer_ranks_for_target(
    model,
    tokenizer,
    prompt: str,
    target_text: str,
    *,
    max_seq_len: int = 512,
) -> list[dict[str, Any]]:
    """Rank of the first target token at each layer, after optional final-norm + unembed."""
    import torch

    model.eval()
    target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]
    if not target_ids:
        raise ValueError("Tokenizer produced no tokens for the target answer.")
    target_id = target_ids[0]

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_len,
    )
    encoded = {k: v.to(model.device) for k, v in encoded.items()}
    with torch.no_grad():
        hidden_states = model(
            **encoded, output_hidden_states=True, use_cache=False
        ).hidden_states
        last = last_token_index(encoded["attention_mask"])
        unembed = _unembedding(model)
        norm = _final_norm(model)
        rows = []
        for layer_i, tensor in enumerate(hidden_states):
            vec = tensor[0, last]
            if norm is not None and layer_i > 0:
                vec = norm(vec)
            logits = unembed(vec).squeeze(0)
            # Rank: how many tokens score strictly higher than the target.
            rank = int((logits > logits[target_id]).sum().item()) + 1
            rows.append(
                {
                    "layer": layer_i,
                    "target_token_id": int(target_id),
                    "rank": rank,
                    "logit": float(logits[target_id].item()),
                    "vocab": int(logits.numel()),
                }
            )
    return rows


def mean_target_rank(
    model,
    tokenizer,
    items: Sequence[dict[str, str]],
) -> list[dict[str, Any]]:
    """Average correct-token rank by layer over a list of {prompt, target} items."""
    import collections

    sums: dict[int, list[int]] = collections.defaultdict(list)
    for item in items:
        ranks = layer_ranks_for_target(model, tokenizer, item["prompt"], item["target"])
        for row in ranks:
            sums[row["layer"]].append(row["rank"])
    return [
        {
            "layer": layer,
            "mean_rank": sum(vals) / len(vals),
            "n": len(vals),
        }
        for layer, vals in sorted(sums.items())
    ]
