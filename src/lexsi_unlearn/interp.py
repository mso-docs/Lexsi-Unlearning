"""Mechanistic-interpretability computations that feed the Evalboard interp tab (RQ2).

Everything here is plain Hugging Face + forward hooks on a Cohere2-style decoder
(`model.model.layers[i]`, `layers[i].mlp.down_proj`, `model.model.norm`, `model.lm_head`,
`model.logit_scale`). Layer outputs are hooked directly instead of using
`output_hidden_states`, because Hugging Face's last hidden state is already final-normed.

Conventions (recorded in artifact notes):
  * residual index 0 = embedding output; index i+1 = output of decoder layer i.
  * logit lens = lm_head(final_norm(residual)) * logit_scale, softmax over the full vocabulary.
  * the "answer token" is the first token of the reference answer after the chat generation prompt.
  * "neuron" = a hidden unit of the MLP (input to down_proj, after act(gate)*up).
  * direct logit effect of a neuron = logit_scale * lm_head.weight @ down_proj.weight[:, j]
    (direct path only; the final norm's per-position scaling and all later layers are ignored).

Torch is imported lazily so the Mac can import the artifact helpers without it.
These functions are CPU-testable on a tiny random Cohere2 (tests/test_interp_compute.py).
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Mapping, Sequence

try:  # numpy is only needed by the compute functions; the Mac can import this module without it
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

# ---- model access -----------------------------------------------------------------------------


def decoder_layers(model):
    inner = getattr(model, "model", model)
    if hasattr(inner, "layers"):
        return inner.layers
    if hasattr(inner, "decoder") and hasattr(inner.decoder, "layers"):
        return inner.decoder.layers
    raise AttributeError("Could not find decoder layers; inspect named_modules().")


def final_norm(model):
    inner = getattr(model, "model", model)
    for name in ("norm", "final_layernorm", "ln_f"):
        if hasattr(inner, name):
            return getattr(inner, name)
    return None


def logit_scale(model) -> float:
    return float(getattr(model, "logit_scale", None) or getattr(getattr(model, "config", None), "logit_scale", None) or 1.0)


def _embed(model):
    inner = getattr(model, "model", model)
    return inner.embed_tokens


def _encode(tokenizer, text: str, device, max_seq_len: int = 512):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_seq_len)
    return {k: v.to(device) for k, v in enc.items()}


def token_strings(tokenizer, input_ids: Sequence[int]) -> list[str]:
    return [tokenizer.decode([int(i)]) for i in input_ids]


def first_answer_token(tokenizer, answer: str) -> int:
    ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError("tokenizer produced no tokens for the answer")
    return int(ids[0])


# ---- residual stream --------------------------------------------------------------------------


def residual_stream(model, enc) -> list:
    """[embeddings, layer0 out, ..., layerN-1 out], each [B, T, D] (detached, same device/dtype)."""
    import torch

    cache: dict[int, Any] = {}
    hooks = []

    def make(idx):
        def hook(_m, _inp, out):
            cache[idx] = (out[0] if isinstance(out, tuple) else out).detach()

        return hook

    layers = decoder_layers(model)
    hooks.append(_embed(model).register_forward_hook(lambda _m, _i, out: cache.__setitem__(-1, out.detach())))
    for i, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(make(i)))
    try:
        with torch.no_grad():
            model(**enc, use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    return [cache[-1]] + [cache[i] for i in range(len(layers))]


def lens_logits(model, hidden):
    """Apply final norm + unembedding + logit_scale to hidden [..., D] -> [..., V] (float32)."""
    import torch

    norm = final_norm(model)
    with torch.no_grad():
        h = norm(hidden) if norm is not None else hidden
        return (model.lm_head(h) * logit_scale(model)).float()


# ---- per-layer scalars ------------------------------------------------------------------------


def last_token_norms(model, tokenizer, texts: Sequence[str], *, max_seq_len: int = 512) -> list[float]:
    """Mean L2 norm of the last-prompt-token residual at each residual index, over `texts`."""
    import torch

    sums = None
    for text in texts:
        enc = _encode(tokenizer, text, next(model.parameters()).device, max_seq_len)
        res = residual_stream(model, enc)
        norms = [float(torch.linalg.vector_norm(r[0, -1].float()).item()) for r in res]
        sums = norms if sums is None else [a + b for a, b in zip(sums, norms)]
    return [s / len(texts) for s in sums] if sums else []


def norm_ratio_series(pre: Sequence[float], post: Sequence[float]) -> list[float | None]:
    return [(b / a) if a else None for a, b in zip(pre, post)]


# ---- logit lens grid --------------------------------------------------------------------------


def lens_probability_grid(model, tokenizer, prompt: str, answer: str, *, max_seq_len: int = 512) -> dict[str, Any]:
    """P(first answer token) under the logit lens at every (residual index, prompt position)."""
    import torch

    device = next(model.parameters()).device
    enc = _encode(tokenizer, prompt, device, max_seq_len)
    target = first_answer_token(tokenizer, answer)
    res = residual_stream(model, enc)
    grid = []
    for r in res:
        logits = lens_logits(model, r[0])  # [T, V]
        probs = torch.softmax(logits, dim=-1)[:, target]
        grid.append([float(p) for p in probs.cpu()])
    return {"tokens": token_strings(tokenizer, enc["input_ids"][0].tolist()), "target_token": tokenizer.decode([target]), "grid": grid}


# ---- activation patching ----------------------------------------------------------------------


def _answer_logprob(logits, target: int) -> float:
    import torch

    return float(torch.log_softmax(logits.float(), dim=-1)[target].item())


def patch_recovery_grid(
    source_model,
    target_model,
    tokenizer,
    prompt: str,
    answer: str,
    *,
    layers: Sequence[int],
    positions: Sequence[int] | None = None,
    max_positions: int = 24,
    min_gap: float = 1.0,
    max_seq_len: int = 512,
) -> dict[str, Any]:
    """Residual patching source -> target at one (decoder layer, position) per run.

    recovery = (lp_patched - lp_target) / (lp_source - lp_target) with lp = log P(first answer
    token | prompt) at the final position. Cells are None when the source/target log-prob gap is
    below `min_gap` nats (recovery is undefined there). Values are NOT clipped.
    Default positions: the last `max_positions` prompt positions. Each layer is one batched
    forward pass (one row per patched position).
    """
    import torch

    t_dev = next(target_model.parameters()).device
    s_dev = next(source_model.parameters()).device
    enc_t = _encode(tokenizer, prompt, t_dev, max_seq_len)
    enc_s = _encode(tokenizer, prompt, s_dev, max_seq_len)
    ids = enc_t["input_ids"]
    T = ids.shape[1]
    if positions is None:
        positions = list(range(max(0, T - max_positions), T))
    positions = [int(p) for p in positions]
    target = first_answer_token(tokenizer, answer)
    with torch.no_grad():
        lp_src = _answer_logprob(source_model(**enc_s, use_cache=False).logits[0, -1], target)
        lp_tgt = _answer_logprob(target_model(**enc_t, use_cache=False).logits[0, -1], target)
    src_res = residual_stream(source_model, enc_s)  # index i+1 = layer i output
    gap = lp_src - lp_tgt
    layer_mods = decoder_layers(target_model)
    rows: list[list[float | None]] = []
    for layer in layers:
        if gap < min_gap:
            rows.append([None] * len(positions))
            continue
        batch_ids = ids.repeat(len(positions), 1)
        mask = enc_t["attention_mask"].repeat(len(positions), 1)
        src_vec = src_res[layer + 1][0].to(t_dev)  # [T, D]
        pos_t = torch.tensor(positions, device=t_dev)

        def hook(_m, _inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            hidden = hidden.clone()
            rows_ix = torch.arange(len(positions), device=hidden.device)
            hidden[rows_ix, pos_t] = src_vec[pos_t].to(hidden.dtype)
            return (hidden,) + tuple(out[1:]) if isinstance(out, tuple) else hidden

        h = layer_mods[layer].register_forward_hook(hook)
        try:
            with torch.no_grad():
                logits = target_model(input_ids=batch_ids, attention_mask=mask, use_cache=False).logits[:, -1]
        finally:
            h.remove()
        row = []
        for i in range(len(positions)):
            lp = _answer_logprob(logits[i], target)
            row.append((lp - lp_tgt) / gap)
        rows.append(row)
    return {
        "tokens": token_strings(tokenizer, ids[0, positions].tolist()),
        "positions": positions,
        "layers": [int(l) for l in layers],
        "grid": rows,
        "logprob_source": lp_src,
        "logprob_target": lp_tgt,
        "min_gap": min_gap,
        "target_token": tokenizer.decode([target]),
    }


# ---- neuron / feature explorer ----------------------------------------------------------------


def mlp_activations(model, tokenizer, texts: Sequence[str], layer_units: Mapping[int, Sequence[int]], *, max_seq_len: int = 512):
    """Per text, per layer: array [T, len(units)] of MLP hidden activations (input to down_proj).

    Returns (tokens_per_text, acts) with acts[text_idx][layer] -> np.ndarray float32.
    """
    import torch

    device = next(model.parameters()).device
    layers = decoder_layers(model)
    all_tokens, all_acts = [], []
    for text in texts:
        enc = _encode(tokenizer, text, device, max_seq_len)
        store: dict[int, Any] = {}
        hooks = []
        for layer, units in layer_units.items():
            idx = torch.tensor(list(units), device=device)

            def pre(_m, args, _layer=layer, _idx=idx):
                store[_layer] = args[0][0].detach().float().index_select(-1, _idx).cpu().numpy()

            hooks.append(layers[layer].mlp.down_proj.register_forward_pre_hook(pre))
        try:
            with torch.no_grad():
                model(**enc, use_cache=False)
        finally:
            for h in hooks:
                h.remove()
        all_tokens.append(token_strings(tokenizer, enc["input_ids"][0].tolist()))
        all_acts.append(store)
    return all_tokens, all_acts


def select_selective_units(
    model,
    tokenizer,
    forget_texts: Sequence[str],
    retain_texts: Sequence[str],
    layers: Sequence[int],
    *,
    per_layer: int = 2,
    max_seq_len: int = 512,
) -> dict[int, list[int]]:
    """Pick, per layer, the MLP neurons with the largest |mean forget - mean retain| last-token activation.

    Selection is a heuristic for choosing what to display (it does not claim these neurons store the
    facts). Only the pre-RMU model is used so before/after compare the same units.
    """
    import torch

    device = next(model.parameters()).device
    layer_mods = decoder_layers(model)

    def collect(texts):
        out: dict[int, list] = {l: [] for l in layers}
        for text in texts:
            enc = _encode(tokenizer, text, device, max_seq_len)
            hooks = []
            for l in layers:
                hooks.append(layer_mods[l].mlp.down_proj.register_forward_pre_hook(lambda _m, a, _l=l: out[_l].append(a[0][0, -1].detach().float().cpu().numpy())))
            try:
                with torch.no_grad():
                    model(**enc, use_cache=False)
            finally:
                for h in hooks:
                    h.remove()
        return {l: np.stack(v) for l, v in out.items()}

    f, r = collect(forget_texts), collect(retain_texts)
    chosen = {}
    for l in layers:
        diff = np.abs(f[l].mean(0) - r[l].mean(0))
        chosen[l] = [int(i) for i in np.argsort(-diff)[:per_layer]]
    return chosen


def direct_logit_effects(model, tokenizer, layer: int, unit: int, *, k: int = 10) -> dict[str, Any]:
    """Top promoted / suppressed vocabulary tokens for one MLP neuron's output direction (direct path)."""
    import torch

    w = decoder_layers(model)[layer].mlp.down_proj.weight[:, unit].detach().float()  # [D]
    head = model.lm_head.weight.detach().float()  # [V, D]
    with torch.no_grad():
        eff = (head.to(w.device) @ w) * logit_scale(model)
        top = torch.topk(eff, k)
        bot = torch.topk(-eff, k)
    tok = lambda i: tokenizer.decode([int(i)])
    return {
        "method": "logit_scale * lm_head.weight @ down_proj.weight[:, j]; direct path only (final norm and later layers ignored)",
        "promoted": [{"token": tok(i), "value": float(v)} for v, i in zip(top.values.cpu(), top.indices.cpu())],
        "suppressed": [{"token": tok(i), "value": float(-v)} for v, i in zip(bot.values.cpu(), bot.indices.cpu())],
    }


def histogram(values: np.ndarray, edges: np.ndarray) -> dict[str, Any]:
    counts, _ = np.histogram(values, bins=edges)
    return {"edges": [float(e) for e in edges], "counts": [int(c) for c in counts]}


def condition_for_unit(
    model,
    tokenizer,
    name: str,
    layer: int,
    unit_pos: int,
    unit: int,
    texts: Sequence[str],
    case_ids: Sequence[str],
    tokens: Sequence[Sequence[str]],
    acts: Sequence[Mapping[int, np.ndarray]],
    edges: np.ndarray,
    *,
    top_examples: int = 4,
    effects_k: int = 10,
    trace_ids: Sequence[str | None] | None = None,
) -> dict[str, Any]:
    per_text = [a[layer][:, unit_pos] for a in acts]
    flat = np.concatenate(per_text)
    order = np.argsort([-float(np.max(np.abs(p))) for p in per_text])[:top_examples]
    examples = []
    for i in order:
        ex = {"tokens": list(tokens[i]), "activations": [float(v) for v in per_text[i]], "case_id": case_ids[i]}
        if trace_ids and trace_ids[i]:
            ex["trace_id"] = trace_ids[i]
        examples.append(ex)
    return {
        "name": name,
        "stats": {"mean": float(flat.mean()), "max": float(flat.max()), "min": float(flat.min()), "count": int(flat.size)},
        "histogram": histogram(flat, edges),
        "examples": examples,
        "logit_effects": direct_logit_effects(model, tokenizer, layer, unit, k=effects_k),
    }


def shared_edges(arrays: Sequence[np.ndarray], bins: int = 24) -> np.ndarray:
    lo = min(float(a.min()) for a in arrays)
    hi = max(float(a.max()) for a in arrays)
    if hi <= lo:
        hi = lo + 1e-6
    return np.linspace(lo, hi, bins + 1)


def feature_units(
    models: Mapping[str, Any],
    tokenizer,
    units_by_layer: Mapping[int, Sequence[int]],
    texts: Sequence[str],
    case_ids: Sequence[str],
    *,
    trace_ids_by_condition: Mapping[str, Sequence[str | None]] | None = None,
    top_examples: int = 4,
    max_seq_len: int = 512,
) -> list[dict[str, Any]]:
    """feature_explorer `units` comparing the same MLP neurons across named models (conditions)."""
    per_model = {}
    for name, model in models.items():
        per_model[name] = mlp_activations(model, tokenizer, texts, units_by_layer, max_seq_len=max_seq_len)
    units = []
    for layer, unit_ids in units_by_layer.items():
        for pos, unit in enumerate(unit_ids):
            flat = [np.concatenate([a[layer][:, pos] for a in per_model[n][1]]) for n in models]
            edges = shared_edges(flat)
            conds = [
                condition_for_unit(
                    models[n], tokenizer, n, layer, pos, unit, texts, case_ids, per_model[n][0], per_model[n][1], edges,
                    top_examples=top_examples,
                    trace_ids=(trace_ids_by_condition or {}).get(n),
                )
                for n in models
            ]
            units.append({"id": f"L{layer}:n{unit}", "kind": "mlp_neuron", "layer": int(layer), "index": int(unit), "label": f"layer {layer} MLP neuron {unit}", "conditions": conds})
    return units


# ---- probes (artifact-friendly wrapper over probes.py) --------------------------------------


def probe_series(rows: Sequence[Mapping[str, Any]], metric: str) -> list[float | None]:
    by_layer = {int(r["layer"]): r.get(metric) for r in rows}
    return [by_layer.get(l) for l in range(max(by_layer) + 1)] if by_layer else []


def snapshot_model(model, *, to_float32: bool = True):
    """Deep copy of `model` (SafeTune trains in place); fp32 like the trainer's upcast so pre/post are comparable."""
    clone = copy.deepcopy(model)
    return clone.float() if to_float32 else clone
