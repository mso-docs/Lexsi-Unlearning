"""Extra mech-interp computations for the Evalboard interp tab (token maps, lens tables, head ablation,
activation-difference distributions, PCA projections, per-prompt norm bands).

Everything here is a forward pass (hooks on decoder layers / o_proj) on a Cohere2-style model, reusing the
helpers in `interp.py`. Torch is imported lazily so the pure helpers (PCA, histogram edges, summary
statistics) are importable and testable on the Mac without torch.

Conventions (same as interp.py): residual index 0 = embeddings, index i+1 = output of decoder layer i.
RMU's `layer_id` L therefore corresponds to residual index L+1.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

try:  # numpy is only needed by the compute functions; the Mac can import this module without it
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

from . import interp as I

# ---- pure helpers (no torch) --------------------------------------------------------------------


def mean_std(rows: Sequence[Sequence[float | None]]) -> tuple[list[float | None], list[float | None]]:
    """Column-wise mean and population std over rows; a column with no finite value is None."""
    means, stds = [], []
    for col in zip(*rows):
        vals = [float(v) for v in col if v is not None and math.isfinite(float(v))]
        if not vals:
            means.append(None)
            stds.append(None)
            continue
        m = sum(vals) / len(vals)
        means.append(m)
        stds.append(math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals)))
    return means, stds


def band(means: Sequence[float | None], stds: Sequence[float | None]) -> tuple[list[float | None], list[float | None]]:
    lo = [None if m is None or s is None else m - s for m, s in zip(means, stds)]
    hi = [None if m is None or s is None else m + s for m, s in zip(means, stds)]
    return lo, hi


def shared_edges_from(values_by_series: Sequence[np.ndarray], bins: int = 30, *, symmetric: bool = False, nonnegative: bool = False, pct: float = 99.5):
    """Bin edges covering the central `pct` percentile of all series (values outside are not counted)."""
    pooled = np.concatenate([np.asarray(v, dtype=np.float64).ravel() for v in values_by_series if len(v)])
    pooled = pooled[np.isfinite(pooled)]
    if pooled.size == 0:
        raise ValueError("no finite values to histogram")
    if symmetric:
        hi = float(np.percentile(np.abs(pooled), pct)) or 1e-6
        lo = -hi
    elif nonnegative:
        lo, hi = 0.0, float(np.percentile(pooled, pct)) or 1e-6
    else:
        lo, hi = float(np.percentile(pooled, 100 - pct)), float(np.percentile(pooled, pct))
        if hi <= lo:
            hi = lo + 1e-6
    return [float(e) for e in np.linspace(lo, hi, bins + 1)]


def summary_stats(values: np.ndarray) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    return {"mean": float(v.mean()), "std": float(v.std()), "median": float(np.median(v)), "p95": float(np.percentile(v, 95))} if v.size else {}


def pca_fit(x: np.ndarray, k: int = 2):
    """(mean, components[k, D], explained_variance_ratio[k]) of rows of x via SVD. Needs >= 3 rows."""
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] < 3:
        raise ValueError("PCA needs at least 3 points")
    mu = x.mean(axis=0)
    xc = x - mu
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    total = float((s ** 2).sum()) or 1.0
    k = min(k, vt.shape[0])
    return mu, vt[:k], [float(s[i] ** 2 / total) for i in range(k)]


def pca_project(x: np.ndarray, mu: np.ndarray, comps: np.ndarray) -> np.ndarray:
    out = (np.asarray(x, dtype=np.float64) - mu) @ comps.T
    if out.shape[1] < 2:  # degenerate rank: pad the missing axis with zeros rather than invent structure
        out = np.concatenate([out, np.zeros((out.shape[0], 2 - out.shape[1]))], axis=1)
    return out


def strided(n: int, stride: int, *, include_last: bool = True) -> list[int]:
    stride = max(1, int(stride))
    idx = list(range(0, n, stride))
    if include_last and idx and idx[-1] != n - 1:
        idx.append(n - 1)
    return idx


# ---- forward-pass computations ------------------------------------------------------------------


def token_norm_grid(model, tokenizer, prompt: str, *, max_tokens: int = 64, max_seq_len: int = 512) -> dict[str, Any]:
    """||residual[layer, position]||_2 for the last `max_tokens` prompt tokens: grid[residual_index][token]."""
    import torch

    enc = I._encode(tokenizer, prompt, next(model.parameters()).device, max_seq_len)
    res = I.residual_stream(model, enc)
    ids = enc["input_ids"][0].tolist()[-max_tokens:]
    grid = [[float(v) for v in torch.linalg.vector_norm(r[0, -len(ids):].float(), dim=-1).cpu()] for r in res]
    return {"tokens": I.token_strings(tokenizer, ids), "grid": grid}


def per_prompt_last_token_norms(model, tokenizer, prompts: Sequence[str], *, max_seq_len: int = 512) -> list[list[float]]:
    """[prompt][residual index] last-token L2 norm."""
    import torch

    out = []
    for text in prompts:
        enc = I._encode(tokenizer, text, next(model.parameters()).device, max_seq_len)
        res = I.residual_stream(model, enc)
        out.append([float(torch.linalg.vector_norm(r[0, -1].float()).item()) for r in res])
    return out


def lens_top_rows(model, tokenizer, prompt: str, answer: str, *, top_k: int = 5, layer_stride: int = 2, max_seq_len: int = 512):
    """Logit lens at the final prompt position: rows for the logit_lens_table kind (full-vocab softmax, target rank over the vocab)."""
    import torch

    enc = I._encode(tokenizer, prompt, next(model.parameters()).device, max_seq_len)
    target = I.first_answer_token(tokenizer, answer)
    res = I.residual_stream(model, enc)
    rows = []
    for idx in strided(len(res), layer_stride):
        probs = torch.softmax(I.lens_logits(model, res[idx][0, -1]), dim=-1)
        top = torch.topk(probs, k=min(top_k, probs.shape[-1]))
        p_t = probs[target]
        rows.append({
            "layer": idx,
            "top": [(tokenizer.decode([int(i)]), float(v)) for v, i in zip(top.values.cpu(), top.indices.cpu())],
            "target_rank": int((probs > p_t).sum().item()) + 1,
            "target_value": float(p_t.item()),
        })
    return {"target_token": tokenizer.decode([target]), "rows": rows}


def head_geometry(model) -> tuple[int, int]:
    cfg = model.config
    n_heads = int(cfg.num_attention_heads)
    head_dim = int(getattr(cfg, "head_dim", None) or cfg.hidden_size // n_heads)
    return n_heads, head_dim


def head_ablation_grid(model, tokenizer, prompt: str, answer: str, *, layers: Sequence[int], head_batch: int = 16, max_seq_len: int = 512):
    """Zero-ablation effect per (layer, head): logP(answer token) clean - logP(answer token) with that head's
    attention output zeroed (its slice of the o_proj input). Heads of one layer share one batched forward pass.

    Returns grid[len(layers)][n_heads] in nats (positive = the head supports the answer), plus the clean logP.
    Only the last position's logits are computed (no [B, T, vocab] tensor).
    """
    import torch

    n_heads, head_dim = head_geometry(model)
    device = next(model.parameters()).device
    enc = I._encode(tokenizer, prompt, device, max_seq_len)
    target = I.first_answer_token(tokenizer, answer)
    dec = I.decoder_layers(model)
    scale = I.logit_scale(model)
    trunk = getattr(model, "model", None)
    if trunk is None:
        raise AttributeError("head ablation needs model.model (decoder trunk)")

    def last_logprob(batch):
        with torch.no_grad():
            h = trunk(**batch, use_cache=False).last_hidden_state[:, -1]
            logits = (model.lm_head(h) * scale).float()
            return torch.log_softmax(logits, dim=-1)[:, target]

    clean = float(last_logprob(enc)[0].item())
    grid = []
    for layer in layers:
        o_proj = dec[layer].self_attn.o_proj
        row: list[float | None] = [None] * n_heads
        for start in range(0, n_heads, max(1, head_batch)):
            heads = list(range(start, min(n_heads, start + head_batch)))
            batch = {k: v.expand(len(heads), -1).contiguous() for k, v in enc.items()}

            def pre(_m, args, _heads=heads):
                x = args[0].clone()
                for b, h in enumerate(_heads):
                    x[b, :, h * head_dim:(h + 1) * head_dim] = 0
                return (x,) + tuple(args[1:])

            hook = o_proj.register_forward_pre_hook(pre)
            try:
                lp = last_logprob(batch)
            finally:
                hook.remove()
            for b, h in enumerate(heads):
                row[h] = clean - float(lp[b].item())
        grid.append(row)
    return {"grid": grid, "n_heads": n_heads, "clean_logprob": clean}


def residuals_at(model, tokenizer, prompts: Sequence[str], residual_index: int, *, max_seq_len: int = 512):
    """Per prompt: (all-token residuals [T, D], last-token residual [D]) at one residual index, float32 numpy."""
    out = []
    for text in prompts:
        enc = I._encode(tokenizer, text, next(model.parameters()).device, max_seq_len)
        r = I.residual_stream(model, enc)[residual_index][0].float().cpu().numpy()
        out.append((r, r[-1]))
    return out


def has_gpu_cache() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover
        return False


def free_gpu_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover
        pass
