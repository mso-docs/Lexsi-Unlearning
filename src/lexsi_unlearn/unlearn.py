"""SafeTune unlearning wrappers (RMU primary, NPO optional).

Verified docs:
  https://safetune.lexsi.ai/user-guide/unlearn/
  https://safetune.lexsi.ai/user-guide/unlearn/rmu/
  https://safetune.lexsi.ai/user-guide/unlearn/npo/

Confirmed call shape from the unlearn guide:
    from safetune.runner import unlearn
    trainer = unlearn.RMUTrainer(model)
    unlearned = trainer.unlearn(forget=forget_batches, retain=retain_batches)

forget and retain batches are tokenized dicts with input_ids, attention_mask,
and labels. Do not use unlearn.load_unlearn_data as the TOFU forget set; that
helper is a generic loader, not the TOFU protocol.

Default RMU hyperparameters (flag layer_id for retuning on 36-layer Tiny Aya):
    layer_id=7, update layers [5, 6, 7], alpha=1200, steering_coeff=20,
    lr=5e-5, max_num_batches=80.
"""

from __future__ import annotations

from typing import Any, Iterator

from .data import apply_chat_template, question_of, answer_of


def tokenize_unlearn_batches(
    tokenizer,
    rows,
    *,
    max_seq_len: int = 512,
    batch_size: int = 2,
) -> list[dict[str, Any]]:
    """Build the batch dicts SafeTune's unlearn trainers expect.

    Docs: https://safetune.lexsi.ai/user-guide/unlearn/
    """
    import torch

    batches: list[dict[str, Any]] = []
    texts = []
    for row in rows:
        question, answer = question_of(row), answer_of(row)
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        else:
            text = f"{question}\n{answer}"
        texts.append(text)

    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        encoded = tokenizer(
            chunk,
            padding=True,
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt",
        )
        encoded["labels"] = encoded["input_ids"].clone()
        # Mask pad tokens in labels.
        pad_id = tokenizer.pad_token_id
        if pad_id is not None:
            encoded["labels"][encoded["input_ids"] == pad_id] = -100
        batches.append(encoded)
    return batches


# Verified on safetune 0.1.18 (Spark container, 2026-11-12):
# RMUTrainer.__init__(model, *, layer_id=7, update_layer_ids=None,
#   max_num_batches=80, lr=5e-5, alpha=1200.0, steering_coeff=20.0, **kwargs)
# NPOTrainer.__init__(model, *, variant="npo_grad_diff", beta=0.1,
#   num_epochs=5, lr=1e-5, **kwargs)
# Sources: safetune/runner/unlearn/_rmu.py, _npo.py, interventions/unlearn/rmu.py
RMU_TRAINER_KEYS = (
    "layer_id",
    "update_layer_ids",
    "max_num_batches",
    "lr",
    "alpha",
    "steering_coeff",
)


def _maybe_set_hparams(trainer, hparams: dict[str, Any]) -> None:
    """Apply RMU hyperparameters onto an already-constructed trainer.

    Prefer passing them to RMUTrainer(...) directly. This remains as a
    fallback if the trainer is built elsewhere.
    """
    for key, value in hparams.items():
        if value is None:
            continue
        if hasattr(trainer, key):
            setattr(trainer, key, value)
        elif hasattr(trainer, "args") and hasattr(trainer.args, key):
            setattr(trainer.args, key, value)
        else:
            print(
                f"TODO: RMU hyperparameter {key}={value!r} was not an attribute "
                f"on {type(trainer).__name__}. Confirmed setter names on "
                "safetune 0.1.18 are: " + ", ".join(RMU_TRAINER_KEYS)
            )


def merge_adapter_if_any(model):
    """Fold a LoRA adapter into the base weights and return the plain model (no-op for a plain model).

    SafeTune's RMUTrainer.unlearn does NOT merge (NPO/SimDPO/FLAT do). On an unmerged PeftModel it trains both the
    base_layer and the LoRA matrices of every `mlp.down_proj` it selects and returns the PeftModel, whose
    save_pretrained() writes only the adapter, silently dropping the base_layer part of the RMU update.
    """
    if hasattr(model, "merge_and_unload"):
        return model.merge_and_unload()
    return model


def _upcast_fp32(model):
    """Same effect as SafeTune's _fp32_for_training: train bf16/fp16 models in fp32 (the caller casts back afterwards)."""
    import torch

    if {p.dtype for p in model.parameters()} & {torch.float16, torch.bfloat16}:
        model.float()
    return model


def cycle_to(batches, n: int) -> list:
    """Repeat `batches` round-robin until there are `n` of them (a list of references, no copies).

    SafeTune's rmu_unlearn zips the forget and retain batch iterables and stops at the SHORTER one. With the 40-row TOFU
    forget01 split and batch_size 8 that is 5 batches, so `max_num_batches: 80` silently became 5 optimizer steps (about
    0.2% relative weight change: a no-op). The authors' RMU loop makes ~80 steps by cycling through its data.
    """
    batches = list(batches)
    if not batches:
        raise ValueError("cycle_to: no batches to cycle")
    return [batches[i % len(batches)] for i in range(int(n))]


class _RmuLossLog:
    """Capture rmu_unlearn's per-step INFO records (step, total, forget, retain losses) and print a short curve."""

    _LOGGERS = ("safetune.interventions.unlearn.rmu", "safetune.unlearn.rmu")

    def __init__(self):
        import logging

        self.rows: list[tuple[int, float, float, float]] = []
        outer = self

        class _H(logging.Handler):
            def emit(self, record):  # noqa: ANN001
                msg = str(record.msg)
                if msg.startswith("rmu_unlearn: step") and isinstance(record.args, tuple) and len(record.args) == 4:
                    outer.rows.append(tuple(float(a) for a in record.args))

        self._handler = _H(level=logging.INFO)
        self._logging = logging
        self._saved = []

    def __enter__(self):
        for name in self._LOGGERS:
            lg = self._logging.getLogger(name)
            self._saved.append((lg, lg.level))
            lg.addHandler(self._handler)
            if lg.getEffectiveLevel() > self._logging.INFO:
                lg.setLevel(self._logging.INFO)
        return self

    def __exit__(self, *exc):
        for lg, level in self._saved:
            lg.removeHandler(self._handler)
            lg.setLevel(level)
        return False

    def report(self, every: int = 10) -> dict:
        if not self.rows:
            return {"steps": 0}
        for step, total, forget, retain in self.rows:
            if int(step) == 1 or int(step) % every == 0 or int(step) == len(self.rows):
                print(f"  rmu step {int(step):3d}: loss={total:.4g} forget={forget:.4g} retain={retain:.4g}", flush=True)
        first, last = self.rows[0], self.rows[-1]
        return {
            "steps": len(self.rows),
            "forget_loss_first": first[2], "forget_loss_last": last[2],
            "retain_loss_first": first[3], "retain_loss_last": last[3],
        }


def _snapshot_update_params(model, layer_ids) -> dict:
    layers = model.model.layers
    return {
        (i, n): p.detach().float().cpu().clone()
        for i in layer_ids for n, p in layers[i].named_parameters() if "mlp.down_proj" in n
    }


def _weight_delta(before: dict, model) -> dict:
    """Max-abs / relative-Frobenius / fraction-changed of the updated down_proj weights, in the model's CURRENT dtype."""
    import torch

    layers = model.model.layers
    per_layer, num, den, changed, total = {}, 0.0, 0.0, 0, 0
    for (i, n), w0 in before.items():
        w1 = dict(layers[i].named_parameters())[n].detach().float().cpu()
        d = w1 - w0
        per_layer[f"{i}.{n}"] = {
            "max_abs": float(d.abs().max()),
            "rel_fro": float(d.norm() / (w0.norm() + 1e-12)),
            "frac_changed": float((d != 0).float().mean()),
        }
        num += float(d.norm()) ** 2
        den += float(w0.norm()) ** 2
        changed += int((d != 0).sum())
        total += d.numel()
    return {
        "rel_fro": (num / max(den, 1e-12)) ** 0.5,
        "frac_changed": changed / max(total, 1),
        "per_layer": per_layer,
    }


def run_rmu(
    model,
    *,
    forget_batches,
    retain_batches,
    hparams: dict[str, Any] | None = None,
    tokenizer=None,
):
    """Run SafeTune RMU and return the unlearned model (same dtype as the model that went in).

    FLAG: default layer_id=7 should be retuned for 36-layer Tiny Aya.
    Record the layers actually used in the write-up.

    Fixes over the first version (which was a silent no-op on the Spark, 2026-11-12):
      * batches are cycled to `max_num_batches` (see cycle_to) instead of stopping at the 5 forget batches;
      * the fp32 training copy is cast back to the incoming dtype, so the in-memory model that is evaluated is exactly
        the model that is saved and reloaded later (probes, relearn, interp all reload in cfg model.dtype);
      * the loss curve and the weight change are printed and returned as `model.rmu_report`; fewer optimizer steps than
        requested raises instead of passing quietly.
    """
    from safetune.runner import unlearn  # type: ignore

    # Constructor kwargs verified in safetune 0.1.18 RMUTrainer.__init__.
    # Default upcast=True (base class) trains in fp32: the installed
    # _fp32_for_training notes that bf16 drops most small AdamW updates on
    # Tiny Aya. The returned model may therefore be fp32.
    kw = {k: v for k, v in (hparams or {}).items() if k in RMU_TRAINER_KEYS and v is not None}
    model = merge_adapter_if_any(model)
    in_dtype = next(model.parameters()).dtype
    n_steps = int(kw.get("max_num_batches") or max(len(forget_batches), len(retain_batches)))
    kw["max_num_batches"] = n_steps
    forget_cycled = cycle_to(forget_batches, n_steps)
    retain_cycled = cycle_to(retain_batches, n_steps)
    layer_ids = list(kw.get("update_layer_ids") or [])
    before = _snapshot_update_params(model, layer_ids) if layer_ids else {}
    # hparams["forget_mask"] == "content": forget loss over the question/answer tokens only (rmu_masked.py explains why).
    masked = (hparams or {}).get("forget_mask") == "content"
    with _RmuLossLog() as losslog:
        if masked:
            if tokenizer is None:
                raise ValueError("rmu.forget_mask: content needs run_rmu(..., tokenizer=tokenizer)")
            from .rmu_masked import content_start, rmu_unlearn_masked

            model = _upcast_fp32(model)
            start = content_start(tokenizer)
            print(f"  RMU forget loss masked to tokens >= {start} (system preamble excluded)", flush=True)
            unlearned = rmu_unlearn_masked(
                model, retain_cycled, forget_cycled, layer_id=kw["layer_id"], update_layer_ids=kw["update_layer_ids"],
                alpha=kw["alpha"], steering_coeff=kw["steering_coeff"], lr=kw["lr"], forget_start=start, max_num_batches=n_steps,
            )
        else:
            trainer = unlearn.RMUTrainer(model, **kw)
            unlearned = trainer.unlearn(forget=forget_cycled, retain=retain_cycled)
    report = losslog.report()
    if unlearned is not None and next(unlearned.parameters()).dtype != in_dtype:
        unlearned = unlearned.to(in_dtype)
    if before:
        report["weight_delta"] = _weight_delta(before, unlearned)
        wd = report["weight_delta"]
        print(f"  rmu weight change after cast to {in_dtype}: rel_fro={wd['rel_fro']:.3e} frac_changed={wd['frac_changed']:.3f}", flush=True)
    report["steps_requested"] = n_steps
    report["batches"] = {"forget": len(forget_batches), "retain": len(retain_batches)}
    if report.get("steps", 0) != n_steps:
        raise RuntimeError(f"RMU ran {report.get('steps', 0)} optimizer steps, expected {n_steps}: {report}")
    try:
        unlearned.rmu_report = report
    except Exception:  # pragma: no cover
        pass
    return unlearned


def run_npo(
    model,
    *,
    forget_batches,
    retain_batches,
    variant: str = "npo_grad_diff",
):
    """Optional NPO comparison.

    Docs: https://safetune.lexsi.ai/user-guide/unlearn/npo/
    Variants (safetune 0.1.18 NPOTrainer.__init__): variant is a constructor
    keyword, default "npo_grad_diff". Also stored as trainer.variant.
    """
    from safetune.runner import unlearn  # type: ignore

    trainer = unlearn.NPOTrainer(model, variant=variant)
    return trainer.unlearn(forget=forget_batches, retain=retain_batches)


def rmu_hparams_from_config(config: dict[str, Any]) -> dict[str, Any]:
    rmu = config.get("rmu") or {}
    layer_id = int(rmu.get("layer_id", 7))
    # Trainer kwarg is update_layer_ids (safetune 0.1.18). Accept the older
    # config key update_layers as an alias.
    update_layer_ids = rmu.get("update_layer_ids") or rmu.get("update_layers")
    if not update_layer_ids:
        n_layers = int((config.get("model") or {}).get("n_layers") or 36)
        update_layer_ids = [i for i in (layer_id - 2, layer_id - 1, layer_id) if 0 <= i < n_layers]
    return {
        "layer_id": layer_id,
        "update_layer_ids": list(update_layer_ids),
        "alpha": rmu.get("alpha", 1200.0),
        "steering_coeff": rmu.get("steering_coeff", 20.0),
        "lr": rmu.get("lr", 5e-5),
        "max_num_batches": rmu.get("max_num_batches", 80),
        "forget_mask": rmu.get("forget_mask"),  # "content" = forget loss ignores the shared system preamble (rmu_masked.py)
    }
