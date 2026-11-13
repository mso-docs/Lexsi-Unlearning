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


def run_rmu(
    model,
    *,
    forget_batches,
    retain_batches,
    hparams: dict[str, Any] | None = None,
):
    """Run SafeTune RMU and return the unlearned model.

    FLAG: default layer_id=7 should be retuned for 36-layer Tiny Aya.
    Record the layers actually used in the write-up.
    """
    from safetune.runner import unlearn  # type: ignore

    # Constructor kwargs verified in safetune 0.1.18 RMUTrainer.__init__.
    # Default upcast=True (base class) trains in fp32: the installed
    # _fp32_for_training notes that bf16 drops most small AdamW updates on
    # Tiny Aya. The returned model may therefore be fp32.
    kw = {k: v for k, v in (hparams or {}).items() if k in RMU_TRAINER_KEYS and v is not None}
    model = merge_adapter_if_any(model)
    trainer = unlearn.RMUTrainer(model, **kw)
    unlearned = trainer.unlearn(forget=forget_batches, retain=retain_batches)
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
    }
