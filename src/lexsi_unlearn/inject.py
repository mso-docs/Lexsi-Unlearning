"""Fine-tune so Tiny Aya actually knows the designated TOFU facts (LoRA or full).

Run this on Colab / a GPU machine / the DGX Spark, not on the Mac.

This uses standard Hugging Face PEFT + Trainer APIs (not a Lexsi toolkit).
AlignTune is a different track and is not required for a valid 2b chain.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .data import answer_of, apply_chat_template, load_tofu_split, question_of


def load_base_model(
    model_id: str = "CohereLabs/tiny-aya-global",
    *,
    dtype: str = "bfloat16",
    device_map: str = "auto",
):
    """Load the gated Tiny Aya chat model.

    Requires HF license acceptance and HF_TOKEN.
    https://huggingface.co/CohereLabs/tiny-aya-global
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(dtype, torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if (Path(str(model_id)) / "adapter_config.json").exists():
        # A saved LoRA adapter dir (the injection checkpoint): load base + adapter and MERGE, so every consumer
        # (RMU, probes, interp, relearn) sees one plain model whose weights are the injected weights.
        from peft import AutoPeftModelForCausalLM  # type: ignore

        model = AutoPeftModelForCausalLM.from_pretrained(
            str(model_id), torch_dtype=torch_dtype, device_map=device_map
        ).merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def attach_lora(model, lora_cfg: dict[str, Any]):
    """Wrap the model with a LoRA adapter.

    Tiny Aya (Cohere2ForCausalLM) module names were verified 2026-11-12 from
    model.safetensors.index.json (download only, no model load):
    self_attn.{q,k,v,o}_proj and mlp.{gate,up,down}_proj. PEFT matches those
    suffixes. If attach still fails, print named_modules() and update the yaml.
    """
    from peft import LoraConfig, get_peft_model, TaskType

    task = lora_cfg.get("task_type", "CAUSAL_LM")
    config = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("lora_alpha", 32)),
        lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
        bias=lora_cfg.get("bias", "none"),
        task_type=getattr(TaskType, task) if isinstance(task, str) else task,
        target_modules=list(lora_cfg.get("target_modules") or []),
    )
    return get_peft_model(model, config)


def _example_text(tokenizer, question: str, answer: str) -> str:
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    return f"User: {question}\nAssistant: {answer}"


def build_sft_dataset(tokenizer, rows, *, max_seq_len: int = 512):
    """Tokenize chat-formatted QA; mask prompt tokens in labels."""

    def tokenize_row(row):
        question, answer = question_of(row), answer_of(row)
        full = _example_text(tokenizer, question, answer)
        prompt_only = apply_chat_template(tokenizer, question, add_generation_prompt=True)
        full_ids = tokenizer(
            full, truncation=True, max_length=max_seq_len, add_special_tokens=False
        )
        prompt_ids = tokenizer(
            prompt_only, truncation=True, max_length=max_seq_len, add_special_tokens=False
        )
        input_ids = full_ids["input_ids"]
        labels = list(input_ids)
        prompt_len = min(len(prompt_ids["input_ids"]), len(labels))
        labels[:prompt_len] = [-100] * prompt_len
        return {
            "input_ids": input_ids,
            "attention_mask": full_ids["attention_mask"],
            "labels": labels,
        }

    return rows.map(tokenize_row, remove_columns=list(rows.column_names))


def inject_facts(
    model,
    tokenizer,
    *,
    config: dict[str, Any],
    output_dir: str | Path = "artifacts/inject_lora",
):
    """Fine-tune on forget + a retain slice (LoRA or full), then save."""
    from datasets import concatenate_datasets  # type: ignore
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

    tofu = config["tofu"]
    inj = config["injection"]
    method = str(inj.get("method", "lora")).lower()
    if method == "lora":
        try:
            from peft import PeftModel
            already = isinstance(model, PeftModel)
        except Exception:
            already = False
        if not already:
            model = attach_lora(model, inj.get("lora") or {})
    elif method == "full":
        # Full-parameter SFT. Feasible on the Spark for 3.35B bf16; tight on Colab.
        if inj.get("full") and inj["full"].get("learning_rate") is not None:
            inj = {**inj, "learning_rate": inj["full"]["learning_rate"]}
    else:
        raise ValueError(
            f"injection.method must be 'lora' or 'full', got {method!r}"
        )
    forget = load_tofu_split(
        tofu["forget_split"],
        repo_id=tofu["repo_id"],
        max_examples=inj.get("max_forget_examples"),
    )
    retain = load_tofu_split(
        tofu["retain_split"],
        repo_id=tofu["repo_id"],
        max_examples=inj.get("max_retain_examples"),
    )
    # forget_repeat > 1 shows each forget fact that many times per epoch, so 40 facts are not drowned by the retain slice.
    repeat = max(1, int(inj.get("forget_repeat", 1)))
    mixed = concatenate_datasets([forget] * repeat + [retain]).shuffle(seed=42)
    tokenized = build_sft_dataset(
        tokenizer, mixed, max_seq_len=int(inj.get("max_seq_len", 512))
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(inj.get("epochs", 3)),
        per_device_train_batch_size=int(inj.get("batch_size", 2)),
        gradient_accumulation_steps=int(inj.get("grad_accum", 8)),
        learning_rate=float(inj.get("learning_rate", 2e-4)),
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        bf16=True,
        remove_unused_columns=False,
    )
    collator = DataCollatorForSeq2Seq(tokenizer, padding=True)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=collator,
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return model, output_dir
