import os
import random
import shutil
from datetime import datetime
from pathlib import Path

import fire
import torch
import wandb
from dotenv import load_dotenv
from peft import LoraConfig, TaskType
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

from datasets import concatenate_datasets
from transformers import AutoConfig

from data import load_data
from model_loading import (
    get_tokenizer,
    is_vlm,
    load_base_model,
    load_processor,
    _NEEDS_THOUGHT_CHANNEL_PREFIX,
)


_NEEDS_MM_TOKEN_TYPE_IDS = {"gemma4"}


class _Gemma4DataCollator:
    def __init__(self, base_collator):
        self.base_collator = base_collator

    def __call__(self, features):
        batch = self.base_collator(features)
        batch["mm_token_type_ids"] = torch.zeros_like(batch["input_ids"])
        return batch

load_dotenv()


def train(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    train_path: str = "data/smcalflow/train.json",
    valid_path: str = "data/smcalflow/valid.json",
    output_dir: str = "outputs/qwen2.5-7b-lora",
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    lora_target_modules: str = "all-linear",
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 1e-4,
    lr_scheduler_type: str = "cosine",
    warmup_ratio: float = 0.05,
    max_seq_length: int = 1024,
    bf16: bool = True,
    gradient_checkpointing: bool = True,
    eval_strategy: str = "steps",
    eval_steps: int = 200,
    save_steps: int = 200,
    save_total_limit: int = 3,
    report_to: str = "wandb",
    logging_steps: int = 10,
    push_to_hub: bool = True,
    attn_implementation: str = "flash_attention_2",
    max_steps: int = -1,
    hub_model_id: str | None = None,
    include_grammar: bool = True,
    task: str = "program",
    mixed_duplicate: bool = False,
    mixed_ratio: float | None = None,
    save_locally: bool = True,
):
    assert not (mixed_duplicate and mixed_ratio is not None), (
        "--mixed_duplicate and --mixed_ratio are mutually exclusive"
    )
    if not save_locally:
        assert push_to_hub, "--nosave_locally requires --push_to_hub"
    if mixed_ratio is not None:
        assert 0.0 <= mixed_ratio <= 1.0, (
            f"--mixed_ratio must be in [0.0, 1.0], got {mixed_ratio}"
        )

    model_alias = model_name.split("/")[-1].lower().removesuffix("-instruct")
    dataset_name = Path(train_path).parent.name
    hf_namespace = os.getenv("HF_NAMESPACE", "")
    hub_repo = hub_model_id or (f"{hf_namespace}/{model_alias}_{dataset_name}" if hf_namespace else None)

    def _mixed_ratio_dataset(path: str):
        ds_with = load_data(path, include_grammar=True, task=task)
        ds_without = load_data(path, include_grammar=False, task=task)
        n = len(ds_with)
        n_without = round(mixed_ratio * n)
        rng = random.Random(42)
        idx = list(range(n))
        rng.shuffle(idx)
        without_idx = idx[:n_without]
        with_idx = idx[n_without:]
        return concatenate_datasets(
            [ds_with.select(with_idx), ds_without.select(without_idx)]
        ).shuffle(seed=42)

    if mixed_duplicate:
        train_with = load_data(train_path, include_grammar=True, task=task)
        train_without = load_data(train_path, include_grammar=False, task=task)
        train_ds = concatenate_datasets([train_with, train_without]).shuffle(seed=42)
        valid_with = load_data(valid_path, include_grammar=True, task=task)
        valid_without = load_data(valid_path, include_grammar=False, task=task)
        valid_ds = concatenate_datasets([valid_with, valid_without]).shuffle(seed=42)
    elif mixed_ratio is not None:
        train_ds = _mixed_ratio_dataset(train_path)
        valid_ds = _mixed_ratio_dataset(valid_path)
    else:
        train_ds = load_data(train_path, include_grammar=include_grammar, task=task)
        valid_ds = load_data(valid_path, include_grammar=include_grammar, task=task)

    if "qwen3" in model_name.lower():
        chat_template_kwargs = {"enable_thinking": False}
        train_ds = train_ds.map(lambda ex: {**ex, "chat_template_kwargs": chat_template_kwargs})
        valid_ds = valid_ds.map(lambda ex: {**ex, "chat_template_kwargs": chat_template_kwargs})

    model_type = AutoConfig.from_pretrained(model_name, trust_remote_code=True).model_type

    processing_class = load_processor(model_name)
    tokenizer = get_tokenizer(processing_class)

    if model_type in _NEEDS_THOUGHT_CHANNEL_PREFIX:
        # The Gemma-4 template calls strip_thinking() on model message content, which strips
        # any <|channel>thought...<channel|> tokens. It also adds <|channel>thought\n<channel|>
        # to the generation prompt (add_generation_prompt=True) but NOT to the full conversation
        # format. This causes a train/inference mismatch: TRL masks tokens up to the generation
        # prompt length (including the thought channel), but those tokens aren't in the full
        # sequence, so answer( gets masked and the model learns to skip it.
        #
        # Fix: patch the processor's Jinja template (NOT tokenizer.chat_template — they are
        # separate objects; TRL uses processing_class.apply_chat_template which reads
        # processing_class.chat_template) to emit <|channel>thought\n<channel|> before each
        # model message in the full conversation format, matching the generation prompt.
        # Two paths: string content (message['content']) and structured sequence content
        # (item['text']). TRL's prepare_multimodal_messages always converts to structured
        # format, so the sequence path is what actually runs during training. Patch both.
        processing_class.chat_template = processing_class.chat_template.replace(
            "{{- strip_thinking(message['content']) -}}",
            "{{- '<|channel>thought\\n<channel|>' + strip_thinking(message['content']) -}}",
        ).replace(
            "{{- strip_thinking(item['text']) -}}",
            "{{- '<|channel>thought\\n<channel|>' + strip_thinking(item['text']) -}}",
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    vlm = is_vlm(model_name)
    sft_model = (
        load_base_model(model_name, attn_implementation=attn_implementation)
        if vlm
        else model_name
    )

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_target_modules,
        task_type=TaskType.CAUSAL_LM,
    )

    if mixed_duplicate:
        run_type = "mixed_duplicate"
    elif mixed_ratio is not None:
        run_type = f"mixed_ratio_{mixed_ratio}"
    elif task == "grammar_program":
        run_type = "grammar_program"
    elif task == "grammar":
        run_type = "predict_grammar"
    elif task == "grammar_cot":
        run_type = "predict_grammar_cot"
    elif task == "program":
        run_type = "grammar_guided" if include_grammar else "baseline"
    else:
        raise ValueError(f"Unknown task: {task!r}. Expected 'program', 'grammar', 'grammar_cot', or 'grammar_program'.")
    run_name = f"{run_type}_{model_alias}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    sft_config = SFTConfig(
        output_dir=output_dir,
        hub_model_id=hub_repo,
        run_name=run_name,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        max_length=max_seq_length,
        bf16=bf16,
        gradient_checkpointing=gradient_checkpointing,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        save_strategy="steps" if save_locally else "no",
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        report_to=report_to,
        logging_steps=logging_steps,
        model_init_kwargs=None if vlm else {
            "torch_dtype": "bfloat16",
            "attn_implementation": attn_implementation,
        },
    )

    trainer = SFTTrainer(
        model=sft_model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        processing_class=processing_class,
        peft_config=lora_config,
    )
    if model_type in _NEEDS_MM_TOKEN_TYPE_IDS:
        trainer.data_collator = _Gemma4DataCollator(trainer.data_collator)

    trainer.train()
    trainer.save_model()

    if push_to_hub and hub_repo:
        trainer.push_to_hub()
        processing_class.push_to_hub(hub_repo)
        if not save_locally:
            shutil.rmtree(output_dir, ignore_errors=True)

    if report_to == "wandb":
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(train)
