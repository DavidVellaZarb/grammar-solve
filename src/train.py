import os
from datetime import datetime
from pathlib import Path

import fire
import wandb
from dotenv import load_dotenv
from peft import LoraConfig, TaskType
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer
from transformers import AutoTokenizer

from data import load_data

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
):
    model_alias = model_name.split("/")[-1].lower().removesuffix("-instruct")
    dataset_name = Path(train_path).parent.name
    hf_namespace = os.getenv("HF_NAMESPACE", "")
    hub_repo = hub_model_id or (f"{hf_namespace}/{model_alias}_{dataset_name}" if hf_namespace else None)

    train_ds = load_data(train_path, include_grammar=include_grammar, task=task)
    valid_ds = load_data(valid_path, include_grammar=include_grammar, task=task)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_target_modules,
        task_type=TaskType.CAUSAL_LM,
    )

    if task == "grammar_program":
        run_type = "grammar_program"
    elif task == "grammar":
        run_type = "predict_grammar"
    elif task == "program":
        run_type = "grammar_guided" if include_grammar else "baseline"
    else:
        raise ValueError(f"Unknown task: {task!r}. Expected 'program', 'grammar', or 'grammar_program'.")
    run_name = f"{run_type}_{model_alias}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    sft_config = SFTConfig(
        output_dir=output_dir,
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
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        report_to=report_to,
        logging_steps=logging_steps,
        push_to_hub=push_to_hub,
        hub_model_id=hub_repo,
        model_init_kwargs={
            "torch_dtype": "bfloat16",
            "attn_implementation": attn_implementation,
        },
    )

    trainer = SFTTrainer(
        model=model_name,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    trainer.train()
    trainer.save_model()

    if push_to_hub and hub_repo:
        trainer.push_to_hub()
        tokenizer.push_to_hub(hub_repo)

    if report_to == "wandb":
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(train)
