from __future__ import annotations

import json
import sys
from pathlib import Path

import fire
import torch
from peft import PeftConfig, PeftModel
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from smoke_test.common import canonical_program, repo_path

SRC_ROOT = repo_path("src")
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data import format_prompt_messages, load_raw_data  # noqa: E402
from model_loading import get_tokenizer, load_base_model, load_processor  # noqa: E402


def evaluate(
    adapter: str,
    test_path: str,
    domain: str,
    model_name: str | None = None,
    batch_size: int = 16,
    max_new_tokens: int = 512,
    output_path: str | None = None,
    attn_implementation: str = "flash_attention_2",
    include_grammar: bool = True,
) -> None:
    peft_config = PeftConfig.from_pretrained(adapter)
    base_model_name = model_name or peft_config.base_model_name_or_path
    if not base_model_name:
        raise ValueError("No model_name provided and adapter config has no base model")

    model = load_base_model(base_model_name, attn_implementation=attn_implementation)
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    processing_class = load_processor(base_model_name)
    tokenizer = get_tokenizer(processing_class)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    examples = load_raw_data(test_path)
    prompts: list[str] = []
    for ex in examples:
        messages = format_prompt_messages(ex, include_grammar=include_grammar, task="program")
        chat_kwargs = {"enable_thinking": False} if "qwen3" in base_model_name.lower() else {}
        prompts.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **chat_kwargs,
            )
        )

    predictions: list[str] = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch_prompts = prompts[i : i + batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(
            model.device
        )
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, prompt_len:]
        predictions.extend(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))

    del model
    torch.cuda.empty_cache()

    results = []
    for ex, prompt, pred in zip(examples, prompts, predictions, strict=True):
        gold = ex["program"]
        gold_norm = canonical_program(domain, gold)
        pred_norm = canonical_program(domain, pred)
        results.append(
            {
                "query": ex["query"],
                "prompt": prompt,
                "gold": gold,
                "prediction": pred,
                "gold_normalized": gold_norm,
                "prediction_normalized": pred_norm,
                "match": gold_norm == pred_norm,
            }
        )

    correct = sum(1 for r in results if r["match"])
    total = len(results)
    metrics = {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "metric": "canonical_exact_match",
    }
    print(f"Accuracy: {metrics['accuracy']:.4f} ({correct}/{total})")

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump({**metrics, "results": results}, f, indent=2)
        print(f"Results saved to {out}")


if __name__ == "__main__":
    fire.Fire(evaluate)
