import json

import fire
import torch
from peft import PeftConfig, PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import format_prompt_messages, load_raw_data
from eval_utils import check_match, compute_metrics, save_results


def evaluate(
    adapter: str,
    test_path: str = "data/smcalflow/test.json",
    model_name: str | None = None,
    batch_size: int = 8,
    max_new_tokens: int = 512,
    output_path: str | None = None,
    attn_implementation: str = "flash_attention_2",
    grammar_file: str | None = None,
    include_grammar: bool = True,
):
    peft_config = PeftConfig.from_pretrained(adapter)
    base_model_name = model_name or peft_config.base_model_name_or_path
    assert base_model_name is not None, "No model_name provided and adapter config has no base_model_name_or_path"

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=attn_implementation,
    )
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    examples = load_raw_data(test_path)

    if grammar_file:
        print(f"Using predicted grammars from {grammar_file}")
        with open(grammar_file) as f:
            grammar_data = json.load(f)["data"]
        assert len(grammar_data) == len(examples), (
            f"Grammar file has {len(grammar_data)} entries but test data has {len(examples)}"
        )
        for ex, gex in zip(examples, grammar_data):
            assert ex["query"] == gex["query"], (
                f"Query mismatch: {ex['query']!r} vs {gex['query']!r}"
            )
            ex["minimal_grammar"] = gex["minimal_grammar"]
    else:
        print("Using gold grammars from test data")

    prompts = []
    for ex in examples:
        messages = format_prompt_messages(ex, include_grammar=include_grammar)
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(text)

    results = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="Evaluating"):
        batch_prompts = prompts[i : i + batch_size]
        batch_examples = examples[i : i + batch_size]

        inputs = tokenizer(
            batch_prompts, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, prompt_len:]
        predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        for ex, prompt, pred in zip(batch_examples, batch_prompts, predictions):
            gold = ex["program"]
            results.append(
                {
                    "prompt": prompt,
                    "gold": gold,
                    "prediction": pred,
                    "match": check_match(gold, pred),
                }
            )

    metrics = compute_metrics(results)
    print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")

    if output_path:
        save_results(metrics, results, output_path)


if __name__ == "__main__":
    fire.Fire(evaluate)
