import fire
import torch
from peft import PeftConfig, PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import format_prompt_messages, load_test_data
from predict_utils import write_output


def generate_grammar(
    adapter: str,
    test_path: str = "data/smcalflow/test.json",
    output_path: str = "outputs/predicted_grammars/generative.json",
    model_name: str | None = None,
    batch_size: int = 8,
    max_new_tokens: int = 512,
    attn_implementation: str = "flash_attention_2",
    task: str = "grammar",
):
    peft_config = PeftConfig.from_pretrained(adapter)
    base_model_name = model_name or peft_config.base_model_name_or_path
    assert base_model_name is not None, (
        "No model_name provided and adapter config has no base_model_name_or_path"
    )

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

    examples = load_test_data(test_path)

    prompts = []
    for ex in examples:
        messages = format_prompt_messages(ex, task=task)
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(text)

    results = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating grammars"):
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
            entry = {**ex, "minimal_grammar": pred, "prompt_text": prompt}
            results.append(entry)

    write_output(results, output_path)


if __name__ == "__main__":
    fire.Fire(generate_grammar)
