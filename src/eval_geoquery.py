import json

import fire
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from peft import PeftConfig, PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import format_prompt_messages, load_raw_data
from eval_utils import save_results
from grammar_utils import extract_grammar_from_output


def extract_program(prediction: str) -> str:
    for line in prediction.strip().split("\n"):
        line = line.strip()
        if line:
            return line
    return prediction.strip()


def _try_execute(program: str, executor) -> str | None:
    try:
        return executor.execute(program)
    except Exception:
        return None


def evaluate(
    adapter: str,
    test_path: str = "data/geoquery/test.json",
    model_name: str | None = None,
    batch_size: int = 32,
    max_new_tokens: int = 512,
    output_path: str | None = None,
    attn_implementation: str = "flash_attention_2",
    grammar_file: str | None = None,
    include_grammar: bool = True,
    task: str = "program",
):
    from geo_executor import GeoExecutor
    executor = GeoExecutor()
    print("GeoQuery executor loaded (execution accuracy will be computed)")

    peft_config = PeftConfig.from_pretrained(adapter)
    base_model_name = model_name or peft_config.base_model_name_or_path
    assert base_model_name is not None

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
        assert len(grammar_data) == len(examples)
        skip_indices = set()
        for i, (ex, gex) in enumerate(zip(examples, grammar_data)):
            assert ex["query"] == gex["query"]
            if gex["minimal_grammar"] is None:
                skip_indices.add(i)
            else:
                ex["minimal_grammar"] = extract_grammar_from_output(gex["minimal_grammar"])
        if skip_indices:
            print(f"WARNING: Skipping {len(skip_indices)} examples with missing grammar predictions")
            examples = [ex for i, ex in enumerate(examples) if i not in skip_indices]
    else:
        print("Using gold grammars from test data")

    prompts = []
    for ex in examples:
        messages = format_prompt_messages(ex, include_grammar=include_grammar, task=task)
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(text)

    predictions = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch_prompts = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch_prompts, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            )

        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, prompt_len:]
        predictions.extend(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))

    del model
    torch.cuda.empty_cache()

    results = []
    for ex, prompt, pred in zip(examples, prompts, predictions):
        gold = ex["program"]
        pred_program = extract_program(pred)

        exact_match = gold in pred
        exec_match = None
        if executor is not None:
            gold_result = _try_execute(gold, executor)
            pred_result = _try_execute(pred_program, executor)
            if gold_result is not None:
                exec_match = gold_result == pred_result
        gold_tokens = gold.replace("(", " ( ").replace(")", " ) ").replace(",", " , ").split()
        pred_tokens = pred_program.replace("(", " ( ").replace(")", " ) ").replace(",", " , ").split()
        bleu = sentence_bleu(
            [gold_tokens], pred_tokens,
            smoothing_function=SmoothingFunction().method1,
        )

        results.append({
            "prompt": prompt,
            "gold": gold,
            "prediction": pred,
            "pred_program": pred_program,
            "exact_match": exact_match,
            "execution_match": exec_match,
            "bleu": bleu,
        })

    total = len(results)
    exact_count = sum(1 for r in results if r["exact_match"])
    bleus = [r["bleu"] for r in results]

    metrics = {
        "accuracy": exact_count / total if total > 0 else 0.0,
        "exact_match": exact_count / total if total > 0 else 0.0,
        "bleu": sum(bleus) / len(bleus) if bleus else 0.0,
        "correct": exact_count,
        "total": total,
    }

    exec_results = [r for r in results if r["execution_match"] is not None]
    exec_count = sum(1 for r in exec_results if r["execution_match"])
    metrics["execution_accuracy"] = exec_count / len(exec_results) if exec_results else 0.0
    metrics["execution_correct"] = exec_count
    metrics["execution_total"] = len(exec_results)

    print(f"Exact match:         {metrics['exact_match']:.4f} ({exact_count}/{total})")
    print(f"Execution accuracy:  {metrics['execution_accuracy']:.4f} "
          f"({metrics['execution_correct']}/{metrics['execution_total']})")
    print(f"BLEU:                {metrics['bleu']:.4f}")

    if output_path:
        save_results(metrics, results, output_path)


if __name__ == "__main__":
    fire.Fire(evaluate)
