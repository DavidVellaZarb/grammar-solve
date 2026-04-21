import json

import fire
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from peft import PeftConfig, PeftModel
from tqdm import tqdm
from data import format_prompt_messages, load_raw_data
from eval_utils import save_results
from grammar_utils import extract_grammar_from_output
from model_loading import get_tokenizer, load_base_model, load_processor


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


def _score_geoquery(
    gold: str,
    raw_prediction: str,
    pred_program: str,
    executor,
) -> dict:
    exact_match = gold in raw_prediction

    if executor is not None:
        gold_result = _try_execute(gold, executor)
        assert gold_result is not None, f"Gold program failed to execute: {gold}"
        pred_result = _try_execute(pred_program, executor)
        exec_match = gold_result == pred_result
    else:
        exec_match = None

    gold_tokens = (
        gold.replace("(", " ( ").replace(")", " ) ").replace(",", " , ").split()
    )
    pred_tokens = (
        pred_program.replace("(", " ( ").replace(")", " ) ").replace(",", " , ").split()
    )
    bleu = sentence_bleu(
        [gold_tokens], pred_tokens,
        smoothing_function=SmoothingFunction().method1,
    )

    return {
        "exact_match": exact_match,
        "execution_match": exec_match,
        "bleu": bleu,
    }


def _summarize_geoquery(results: list[dict]) -> dict:
    total = len(results)
    exact_count = sum(1 for r in results if r["exact_match"])
    bleus = [r["bleu"] for r in results]
    metrics = {
        "accuracy": exact_count / total if total else 0.0,
        "exact_match": exact_count / total if total else 0.0,
        "bleu": sum(bleus) / len(bleus) if bleus else 0.0,
        "correct": exact_count,
        "total": total,
    }
    if any(r["execution_match"] is not None for r in results):
        exec_count = sum(1 for r in results if r["execution_match"])
        metrics["execution_accuracy"] = exec_count / total if total else 0.0
        metrics["execution_correct"] = exec_count
        metrics["execution_total"] = total
    return metrics


def evaluate_predictions(
    predictions_path: str,
    output_path: str,
):
    try:
        from geo_executor import GeoExecutor
        executor = GeoExecutor()
        print("GeoQuery executor loaded (execution accuracy will be computed)")
    except Exception as e:
        executor = None
        print(f"GeoQuery executor unavailable ({type(e).__name__}); skipping execution accuracy")

    with open(predictions_path) as f:
        preds = json.load(f)["data"]

    results = []
    for entry in preds:
        gold = entry["gold_program"]
        raw = entry.get("raw_prediction") or ""
        pred_program = entry.get("extracted_program") or extract_program(raw)
        scores = _score_geoquery(gold, raw, pred_program, executor)
        results.append({
            "query": entry["query"],
            "gold": gold,
            "prediction": raw,
            "pred_program": pred_program,
            **scores,
        })

    metrics = _summarize_geoquery(results)
    print(f"Exact match:         {metrics['exact_match']:.4f} "
          f"({metrics['correct']}/{metrics['total']})")
    if "execution_accuracy" in metrics:
        print(f"Execution accuracy:  {metrics['execution_accuracy']:.4f} "
              f"({metrics['execution_correct']}/{metrics['execution_total']})")
    print(f"BLEU:                {metrics['bleu']:.4f}")

    save_results(metrics, results, output_path)


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

    model = load_base_model(base_model_name, attn_implementation=attn_implementation)
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    processing_class = load_processor(base_model_name)
    tokenizer = get_tokenizer(processing_class)
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
        chat_kwargs = {}
        if "qwen3" in base_model_name.lower():
            chat_kwargs["enable_thinking"] = False
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            **chat_kwargs,
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
        scores = _score_geoquery(gold, pred, pred_program, executor)
        results.append({
            "prompt": prompt,
            "gold": gold,
            "prediction": pred,
            "pred_program": pred_program,
            **scores,
        })

    metrics = _summarize_geoquery(results)
    print(f"Exact match:         {metrics['exact_match']:.4f} "
          f"({metrics['correct']}/{metrics['total']})")
    print(f"Execution accuracy:  {metrics['execution_accuracy']:.4f} "
          f"({metrics['execution_correct']}/{metrics['execution_total']})")
    print(f"BLEU:                {metrics['bleu']:.4f}")

    if output_path:
        save_results(metrics, results, output_path)


if __name__ == "__main__":
    fire.Fire(evaluate)
