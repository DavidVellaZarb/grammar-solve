import json
import os
import re

import fire
import torch
from peft import PeftConfig, PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from verilog_eval.data import read_problems, write_jsonl
from verilog_eval.evaluation import evaluate_functional_correctness

from data import format_prompt_messages
from grammar_parser import extract_minimal_grammar

VERILOG_GRAMMAR_PATH = "grammars/verilog.lark"
VERILOG_SKIP_RULES = {
    "start", "module", "list_of_ports", "parameter_list", "port_item",
    "port_declaration", "port_dir",
}


def parse_verilog_eval_prompt(prompt: str) -> tuple[str, str]:
    lines = prompt.split("\n")
    comment_lines = []
    rest_lines = []
    found_module = False
    for line in lines:
        if not found_module and line.strip().startswith("//"):
            comment_lines.append(line.lstrip("/ ").strip())
        else:
            found_module = True
            rest_lines.append(line)

    description = " ".join(comment_lines).strip()
    module_header = "\n".join(rest_lines).strip()
    return description, module_header


def extract_completion(raw_prediction: str) -> str:
    pred = raw_prediction.strip()

    endmodule_match = re.search(r"\bendmodule\b", pred)
    if endmodule_match:
        pred = pred[: endmodule_match.end()]
    else:
        pred = pred.rstrip() + "\nendmodule"

    if not pred.startswith("\n"):
        pred = "\n" + pred

    return pred


def evaluate(
    adapter: str,
    problem_file: str = "data/verilog_eval/VerilogEval_Machine.jsonl",
    model_name: str | None = None,
    batch_size: int = 8,
    max_new_tokens: int = 1024,
    n_samples: int = 1,
    temperature: float = 0.0,
    top_p: float = 0.95,
    k: str | None = None,
    n_workers: int = 4,
    timeout: float = 30.0,
    output_path: str | None = None,
    attn_implementation: str = "flash_attention_2",
    include_grammar: bool = False,
    grammar_file: str | None = None,
):
    if k is not None:
        k_values = [int(x.strip()) for x in k.split(",")]
    else:
        k_values = sorted(set(v for v in [1, 3, 5, 10, 100] if v <= n_samples))

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

    problems = read_problems(problem_file)
    print(f"Loaded {len(problems)} problems from {problem_file}")

    grammar_map = {}
    if include_grammar and grammar_file:
        print(f"Using predicted grammars from {grammar_file}")
        with open(grammar_file) as f:
            grammar_data = json.load(f)["data"]
        for entry in grammar_data:
            key = entry.get("task_id") or entry.get("query")
            grammar_map[key] = entry["minimal_grammar"]
    elif include_grammar:
        print("Extracting oracle grammars from canonical solutions...")
        failures = 0
        for task_id, problem in problems.items():
            full_module = problem["prompt"] + problem["canonical_solution"]
            try:
                grammar = extract_minimal_grammar(
                    full_module,
                    grammar_path=VERILOG_GRAMMAR_PATH,
                    start="module",
                    skip_rules=VERILOG_SKIP_RULES,
                )
                grammar_map[task_id] = grammar
            except Exception:
                grammar_map[task_id] = ""
                failures += 1
        print(f"  Extracted {len(grammar_map) - failures}/{len(problems)} grammars "
              f"({failures} failures)")

    task_ids = list(problems.keys())
    formatted_prompts = []

    for task_id in task_ids:
        problem = problems[task_id]
        description, module_header = parse_verilog_eval_prompt(problem["prompt"])
        query = description if description else task_id

        example = {"query": query, "module_header": module_header}
        if include_grammar:
            grammar = grammar_map.get(task_id, "")
            if not grammar:
                print(f"Warning: no grammar for {task_id}, using empty")
            example["minimal_grammar"] = grammar

        messages = format_prompt_messages(
            example, include_grammar=include_grammar, task="program"
        )
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        formatted_prompts.append(text)

    gen_kwargs: dict[str, int | float | bool] = dict(max_new_tokens=max_new_tokens)
    if temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
    else:
        gen_kwargs.update(do_sample=False)

    all_samples = []

    for sample_idx in range(n_samples):
        if n_samples > 1:
            print(f"\n--- Sample {sample_idx + 1}/{n_samples} ---")

        for i in tqdm(
            range(0, len(formatted_prompts), batch_size),
            desc="Generating",
        ):
            batch_prompts = formatted_prompts[i : i + batch_size]
            batch_task_ids = task_ids[i : i + batch_size]

            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(model.device)

            with torch.no_grad():
                output_ids = model.generate(**inputs, **gen_kwargs)

            prompt_len = inputs["input_ids"].shape[1]
            generated_ids = output_ids[:, prompt_len:]
            predictions = tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )

            for tid, pred in zip(batch_task_ids, predictions):
                completion = extract_completion(pred)
                all_samples.append({"task_id": tid, "completion": completion})

    print(f"\nGenerated {len(all_samples)} total completions")

    if output_path is None:
        output_path = "results/verilog_eval/eval.json"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    samples_path = output_path.replace(".json", "_samples.jsonl")
    write_jsonl(samples_path, all_samples)
    print(f"Samples written to {samples_path}")

    print(f"\nRunning functional evaluation (k={k_values}, workers={n_workers})...")
    pass_at_k = evaluate_functional_correctness(
        samples_path,
        problem_file,
        k=k_values,
        n_workers=n_workers,
        timeout=timeout,
    )
    print("\nResults:")
    for metric, value in pass_at_k.items():
        print(f"  {metric}: {value:.4f}")

    results = {
        **pass_at_k,
        "config": {
            "adapter": adapter,
            "problem_file": problem_file,
            "n_samples": n_samples,
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            "include_grammar": include_grammar,
            "grammar_file": grammar_file,
        },
        "samples_path": samples_path,
    }
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    fire.Fire(evaluate)
