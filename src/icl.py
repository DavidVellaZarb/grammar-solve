import asyncio
import json
import os
import random
import sys
import time

import fire
from dotenv import load_dotenv
from tqdm import tqdm

from data import (
    SYSTEM_PROMPT_WITH_GRAMMAR,
    SYSTEM_PROMPT_WITHOUT_GRAMMAR,
    load_raw_data,
)
from eval_utils import check_match, compute_metrics, save_results
from llm_client import (
    LLMClient,
    cache_key,
    find_latest_metadata,
    load_cache,
    save_cache,
)

load_dotenv()


def _format_user_message(example: dict, mode: str) -> str:
    parts = [f"Query: {example['query']}"]

    module_header = example.get("module_header")
    if module_header:
        parts.append(f"Module header:\n{module_header}")

    if mode == "oracle":
        parts.append(f"Grammar:\n{example['minimal_grammar']}")

    return "\n\n".join(parts)


def _build_messages(
    example: dict,
    demos: list[dict],
    mode: str,
) -> list[dict]:
    system_prompt = (
        SYSTEM_PROMPT_WITH_GRAMMAR if mode == "oracle" else SYSTEM_PROMPT_WITHOUT_GRAMMAR
    )
    messages: list[dict] = [
        {"role": "system", "content": system_prompt}
    ]

    for demo in demos:
        messages.append({"role": "user", "content": _format_user_message(demo, mode)})
        messages.append({"role": "assistant", "content": demo["program"]})

    messages.append({"role": "user", "content": _format_user_message(example, mode)})
    return messages


def _select_demos_first_k(train_data: list[dict], k: int) -> list[dict]:
    return train_data[:k]


def _select_demos_knn(
    train_data: list[dict],
    test_data: list[dict],
    k: int,
    embedding_model: str = "BAAI/bge-large-en-v1.5",
    cache_dir: str = "cache/knn",
) -> list[list[dict]]:
    from knn import _find_knn, _load_or_compute_embeddings
    from sentence_transformers import SentenceTransformer

    train_queries = [ex["query"] for ex in train_data]
    test_queries = [ex["query"] for ex in test_data]

    model = SentenceTransformer(embedding_model)
    train_embeddings = _load_or_compute_embeddings(
        train_queries, model, cache_dir, embedding_model
    )
    test_embeddings = _load_or_compute_embeddings(
        test_queries, model, cache_dir, embedding_model
    )

    knn_indices = _find_knn(test_embeddings, train_embeddings, k)

    return [[train_data[idx] for idx in knn_indices[i]] for i in range(len(test_data))]


async def _evaluate_async(
    test_data: list[dict],
    demos: list[dict],
    mode: str,
    llm: LLMClient,
    cache: dict,
    max_concurrent: int,
    demos_per_example: list[list[dict]] | None = None,
) -> list[dict]:
    semaphore = asyncio.Semaphore(max_concurrent)
    pbar = tqdm(total=len(test_data), desc=f"ICL ({mode})")

    async def process(i: int, ex: dict) -> dict:
        example_demos = demos_per_example[i] if demos_per_example else demos
        messages = _build_messages(ex, example_demos, mode)
        prediction = await llm.call(messages, cache, semaphore)
        gold = ex["program"]
        pbar.update(1)
        return {
            "prompt": messages,
            "query": ex["query"],
            "gold": gold,
            "prediction": prediction,
            "match": check_match(gold, prediction),
        }

    tasks = [process(i, ex) for i, ex in enumerate(test_data)]
    results = await asyncio.gather(*tasks)
    pbar.close()
    return list(results)


def evaluate(
    test_path: str = "data/smcalflow/test.json",
    train_path: str = "data/smcalflow/train.json",
    k: int = 16,
    mode: str = "standard",
    model: str = "anthropic/claude-opus-4.6",
    seed: int = 42,
    output_path: str | None = None,
    cache_path: str | None = None,
    max_concurrent: int = 10,
    max_tokens: int = 2048,
    api: str = "openrouter",
):
    assert mode in ("standard", "oracle"), f"Invalid mode: {mode}"

    if output_path is None:
        model_alias = model.split("/")[-1]
        output_path = f"results/icl/{model_alias}_{mode}_k{k}.json"

    if cache_path is None:
        cache_path = "cache/icl_cache.json"

    train_data = load_raw_data(train_path)
    test_data = load_raw_data(test_path)

    rng = random.Random(seed)
    demos = rng.sample(train_data, min(k, len(train_data)))

    print(f"Mode: {mode} | Model: {model} | k={k} | Test: {len(test_data)} examples")

    cache = load_cache(cache_path)
    print(f"Loaded cache with {len(cache)} entries")

    llm = LLMClient(api=api, model=model, max_tokens=max_tokens)

    results = asyncio.run(
        _evaluate_async(
            test_data, demos, mode, llm, cache, max_concurrent,
        )
    )

    save_cache(cache, cache_path)

    metrics = compute_metrics(results)
    print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")

    save_results(metrics, results, output_path)


def evaluate_gpt(
    test_path: str = "data/smcalflow/test.json",
    train_path: str = "data/smcalflow/train.json",
    k: int = 64,
    mode: str = "standard",
    model: str = "gpt-5.4",
    output_path: str | None = None,
    cache_path: str | None = None,
    max_concurrent: int = 10,
    max_tokens: int = 2048,
    embedding_model: str = "BAAI/bge-large-en-v1.5",
    knn_cache_dir: str = "cache/knn",
    api: str = "openai",
):
    assert mode in ("standard", "knn", "oracle"), f"Invalid mode: {mode}"

    if output_path is None:
        model_alias = model.replace("/", "-")
        output_path = f"results/icl_{model_alias}/{mode}_k{k}.json"

    if cache_path is None:
        model_alias = model.replace("/", "-")
        cache_path = f"cache/icl_{model_alias}_cache.json"

    train_data = load_raw_data(train_path)
    test_data = load_raw_data(test_path)

    icl_mode = "oracle" if mode == "oracle" else "standard"
    demos_per_example = None

    if mode == "knn":
        print(f"Computing kNN demos (k={k}, model={embedding_model})...")
        demos_per_example = _select_demos_knn(
            train_data, test_data, k,
            embedding_model=embedding_model,
            cache_dir=knn_cache_dir,
        )
        demos = []
    else:
        demos = _select_demos_first_k(train_data, k)

    print(f"Mode: {mode} | Model: {model} | k={k} | Test: {len(test_data)} examples")

    cache = load_cache(cache_path)
    print(f"Loaded cache with {len(cache)} entries")

    llm = LLMClient(api=api, model=model, max_tokens=max_tokens)

    results = asyncio.run(
        _evaluate_async(
            test_data, demos, icl_mode, llm, cache, max_concurrent,
            demos_per_example=demos_per_example,
        )
    )

    save_cache(cache, cache_path)

    metrics = compute_metrics(results)
    print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")

    save_results(metrics, results, output_path)


SYSTEM_PROMPT_GRAMMAR_PROMPTING = (
    "You are a semantic parser. The in-context examples show, for each query, "
    "the minimal grammar wrapped in <grammar>...</grammar> tags followed by a "
    '"Program:" label and the program.\n\n'
    "Given a new query, first emit the minimal grammar in <grammar>...</grammar> "
    "tags, then emit a blank line, then 'Program:', then the program. Output "
    "nothing else."
)

SYSTEM_PROMPT_GRAMMAR_PROMPTING_COT = (
    "You are a semantic parser. The in-context examples show, for each query, "
    "step-by-step reasoning about which grammar rules are needed, the minimal "
    "grammar wrapped in <grammar>...</grammar> tags, and then a 'Program:' "
    "label followed by the program.\n\n"
    "Given a new query, follow the same pattern: produce the reasoning, then "
    "the grammar in <grammar>...</grammar> tags, then a blank line, then "
    "'Program:', then the program. Output nothing else."
)

SYSTEM_PROMPT_PREDICTED_GRAMMAR = (
    "You are a semantic parser. You are given a user query and a predicted "
    "grammar (possibly with step-by-step reasoning and wrapped in <grammar> "
    "tags). Use the predicted grammar to produce the program. Output only the "
    "program, nothing else."
)

_SUPPORTED_DATASETS = ("geoquery", "verilog", "smcalflow", "spice", "overnight")


MODE_CONFIGS: dict[str, dict] = {
    "zero_shot": {
        "demos_grammar": False,
        "demos_cot": False,
        "test_grammar": False,
        "demo_selection": "none",
        "system": SYSTEM_PROMPT_WITHOUT_GRAMMAR,
    },
    "baseline": {
        "demos_grammar": False,
        "demos_cot": False,
        "test_grammar": False,
        "demo_selection": "first_k",
        "system": SYSTEM_PROMPT_WITHOUT_GRAMMAR,
    },
    "grammar_first": {
        "demos_grammar": True,
        "demos_cot": False,
        "test_grammar": False,
        "demo_selection": "first_k",
        "system": SYSTEM_PROMPT_GRAMMAR_PROMPTING,
    },
    "grammar_knn": {
        "demos_grammar": True,
        "demos_cot": False,
        "test_grammar": False,
        "demo_selection": "knn",
        "system": SYSTEM_PROMPT_GRAMMAR_PROMPTING,
    },
    "rag_cot": {
        "demos_grammar": True,
        "demos_cot": True,
        "test_grammar": False,
        "demo_selection": "knn",
        "system": SYSTEM_PROMPT_GRAMMAR_PROMPTING_COT,
    },
    "rag_cot_with_grammar": {
        "demos_grammar": True,
        "demos_cot": True,
        "test_grammar": True,
        "demo_selection": "knn",
        "system": SYSTEM_PROMPT_PREDICTED_GRAMMAR,
    },
}


def _example_user_message(ex: dict, include_predicted_grammar: bool = False) -> str:
    parts = [f"Query: {ex['query']}"]
    module_header = ex.get("module_header")
    if module_header:
        parts.append(f"Module header:\n{module_header}")
    if include_predicted_grammar:
        parts.append(f"Predicted grammar:\n{ex['predicted_grammar']}")
    return "\n\n".join(parts)


def _demo_user_message(demo: dict, mode_cfg: dict) -> str:
    parts = [f"Query: {demo['query']}"]
    if demo.get("module_header"):
        parts.append(f"Module header:\n{demo['module_header']}")
    if mode_cfg["test_grammar"]:
        if mode_cfg["demos_cot"] and demo.get("grammar_cot"):
            parts.append(f"Predicted grammar:\n{demo['grammar_cot']}")
        else:
            parts.append(
                "Predicted grammar:\n"
                f"<grammar>\n{demo['minimal_grammar']}\n</grammar>"
            )
    return "\n\n".join(parts)


def _demo_assistant_message(demo: dict, mode_cfg: dict) -> str:
    if mode_cfg["test_grammar"]:
        return demo["program"]
    if not mode_cfg["demos_grammar"]:
        return demo["program"]
    if mode_cfg["demos_cot"]:
        grammar_block = demo["grammar_cot"]
    else:
        grammar_block = f"<grammar>\n{demo['minimal_grammar']}\n</grammar>"
    return f"{grammar_block}\n\nProgram:\n{demo['program']}"


def _build_frontier_messages(
    ex: dict,
    demos: list[dict],
    mode: str,
) -> list[dict]:
    mode_cfg = MODE_CONFIGS[mode]
    messages: list[dict] = [{"role": "system", "content": mode_cfg["system"]}]
    for demo in demos:
        messages.append({"role": "user", "content": _demo_user_message(demo, mode_cfg)})
        messages.append({"role": "assistant", "content": _demo_assistant_message(demo, mode_cfg)})
    messages.append({
        "role": "user",
        "content": _example_user_message(ex, include_predicted_grammar=mode_cfg["test_grammar"]),
    })
    return messages


def _extract_program(raw: str | None, mode: str) -> str:
    if raw is None:
        return ""
    text = raw.strip()
    mode_cfg = MODE_CONFIGS[mode]
    if mode_cfg["demos_grammar"] and not mode_cfg["test_grammar"]:
        idx = text.rfind("Program:")
        if idx != -1:
            text = text[idx + len("Program:"):]
    return text.strip()


def _load_test_examples(dataset: str, test_path: str) -> list[dict]:
    if dataset == "verilog":
        from verilog_eval.data import read_problems
        from eval_verilog import parse_verilog_eval_prompt

        problems = read_problems(test_path)
        examples: list[dict] = []
        for task_id, problem in problems.items():
            _, module_header = parse_verilog_eval_prompt(problem["prompt"])
            examples.append({
                "id": task_id,
                "query": problem["description"],
                "module_header": module_header,
                "gold_program": problem["canonical_solution"],
                "prompt": problem["prompt"],
            })
        return examples

    raw = load_raw_data(test_path)
    return [
        {"id": ex["query"], "query": ex["query"], "gold_program": ex["program"]}
        for ex in raw
    ]


def _load_predicted_grammars(path: str, dataset: str) -> dict[str, str | None]:
    with open(path) as f:
        data = json.load(f)["data"]
    key_field = "task_id" if dataset == "verilog" else "query"
    return {entry[key_field]: entry["minimal_grammar"] for entry in data}


def _select_demos(
    mode: str,
    test_examples: list[dict],
    train_data: list[dict],
    k: int,
    embedding_model: str,
    knn_cache_dir: str,
) -> list[list[dict]]:
    mode_cfg = MODE_CONFIGS[mode]
    if mode_cfg["demo_selection"] == "none":
        return [[] for _ in test_examples]
    if mode_cfg["demo_selection"] == "first_k":
        first_k = train_data[:k]
        return [first_k for _ in test_examples]

    from knn import _find_knn, _load_or_compute_embeddings
    from sentence_transformers import SentenceTransformer

    train_queries = [ex["query"] for ex in train_data]
    test_queries = [ex["query"] for ex in test_examples]
    encoder = SentenceTransformer(embedding_model)
    train_emb = _load_or_compute_embeddings(
        train_queries, encoder, knn_cache_dir, embedding_model
    )
    test_emb = _load_or_compute_embeddings(
        test_queries, encoder, knn_cache_dir, embedding_model
    )
    del encoder
    knn_idx = _find_knn(test_emb, train_emb, k)
    return [[train_data[j] for j in knn_idx[i]] for i in range(len(test_examples))]


def _prepare(
    mode: str,
    dataset: str,
    test_path: str,
    train_path: str,
    predicted_grammar_path: str | None,
    k: int,
    embedding_model: str,
    knn_cache_dir: str,
) -> tuple[list[dict], list[list[dict]]]:
    assert mode in MODE_CONFIGS, f"Unknown mode: {mode}"
    mode_cfg = MODE_CONFIGS[mode]

    test_examples = _load_test_examples(dataset, test_path)
    train_data = load_raw_data(train_path)
    if mode_cfg["demos_cot"]:
        assert all("grammar_cot" in ex for ex in train_data[:1]), (
            f"Train file {train_path} lacks grammar_cot field; use the _cot train file."
        )

    if mode_cfg["test_grammar"]:
        assert predicted_grammar_path is not None, (
            f"Mode {mode} requires --predicted_grammar_path"
        )
        pg = _load_predicted_grammars(predicted_grammar_path, dataset)
        kept: list[dict] = []
        n_missing = 0
        for ex in test_examples:
            grammar = pg.get(ex["id"])
            if not grammar:
                n_missing += 1
                continue
            ex["predicted_grammar"] = grammar
            kept.append(ex)
        if n_missing:
            print(f"Dropping {n_missing} examples with missing predicted grammar")
        test_examples = kept

    demos_per = _select_demos(
        mode, test_examples, train_data, k, embedding_model, knn_cache_dir
    )
    return test_examples, demos_per


def _task_name_for(output_path: str) -> str:
    return output_path.replace("/", "_").replace(".", "_")


def _build_requests(
    test_examples: list[dict],
    demos_per: list[list[dict]],
    mode: str,
) -> list[tuple[str, list[dict]]]:
    requests: list[tuple[str, list[dict]]] = []
    for i, ex in enumerate(test_examples):
        msgs = _build_frontier_messages(ex, demos_per[i], mode)
        requests.append((f"req-{i}", msgs))
    return requests


def _write_predictions(
    test_examples: list[dict],
    demos_per: list[list[dict]],
    mode: str,
    model: str,
    cache: dict,
    output_path: str,
) -> None:
    records = []
    n_missing = 0
    for i, ex in enumerate(test_examples):
        msgs = _build_frontier_messages(ex, demos_per[i], mode)
        key = cache_key(msgs, model)
        raw = cache.get(key)
        if raw is None:
            n_missing += 1
        records.append({
            "id": ex["id"],
            "query": ex["query"],
            "module_header": ex.get("module_header"),
            "gold_program": ex.get("gold_program"),
            "predicted_grammar": ex.get("predicted_grammar"),
            "raw_prediction": raw,
            "extracted_program": _extract_program(raw, mode),
        })
    if n_missing == len(records) and records:
        print(
            f"ERROR: all {len(records)} predictions missing from cache. "
            "Not writing output file — re-run to resubmit the batch."
        )
        return
    if n_missing:
        print(f"WARNING: {n_missing} predictions missing from cache")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"mode": mode, "model": model, "data": records}, f, indent=2)
    print(f"Wrote {len(records)} predictions to {output_path}")


def _default_cache_path(model: str) -> str:
    alias = model.replace("/", "-")
    return f"cache/icl_{alias}_cache.json"


def submit(
    mode: str,
    dataset: str,
    test_path: str,
    train_path: str,
    output_path: str,
    predicted_grammar_path: str | None = None,
    model: str = "claude-opus-4-7",
    api: str = "anthropic",
    k: int = 16,
    max_tokens: int = 4096,
    cache_path: str | None = None,
    embedding_model: str = "BAAI/bge-large-en-v1.5",
    knn_cache_dir: str = "cache/knn",
) -> str | None:
    """Submit a batch for the given mode. Non-blocking; skip if one is already running."""
    assert dataset in _SUPPORTED_DATASETS, f"Unknown dataset: {dataset}"
    if cache_path is None:
        cache_path = _default_cache_path(model)

    task_name = _task_name_for(output_path)

    try:
        existing = find_latest_metadata(task_name)
        status = LLMClient.check(metadata_path=existing)
        if status in ("in_progress", "completed"):
            print(f"[{mode}] batch already submitted (status={status}): {existing}")
            return existing
        print(f"[{mode}] previous batch failed, resubmitting")
    except FileNotFoundError:
        pass

    test_examples, demos_per = _prepare(
        mode, dataset, test_path, train_path, predicted_grammar_path,
        k, embedding_model, knn_cache_dir,
    )
    requests = _build_requests(test_examples, demos_per, mode)

    cache = load_cache(cache_path)
    print(f"[{mode}] {len(requests)} requests, cache has {len(cache)} entries")

    llm = LLMClient(api=api, model=model, max_tokens=max_tokens)
    meta_path = llm.submit(requests, cache, task_name)
    save_cache(cache, cache_path)

    if not meta_path:
        _write_predictions(test_examples, demos_per, mode, model, cache, output_path)
        return None
    return meta_path


def collect(
    mode: str,
    dataset: str,
    test_path: str,
    train_path: str,
    output_path: str,
    predicted_grammar_path: str | None = None,
    model: str = "claude-opus-4-7",
    api: str = "anthropic",
    k: int = 16,
    cache_path: str | None = None,
    embedding_model: str = "BAAI/bge-large-en-v1.5",
    knn_cache_dir: str = "cache/knn",
    poll_interval: int = 60,
) -> None:
    """Poll the batch for the given mode, collect results, and write predictions."""
    assert dataset in _SUPPORTED_DATASETS, f"Unknown dataset: {dataset}"
    if cache_path is None:
        cache_path = _default_cache_path(model)

    if os.path.exists(output_path):
        print(f"[{mode}] predictions file already exists: {output_path}")
        return

    task_name = _task_name_for(output_path)
    try:
        meta_path = find_latest_metadata(task_name)
    except FileNotFoundError:
        print(f"[{mode}] no batch metadata; trying to write predictions from cache")
        test_examples, demos_per = _prepare(
            mode, dataset, test_path, train_path, predicted_grammar_path,
            k, embedding_model, knn_cache_dir,
        )
        cache = load_cache(cache_path)
        _write_predictions(test_examples, demos_per, mode, model, cache, output_path)
        return

    print(f"[{mode}] polling {meta_path} every {poll_interval}s...")
    while True:
        status = LLMClient.check(metadata_path=meta_path)
        if status == "completed":
            break
        if status == "failed":
            print(f"[{mode}] batch failed")
            sys.exit(1)
        time.sleep(poll_interval)

    cache = load_cache(cache_path)
    LLMClient.collect(metadata_path=meta_path, cache=cache, cache_path=cache_path)

    test_examples, demos_per = _prepare(
        mode, dataset, test_path, train_path, predicted_grammar_path,
        k, embedding_model, knn_cache_dir,
    )
    _write_predictions(test_examples, demos_per, mode, model, cache, output_path)


_DEFAULT_MODE_LABELS = {
    "zero_shot": "Zero-shot",
    "baseline": "Baseline",
    "grammar_first": "GP (first-k)",
    "grammar_knn": "GP (kNN)",
    "rag_cot": "RAG CoT",
    "rag_cot_with_grammar": "RAG CoT + pred. grammar",
}

_METRIC_FIELDS: dict[str, dict[str, str]] = {
    "geoquery": {
        "accuracy": "exact_match",
        "execution_accuracy": "execution_match",
    },
    "verilog": {"pass@1": "pass@1"},
    "smcalflow": {"accuracy": "match"},
    "spice": {
        "accuracy": "exact_match",
        "ged_similarity": "ged_similarity",
        "component_f1": "component_f1",
        "syntax_validity": "valid",
    },
    "overnight": {
        "accuracy": "exact_match",
        "execution_accuracy": "execution_match",
    },
}

_METRIC_LABELS: dict[str, dict[str, str]] = {
    "geoquery": {"accuracy": "Exact Match", "execution_accuracy": "Execution Accuracy"},
    "verilog": {"pass@1": "pass@1"},
    "smcalflow": {"accuracy": "Exact Match"},
    "spice": {
        "accuracy": "Exact Match",
        "ged_similarity": "GED Similarity",
        "component_f1": "Component F1",
        "syntax_validity": "Syntax Validity",
    },
    "overnight": {"accuracy": "Exact Match", "execution_accuracy": "Execution Accuracy"},
}


def plot(
    dataset: str,
    results_dir: str,
    output_path: str,
    title: str | None = None,
    modes: tuple[str, ...] = (
        "zero_shot", "baseline", "grammar_first", "grammar_knn",
        "rag_cot", "rag_cot_with_grammar",
    ),
) -> None:
    """Plot per-mode bars for each available metric in the result JSONs."""
    if dataset not in _METRIC_FIELDS:
        raise ValueError(f"Unknown dataset: {dataset}")
    field_map = _METRIC_FIELDS[dataset]
    label_map = _METRIC_LABELS[dataset]

    result_files: list[str] = []
    labels: list[str] = []
    metric_available: dict[str, bool] = {m: True for m in field_map}

    for mode in modes:
        path = os.path.join(results_dir, f"{mode}.json")
        if not os.path.exists(path):
            print(f"skip {mode}: {path} not found")
            continue
        with open(path) as f:
            data = json.load(f)
        result_files.append(path)
        labels.append(_DEFAULT_MODE_LABELS.get(mode, mode))
        for metric, field in field_map.items():
            has_metric = metric in data
            per_example = [
                r.get(field) for r in data.get("results", [])
                if r.get(field) is not None
            ]
            if not has_metric or (dataset == "geoquery" and not per_example):
                metric_available[metric] = False

    metrics = [m for m, ok in metric_available.items() if ok]
    if not result_files or not metrics:
        print("nothing to plot")
        return

    if dataset == "verilog":
        from plot import plot_multi_metrics
        plot_multi_metrics(
            result_files=result_files,
            metrics=metrics,
            labels=labels,
            metric_labels={m: label_map[m] for m in metrics},
            output_path=output_path,
            title=title,
        )
    else:
        from plot import plot_paper_results
        plot_paper_results(
            result_files=result_files,
            labels=labels,
            metrics=metrics,
            per_example_fields={m: field_map[m] for m in metrics},
            metric_labels={m: label_map[m] for m in metrics},
            output_path=output_path,
            title=title,
        )


def eval_predictions(
    dataset: str,
    predictions_path: str,
    output_path: str,
    problem_file: str | None = None,
    k: str | int = "1",
    n_workers: int = 4,
    timeout: float = 30.0,
) -> None:
    """Compute metrics for a predictions JSON. Dispatches by dataset."""
    assert dataset in _SUPPORTED_DATASETS, f"Unknown dataset: {dataset}"
    if dataset == "geoquery":
        from eval_geoquery import evaluate_predictions as _eval
        _eval(predictions_path=predictions_path, output_path=output_path)
    elif dataset == "verilog":
        assert problem_file is not None, "verilog eval requires --problem_file"
        from eval_verilog import evaluate_predictions as _eval
        _eval(
            predictions_path=predictions_path,
            problem_file=problem_file,
            output_path=output_path,
            k=k,
            n_workers=n_workers,
            timeout=timeout,
        )
    elif dataset == "smcalflow":
        from eval import evaluate_predictions as _eval
        _eval(predictions_path=predictions_path, output_path=output_path)
    elif dataset == "spice":
        from eval_spice import evaluate_predictions as _eval
        _eval(predictions_path=predictions_path, output_path=output_path)
    elif dataset == "overnight":
        from eval_overnight import evaluate_predictions as _eval
        _eval(predictions_path=predictions_path, output_path=output_path)


if __name__ == "__main__":
    fire.Fire({
        "evaluate": evaluate,
        "evaluate_gpt": evaluate_gpt,
        "submit": submit,
        "collect": collect,
        "eval_predictions": eval_predictions,
        "plot": plot,
    })
