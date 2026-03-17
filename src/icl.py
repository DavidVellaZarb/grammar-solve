import asyncio
import random

import fire
from dotenv import load_dotenv
from tqdm import tqdm

from data import (
    SYSTEM_PROMPT_WITH_GRAMMAR,
    SYSTEM_PROMPT_WITHOUT_GRAMMAR,
    load_raw_data,
)
from eval_utils import check_match, compute_metrics, save_results
from llm_client import LLMClient, load_cache, save_cache

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


if __name__ == "__main__":
    fire.Fire({"evaluate": evaluate, "evaluate_gpt": evaluate_gpt})
