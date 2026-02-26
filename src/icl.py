import asyncio
import random

import fire
from dotenv import load_dotenv
from openai.types.chat import ChatCompletionMessageParam
from tqdm import tqdm

from data import (
    SYSTEM_PROMPT_WITH_GRAMMAR,
    SYSTEM_PROMPT_WITHOUT_GRAMMAR,
    load_raw_data,
)
from eval_utils import check_match, compute_metrics, save_results
from openrouter import call_llm, load_cache, make_client, save_cache

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
) -> list[ChatCompletionMessageParam]:
    system_prompt = (
        SYSTEM_PROMPT_WITH_GRAMMAR if mode == "oracle" else SYSTEM_PROMPT_WITHOUT_GRAMMAR
    )
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt}
    ]

    for demo in demos:
        messages.append({"role": "user", "content": _format_user_message(demo, mode)})
        messages.append({"role": "assistant", "content": demo["program"]})

    messages.append({"role": "user", "content": _format_user_message(example, mode)})
    return messages


async def _evaluate_async(
    test_data: list[dict],
    demos: list[dict],
    mode: str,
    model: str,
    cache: dict,
    max_concurrent: int,
    max_tokens: int,
) -> list[dict]:
    client = make_client()
    semaphore = asyncio.Semaphore(max_concurrent)
    pbar = tqdm(total=len(test_data), desc=f"ICL ({mode})")

    async def process(ex: dict) -> dict:
        messages = _build_messages(ex, demos, mode)
        prediction = await call_llm(
            client, model, messages, cache, semaphore, max_tokens
        )
        gold = ex["program"]
        pbar.update(1)
        return {
            "prompt": messages,
            "query": ex["query"],
            "gold": gold,
            "prediction": prediction,
            "match": check_match(gold, prediction),
        }

    tasks = [process(ex) for ex in test_data]
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

    results = asyncio.run(
        _evaluate_async(test_data, demos, mode, model, cache, max_concurrent, max_tokens)
    )

    save_cache(cache, cache_path)

    metrics = compute_metrics(results)
    print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")

    save_results(metrics, results, output_path)


if __name__ == "__main__":
    fire.Fire(evaluate)
