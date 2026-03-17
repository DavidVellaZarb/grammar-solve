import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import fire
from dotenv import load_dotenv
from tqdm import tqdm

from data import load_raw_data
from llm_client import LLMClient, cache_key, find_latest_metadata, load_cache, save_cache

load_dotenv()

MAX_RETRIES = 5
RETRY_BASE_DELAY = 2

SYSTEM_PROMPT = (
    "You are an expert grammar analyst. You are given a context-free grammar "
    "for a calendar-domain semantic parsing language, along with a natural "
    "language query, its gold program, and the minimal grammar needed to "
    "parse that program.\n\n"
    "Your task: explain step-by-step WHY these specific grammar rules are "
    "needed for this query/program. Reason about:\n"
    "- What the query is asking for\n"
    "- What top-level construct the program uses\n"
    "- Which rules are needed to build each part of the program\n"
    "- Why certain alternatives were chosen over others in the full grammar\n\n"
    "Output ONLY the reasoning. Do NOT output the grammar itself.\n\n"
    "Here is the full reference grammar:\n\n{full_grammar}"
)


def _build_messages(
    example: dict, system_prompt: str
) -> list[dict]:
    user_content = (
        f"Query: {example['query']}\n\n"
        f"Gold program:\n{example['program']}\n\n"
        f"Minimal grammar:\n{example['minimal_grammar']}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


async def _process_examples(
    examples: list[dict],
    llm: LLMClient,
    system_prompt: str,
    cache: dict,
    cache_path: str,
    max_concurrent: int,
    save_every: int,
) -> list[dict]:
    semaphore = asyncio.Semaphore(max_concurrent)
    pbar = tqdm(total=len(examples), desc="Generating CoT")
    processed_count = 0

    async def process_one(ex: dict) -> dict:
        nonlocal processed_count
        messages = _build_messages(ex, system_prompt)
        reasoning = await llm.call(messages, cache, semaphore)
        grammar_cot = f"{reasoning}\n\n<grammar>\n{ex['minimal_grammar']}\n</grammar>"
        pbar.update(1)
        processed_count += 1
        if processed_count % save_every == 0:
            save_cache(cache, cache_path)
        return {**ex, "grammar_cot": grammar_cot}

    tasks = [process_one(ex) for ex in examples]
    results = await asyncio.gather(*tasks)
    pbar.close()
    return list(results)


def submit(
    input_path: str = "data/smcalflow/train_balanced.json",
    output_path: str = "data/smcalflow/train_balanced_cot.json",
    grammar_path: str = "grammars/smcalflow.lark",
    model: str = "gpt-5.4",
    cache_path: str = "cache/cot_cache.json",
    api: str = "openai",
):
    with open(grammar_path) as f:
        full_grammar = f.read()

    system_prompt = SYSTEM_PROMPT.format(full_grammar=full_grammar)
    examples = load_raw_data(input_path)
    cache = load_cache(cache_path)
    print(f"Loaded {len(examples)} examples, cache has {len(cache)} entries")

    llm = LLMClient(api=api, model=model)

    requests = []
    for i, ex in enumerate(examples):
        messages = _build_messages(ex, system_prompt)
        requests.append((f"req-{i}", messages))

    task_name = Path(input_path).stem
    meta_path = llm.submit(requests, cache, task_name)
    save_cache(cache, cache_path)

    if meta_path:
        print(f"\nBatch submitted. Use 'check' to monitor, 'collect' when done.")

    # Store output_path and other context in metadata for collect
    if meta_path:
        with open(meta_path) as f:
            metadata = json.load(f)
        metadata["input_path"] = input_path
        metadata["output_path"] = output_path
        metadata["grammar_path"] = grammar_path
        metadata["cache_path"] = cache_path
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)


def check(
    metadata_path: str | None = None,
    task_name: str | None = None,
):
    status = LLMClient.check(metadata_path=metadata_path, task_name=task_name)
    print(f"Status: {status}")
    return status


def collect(
    metadata_path: str | None = None,
    task_name: str | None = None,
):
    if metadata_path is None:
        metadata_path = find_latest_metadata(task_name)
    with open(metadata_path) as f:
        metadata = json.load(f)

    cache_path = metadata["cache_path"]
    cache = load_cache(cache_path)

    LLMClient.collect(
        metadata_path=metadata_path,
        cache=cache, cache_path=cache_path,
    )

    with open(metadata["grammar_path"]) as f:
        full_grammar = f.read()
    system_prompt = SYSTEM_PROMPT.format(full_grammar=full_grammar)
    examples = load_raw_data(metadata["input_path"])
    model = metadata["model"]

    results = []
    n_missing = 0
    for ex in examples:
        messages = _build_messages(ex, system_prompt)
        key = cache_key(messages, model)
        if key in cache:
            reasoning = cache[key]
            grammar_cot = f"{reasoning}\n\n<grammar>\n{ex['minimal_grammar']}\n</grammar>"
            results.append({**ex, "grammar_cot": grammar_cot})
        else:
            results.append(ex)
            n_missing += 1

    output_path = metadata["output_path"]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"data": results}, f, indent=2)

    n_with_cot = len(results) - n_missing
    print(f"Wrote {len(results)} examples to {output_path} ({n_with_cot} with CoT, {n_missing} missing)")


def run(
    input_path: str = "data/smcalflow/train_balanced.json",
    output_path: str = "data/smcalflow/train_balanced_cot.json",
    grammar_path: str = "grammars/smcalflow.lark",
    model: str = "gpt-5.4",
    cache_path: str = "cache/cot_cache.json",
    max_concurrent: int = 20,
    save_every: int = 500,
    mode: str = "batch",
    poll_interval: int = 60,
    api: str = "openai",
):
    llm = LLMClient(api=api, model=model)

    if mode == "async":
        with open(grammar_path) as f:
            full_grammar = f.read()
        system_prompt = SYSTEM_PROMPT.format(full_grammar=full_grammar)
        examples = load_raw_data(input_path)
        cache = load_cache(cache_path)
        print(f"Loaded {len(examples)} examples, cache has {len(cache)} entries")

        results = asyncio.run(
            _process_examples(
                examples, llm, system_prompt, cache, cache_path, max_concurrent, save_every
            )
        )

        save_cache(cache, cache_path)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({"data": results}, f, indent=2)
        print(f"Wrote {len(results)} examples with CoT to {output_path}")
        return

    metadata_path = None
    try:
        task_name = Path(input_path).stem
        metadata_path = find_latest_metadata(task_name)
        status = LLMClient.check(metadata_path=metadata_path)
        if status == "completed":
            print("Batch already completed, collecting results...")
            collect(metadata_path=metadata_path)
            return
        if status == "failed":
            metadata_path = None
        else:
            print("Resuming existing batch...")
    except FileNotFoundError:
        metadata_path = None

    if metadata_path is None:
        submit(
            input_path=input_path,
            output_path=output_path,
            grammar_path=grammar_path,
            model=model,
            cache_path=cache_path,
            api=api,
        )
        task_name = Path(input_path).stem
        metadata_path = find_latest_metadata(task_name)

    print(f"\nPolling every {poll_interval}s...")
    while True:
        status = LLMClient.check(metadata_path=metadata_path)
        if status == "completed":
            break
        if status == "failed":
            print("One or more batches failed.")
            sys.exit(1)
        time.sleep(poll_interval)

    collect(metadata_path=metadata_path)


if __name__ == "__main__":
    fire.Fire({"submit": submit, "check": check, "collect": collect, "run": run})
