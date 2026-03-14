import asyncio
import hashlib
import json
import os

import fire
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm import tqdm

from data import load_raw_data

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


def _cache_key(messages: list[dict], model: str) -> str:
    key_data = {"messages": messages, "model": model}
    serialized = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()


def _load_cache(cache_path: str) -> dict:
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict, cache_path: str) -> None:
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)


async def _call_llm(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict],
    cache: dict,
    semaphore: asyncio.Semaphore,
    max_completion_tokens: int = 1024,
) -> str:
    key = _cache_key(messages, model)
    if key in cache:
        return cache[key]

    for attempt in range(MAX_RETRIES):
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0,
                    max_completion_tokens=max_completion_tokens,
                )
            result = (response.choices[0].message.content or "").strip()
            cache[key] = result
            return result
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            delay = RETRY_BASE_DELAY * (2 ** attempt)
            print(f"\nRetry {attempt + 1}/{MAX_RETRIES} after error: {e}")
            await asyncio.sleep(delay)


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
    model: str,
    system_prompt: str,
    cache: dict,
    cache_path: str,
    max_concurrent: int,
    save_every: int,
) -> list[dict]:
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    semaphore = asyncio.Semaphore(max_concurrent)
    pbar = tqdm(total=len(examples), desc="Generating CoT")
    processed_count = 0

    async def process_one(ex: dict) -> dict:
        nonlocal processed_count
        messages = _build_messages(ex, system_prompt)
        reasoning = await _call_llm(client, model, messages, cache, semaphore)
        grammar_cot = f"{reasoning}\n\n<grammar>\n{ex['minimal_grammar']}\n</grammar>"
        pbar.update(1)
        processed_count += 1
        if processed_count % save_every == 0:
            _save_cache(cache, cache_path)
        return {**ex, "grammar_cot": grammar_cot}

    tasks = [process_one(ex) for ex in examples]
    results = await asyncio.gather(*tasks)
    pbar.close()
    return list(results)


def generate_cot(
    input_path: str = "data/smcalflow/train_balanced.json",
    output_path: str = "data/smcalflow/train_balanced_cot.json",
    grammar_path: str = "grammars/smcalflow.lark",
    model: str = "gpt-5.4",
    cache_path: str = "cache/cot_cache.json",
    max_concurrent: int = 20,
    save_every: int = 500,
):
    with open(grammar_path) as f:
        full_grammar = f.read()

    system_prompt = SYSTEM_PROMPT.format(full_grammar=full_grammar)
    examples = load_raw_data(input_path)
    cache = _load_cache(cache_path)
    print(f"Loaded {len(examples)} examples, cache has {len(cache)} entries")

    results = asyncio.run(
        _process_examples(
            examples, model, system_prompt, cache, cache_path, max_concurrent, save_every
        )
    )

    _save_cache(cache, cache_path)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"data": results}, f, indent=2)

    print(f"Wrote {len(results)} examples with CoT to {output_path}")


if __name__ == "__main__":
    fire.Fire(generate_cot)
