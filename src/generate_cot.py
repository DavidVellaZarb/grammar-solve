import asyncio
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import fire
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI
from tqdm import tqdm

from data import load_raw_data

load_dotenv()

MAX_RETRIES = 5
RETRY_BASE_DELAY = 2
BATCH_METADATA_DIR = "cache/cot_batches"

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


def _build_batch_jsonl(
    examples: list[dict],
    indices: list[int],
    model: str,
    system_prompt: str,
) -> tuple[str, dict, dict]:
    lines = []
    id_to_index = {}
    id_to_cache_key = {}
    for idx in indices:
        ex = examples[idx]
        custom_id = f"req-{idx}"
        messages = _build_messages(ex, system_prompt)
        line = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": messages,
                "temperature": 0,
                "max_completion_tokens": 1024,
            },
        }
        lines.append(json.dumps(line))
        id_to_index[custom_id] = idx
        id_to_cache_key[custom_id] = _cache_key(messages, model)
    return "\n".join(lines), id_to_index, id_to_cache_key


def _save_batch_metadata(metadata: dict) -> str:
    os.makedirs(BATCH_METADATA_DIR, exist_ok=True)
    input_basename = Path(metadata["input_path"]).stem
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    filename = f"{input_basename}_{timestamp}.json"
    path = os.path.join(BATCH_METADATA_DIR, filename)
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
    return path


def _load_batch_metadata(metadata_path: str) -> dict:
    with open(metadata_path) as f:
        return json.load(f)


def _find_latest_metadata(input_path: str | None = None) -> str:
    meta_dir = Path(BATCH_METADATA_DIR)
    if not meta_dir.exists():
        raise FileNotFoundError(f"No batch metadata directory found at {BATCH_METADATA_DIR}")

    candidates = sorted(meta_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No metadata files found in {BATCH_METADATA_DIR}")

    if input_path:
        for c in candidates:
            meta = _load_batch_metadata(str(c))
            if meta.get("input_path") == input_path:
                return str(c)
        raise FileNotFoundError(
            f"No metadata file found for input_path={input_path} in {BATCH_METADATA_DIR}"
        )

    return str(candidates[0])


def submit(
    input_path: str = "data/smcalflow/train_balanced.json",
    output_path: str = "data/smcalflow/train_balanced_cot.json",
    grammar_path: str = "grammars/smcalflow.lark",
    model: str = "gpt-5.4",
    cache_path: str = "cache/cot_cache.json",
):
    with open(grammar_path) as f:
        full_grammar = f.read()

    system_prompt = SYSTEM_PROMPT.format(full_grammar=full_grammar)
    examples = load_raw_data(input_path)
    cache = _load_cache(cache_path)
    print(f"Loaded {len(examples)} examples, cache has {len(cache)} entries")

    uncached_indices = []
    for i, ex in enumerate(examples):
        messages = _build_messages(ex, system_prompt)
        key = _cache_key(messages, model)
        if key not in cache:
            uncached_indices.append(i)

    n_cached = len(examples) - len(uncached_indices)
    print(f"Cached: {n_cached}, to submit: {len(uncached_indices)}")

    if not uncached_indices:
        print("All examples are cached. Nothing to submit.")
        return

    jsonl_content, id_to_index, id_to_cache_key = _build_batch_jsonl(
        examples, uncached_indices, model, system_prompt
    )

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    jsonl_bytes = jsonl_content.encode("utf-8")
    input_file = client.files.create(
        file=("batch_input.jsonl", jsonl_bytes),
        purpose="batch",
    )
    print(f"Uploaded input file: {input_file.id}")

    batch = client.batches.create(
        input_file_id=input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"Created batch: {batch.id}")

    metadata = {
        "batch_id": batch.id,
        "input_file_id": input_file.id,
        "input_path": input_path,
        "output_path": output_path,
        "grammar_path": grammar_path,
        "model": model,
        "cache_path": cache_path,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "custom_id_to_index": id_to_index,
        "custom_id_to_cache_key": id_to_cache_key,
        "n_total": len(examples),
        "n_cached": n_cached,
        "n_submitted": len(uncached_indices),
    }
    meta_path = _save_batch_metadata(metadata)
    print(f"Metadata saved to: {meta_path}")


def check(
    metadata_path: str | None = None,
    input_path: str | None = None,
):
    if metadata_path is None:
        metadata_path = _find_latest_metadata(input_path)
    metadata = _load_batch_metadata(metadata_path)

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    batch = client.batches.retrieve(metadata["batch_id"])

    print(f"Batch: {batch.id}")
    print(f"Status: {batch.status}")
    print(f"Input: {metadata['input_path']}")
    print(f"Submitted: {metadata['n_submitted']} / Total: {metadata['n_total']}")
    if batch.request_counts:
        rc = batch.request_counts
        print(f"Completed: {rc.completed}, Failed: {rc.failed}, Total: {rc.total}")
    if batch.output_file_id:
        print(f"Output file: {batch.output_file_id}")
    if batch.error_file_id:
        print(f"Error file: {batch.error_file_id}")
    print(f"Metadata: {metadata_path}")

    return batch.status


def collect(
    metadata_path: str | None = None,
    input_path: str | None = None,
):
    if metadata_path is None:
        metadata_path = _find_latest_metadata(input_path)
    metadata = _load_batch_metadata(metadata_path)

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    batch = client.batches.retrieve(metadata["batch_id"])

    if batch.status not in ("completed", "failed", "expired"):
        print(f"Batch status is '{batch.status}', not ready for collection.")
        print("Run 'check' to see progress, or wait for completion.")
        return

    with open(metadata["grammar_path"]) as f:
        full_grammar = f.read()
    system_prompt = SYSTEM_PROMPT.format(full_grammar=full_grammar)
    examples = load_raw_data(metadata["input_path"])
    cache = _load_cache(metadata["cache_path"])
    id_to_index = metadata["custom_id_to_index"]
    id_to_cache_key = metadata["custom_id_to_cache_key"]

    n_collected = 0
    n_failed = 0
    if batch.output_file_id:
        content = client.files.content(batch.output_file_id)
        for line in content.text.strip().split("\n"):
            if not line:
                continue
            result = json.loads(line)
            custom_id = result["custom_id"]
            cache_key = id_to_cache_key.get(custom_id)
            if cache_key is None:
                continue
            response_body = result.get("response", {}).get("body", {})
            choices = response_body.get("choices", [])
            if choices:
                text = (choices[0].get("message", {}).get("content") or "").strip()
                cache[cache_key] = text
                n_collected += 1
            else:
                error = result.get("error") or result.get("response", {}).get("error")
                print(f"No choices for {custom_id}: {error}")
                n_failed += 1

    if batch.error_file_id:
        error_content = client.files.content(batch.error_file_id)
        for line in error_content.text.strip().split("\n"):
            if not line:
                continue
            error_result = json.loads(line)
            custom_id = error_result.get("custom_id", "unknown")
            error = error_result.get("error", {})
            print(f"Error for {custom_id}: {error}")
            n_failed += 1

    print(f"Collected: {n_collected}, Failed: {n_failed}")
    _save_cache(cache, metadata["cache_path"])
    print(f"Cache updated: {metadata['cache_path']}")

    results = []
    n_missing = 0
    model = metadata["model"]
    for ex in examples:
        messages = _build_messages(ex, system_prompt)
        key = _cache_key(messages, model)
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
):
    if mode == "async":
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
        return

    submit(
        input_path=input_path,
        output_path=output_path,
        grammar_path=grammar_path,
        model=model,
        cache_path=cache_path,
    )

    metadata_path = _find_latest_metadata(input_path)
    metadata = _load_batch_metadata(metadata_path)
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    print(f"\nPolling batch {metadata['batch_id']} every {poll_interval}s...")
    while True:
        batch = client.batches.retrieve(metadata["batch_id"])
        status = batch.status
        progress = ""
        if batch.request_counts:
            rc = batch.request_counts
            progress = f" ({rc.completed}/{rc.total} completed"
            if rc.failed:
                progress += f", {rc.failed} failed"
            progress += ")"
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Status: {status}{progress}")

        if status in ("completed", "failed", "expired", "cancelled"):
            break
        time.sleep(poll_interval)

    if status != "completed":
        print(f"Batch ended with status '{status}'. Running collect for partial results...")

    collect(metadata_path=metadata_path)


if __name__ == "__main__":
    fire.Fire({"submit": submit, "check": check, "collect": collect, "run": run})
