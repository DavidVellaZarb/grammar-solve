import asyncio
import hashlib
import json
import os

import fire
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from data import load_raw_data, load_test_data
from knn import _find_knn, _load_or_compute_embeddings
from predict_utils import write_output

load_dotenv()

SYSTEM_PROMPT_TEMPLATE = (
    "You are a grammar prediction assistant for semantic parsing. You are given:\n"
    "1. A reference grammar defining all valid rules for a formal language\n"
    "2. Similar example queries with their minimal grammars and programs\n\n"
    "Your task: given a new query, predict the minimal grammar needed to parse it.\n\n"
    "Guidelines:\n"
    '- Output the grammar in BNF format: rule_name ::= alt1 | alt2 | ...\n'
    '- Include concrete string values (e.g., "\\\"Meeting\\\"") and numbers (e.g., 4L) '
    "inferred from the query\n"
    "- The provided examples are retrieved by similarity — not all needed rules may "
    "appear in them\n"
    "- Use the full reference grammar to identify any additional rules beyond what "
    "the examples show\n"
    "- Include only necessary rules and alternatives; do not add extras\n\n"
    "Think step-by-step about which rules are needed, then output the grammar inside "
    "<grammar>...</grammar> tags.\n\n"
    "Reference Grammar:\n{full_grammar}"
)


def _build_user_message(
    test_query: str,
    neighbors: list[dict],
) -> str:
    parts = ["Similar examples:\n"]
    for i, ex in enumerate(neighbors, 1):
        parts.append(
            f"--- Example {i} ---\n"
            f"Query: {ex['query']}\n"
            f"Grammar:\n{ex['minimal_grammar']}\n"
            f"Program:\n{ex['program']}\n"
        )
    parts.append(f"--- Your Task ---\nQuery: {test_query}")
    return "\n".join(parts)


def _cache_key(system: str, user: str, model: str) -> str:
    data = json.dumps({"system": system, "user": user, "model": model}, sort_keys=True)
    return hashlib.sha256(data.encode()).hexdigest()


def _load_cache(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(cache, f, indent=2)


async def _call_anthropic(
    client: AsyncAnthropic,
    model: str,
    system: str,
    user: str,
    cache: dict,
    semaphore: asyncio.Semaphore,
    max_tokens: int,
) -> str:
    key = _cache_key(system, user, model)
    if key in cache:
        return cache[key]
    async with semaphore:
        response = await client.messages.create(
            model=model,
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=0,
        )
    text = getattr(response.content[0], "text", None)
    assert text is not None, f"Unexpected content block type: {type(response.content[0])}"
    result = text.strip()
    cache[key] = result
    return result


async def _process_example(
    ex: dict,
    neighbors: list[dict],
    system_prompt: str,
    model: str,
    client: AsyncAnthropic,
    cache: dict,
    semaphore: asyncio.Semaphore,
    pbar: tqdm,
    max_tokens: int,
) -> dict:
    user_message = _build_user_message(ex["query"], neighbors)
    response = await _call_anthropic(
        client, model, system_prompt, user_message, cache, semaphore, max_tokens,
    )
    pbar.update(1)
    return {**ex, "minimal_grammar": response}


async def _predict_async(
    test_data: list[dict],
    train_data: list[dict],
    knn_indices,
    system_prompt: str,
    model: str,
    cache: dict,
    max_concurrent: int,
    max_tokens: int,
) -> list[dict]:
    client = AsyncAnthropic()
    semaphore = asyncio.Semaphore(max_concurrent)
    pbar = tqdm(total=len(test_data), desc="RAG predict")

    tasks = []
    for i, ex in enumerate(test_data):
        neighbors = [train_data[idx] for idx in knn_indices[i]]
        tasks.append(
            _process_example(
                ex, neighbors, system_prompt, model, client, cache, semaphore, pbar,
                max_tokens,
            )
        )
    results = await asyncio.gather(*tasks)
    pbar.close()
    return list(results)


def predict(
    test_path: str = "data/smcalflow/test.json",
    train_path: str = "data/smcalflow/train.json",
    grammar_path: str = "grammars/smcalflow.lark",
    output_path: str = "outputs/predicted_grammars/rag/test_k8.json",
    model: str = "claude-opus-4-6",
    embedding_model: str = "BAAI/bge-large-en-v1.5",
    k: int = 8,
    cache_path: str = "cache/rag_cache.json",
    cache_dir: str = "cache/knn",
    max_concurrent: int = 10,
    max_tokens: int = 2048,
    batch_size: int = 256,
):
    train_data = load_raw_data(train_path)
    test_data = load_test_data(test_path)

    with open(grammar_path) as f:
        full_grammar = f.read()

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(full_grammar=full_grammar)

    print(f"Train: {len(train_data)}, Test: {len(test_data)}, k={k}")
    print(f"Model: {model}, Embedding: {embedding_model}")

    encoder = SentenceTransformer(embedding_model)
    train_queries = [ex["query"] for ex in train_data]
    test_queries = [ex["query"] for ex in test_data]

    train_embeddings = _load_or_compute_embeddings(
        train_queries, encoder, cache_dir, embedding_model, batch_size
    )
    test_embeddings = _load_or_compute_embeddings(
        test_queries, encoder, cache_dir, embedding_model, batch_size
    )

    knn_indices = _find_knn(test_embeddings, train_embeddings, k)
    print(f"Found {k}-NN for {len(test_queries)} test queries")

    cache = _load_cache(cache_path)
    print(f"Loaded cache with {len(cache)} entries")

    results = asyncio.run(
        _predict_async(
            test_data, train_data, knn_indices, system_prompt, model, cache,
            max_concurrent, max_tokens,
        )
    )

    _save_cache(cache, cache_path)
    write_output(results, output_path)


if __name__ == "__main__":
    fire.Fire(predict)
