import asyncio
import sys
import time

import fire
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from data import load_raw_data, load_test_data
from grammar_utils import parse_lark_grammar, reconstruct_minimal_grammar
from knn import _find_knn, _load_or_compute_embeddings
from llm_client import LLMClient, cache_key, find_latest_metadata, load_cache, save_cache
from predict_utils import write_output

load_dotenv()


def _load_grammar_as_bnf(grammar_path: str) -> str:
    with open(grammar_path) as f:
        lark_text = f.read()
    rules = parse_lark_grammar(lark_text)
    return reconstruct_minimal_grammar(rules)


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


def _build_messages(test_query: str, neighbors: list[dict], system_prompt: str) -> list[dict]:
    user_message = _build_user_message(test_query, neighbors)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]


async def _process_example(
    ex: dict,
    neighbors: list[dict],
    system_prompt: str,
    llm: LLMClient,
    cache: dict,
    semaphore: asyncio.Semaphore,
    pbar: tqdm,
) -> dict:
    messages = _build_messages(ex["query"], neighbors, system_prompt)
    response = await llm.call(messages, cache, semaphore)
    pbar.update(1)
    return {**ex, "minimal_grammar": response}


async def _predict_async(
    test_data: list[dict],
    train_data: list[dict],
    knn_indices,
    system_prompt: str,
    llm: LLMClient,
    cache: dict,
    max_concurrent: int,
) -> list[dict]:
    semaphore = asyncio.Semaphore(max_concurrent)
    pbar = tqdm(total=len(test_data), desc="RAG predict")

    tasks = []
    for i, ex in enumerate(test_data):
        neighbors = [train_data[idx] for idx in knn_indices[i]]
        tasks.append(
            _process_example(ex, neighbors, system_prompt, llm, cache, semaphore, pbar)
        )
    results = await asyncio.gather(*tasks)
    pbar.close()
    return list(results)


def _load_knn(
    test_path: str,
    train_path: str,
    embedding_model: str,
    cache_dir: str,
    k: int,
    batch_size: int,
):
    train_data = load_raw_data(train_path)
    test_data = load_test_data(test_path)

    print(f"Train: {len(train_data)}, Test: {len(test_data)}, k={k}")

    train_queries = [ex["query"] for ex in train_data]
    test_queries = [ex["query"] for ex in test_data]

    encoder = SentenceTransformer(embedding_model)
    train_embeddings = _load_or_compute_embeddings(
        train_queries, encoder, cache_dir, embedding_model, batch_size
    )
    test_embeddings = _load_or_compute_embeddings(
        test_queries, encoder, cache_dir, embedding_model, batch_size
    )
    del encoder

    knn_indices = _find_knn(test_embeddings, train_embeddings, k)
    print(f"Found {k}-NN for {len(test_queries)} test queries")

    return train_data, test_data, knn_indices


def _write_from_cache(
    test_data, train_data, knn_indices, system_prompt,
    model, cache, output_path,
):
    results = []
    n_missing = 0
    for i, ex in enumerate(test_data):
        neighbors = [train_data[idx] for idx in knn_indices[i]]
        messages = _build_messages(ex["query"], neighbors, system_prompt)
        key = cache_key(messages, model)
        if key in cache:
            results.append({**ex, "minimal_grammar": cache[key]})
        else:
            results.append(ex)
            n_missing += 1
    if n_missing:
        print(f"Warning: {n_missing} examples missing from cache")
    write_output(results, output_path)


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
    max_concurrent: int = 5,
    max_tokens: int = 2048,
    batch_size: int = 256,
    api: str = "anthropic",
    mode: str = "async",
    poll_interval: int = 60,
):
    print(f"Model: {model}, Embedding: {embedding_model}")

    llm = LLMClient(api=api, model=model, max_tokens=max_tokens)

    if mode == "batch":
        task_name = output_path.replace("/", "_").replace(".", "_")


        meta_path = None
        try:
            meta_path = find_latest_metadata(task_name)
            status = LLMClient.check(metadata_path=meta_path)
            if status == "failed":
                print("Previous batch failed, resubmitting...")
                meta_path = None
            elif status == "in_progress":
                print(f"Resuming existing batch from {meta_path}")
        except FileNotFoundError:
            pass

        if meta_path is None:
            full_grammar = _load_grammar_as_bnf(grammar_path)
            system_prompt = SYSTEM_PROMPT_TEMPLATE.format(full_grammar=full_grammar)

            train_data, test_data, knn_indices = _load_knn(
                test_path, train_path, embedding_model, cache_dir, k, batch_size
            )
            cache = load_cache(cache_path)
            print(f"Loaded cache with {len(cache)} entries")

            requests = []
            for i, ex in enumerate(test_data):
                neighbors = [train_data[idx] for idx in knn_indices[i]]
                messages = _build_messages(ex["query"], neighbors, system_prompt)
                requests.append((f"req-{i}", messages))

            meta_path = llm.submit(requests, cache, task_name)
            save_cache(cache, cache_path)

            if not meta_path:
                _write_from_cache(
                    test_data, train_data, knn_indices, system_prompt,
                    model, cache, output_path,
                )
                return

        print(f"\nPolling every {poll_interval}s...")
        while True:
            status = LLMClient.check(metadata_path=meta_path)
            if status == "completed":
                break
            if status == "failed":
                print("One or more batches failed.")
                sys.exit(1)
            time.sleep(poll_interval)

        full_grammar = _load_grammar_as_bnf(grammar_path)
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(full_grammar=full_grammar)

        train_data, test_data, knn_indices = _load_knn(
            test_path, train_path, embedding_model, cache_dir, k, batch_size
        )
        cache = load_cache(cache_path)
        LLMClient.collect(metadata_path=meta_path, cache=cache, cache_path=cache_path)
        _write_from_cache(
            test_data, train_data, knn_indices, system_prompt,
            model, cache, output_path,
        )
        return

    full_grammar = _load_grammar_as_bnf(grammar_path)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(full_grammar=full_grammar)

    train_data, test_data, knn_indices = _load_knn(
        test_path, train_path, embedding_model, cache_dir, k, batch_size
    )
    cache = load_cache(cache_path)
    print(f"Loaded cache with {len(cache)} entries")

    results = asyncio.run(
        _predict_async(
            test_data, train_data, knn_indices, system_prompt, llm, cache,
            max_concurrent,
        )
    )

    save_cache(cache, cache_path)
    write_output(results, output_path)


def check(
    metadata_path: str | None = None,
    task_name: str | None = None,
):
    status = LLMClient.check(metadata_path=metadata_path, task_name=task_name)
    print(f"Status: {status}")
    return status


def collect(
    test_path: str = "data/smcalflow/test.json",
    train_path: str = "data/smcalflow/train.json",
    grammar_path: str = "grammars/smcalflow.lark",
    output_path: str = "outputs/predicted_grammars/rag/test_k8.json",
    model: str = "claude-opus-4-6",
    embedding_model: str = "BAAI/bge-large-en-v1.5",
    k: int = 8,
    cache_path: str = "cache/rag_cache.json",
    cache_dir: str = "cache/knn",
    batch_size: int = 256,
    metadata_path: str | None = None,
    task_name: str | None = None,
):
    cache = load_cache(cache_path)
    LLMClient.collect(
        metadata_path=metadata_path, task_name=task_name,
        cache=cache, cache_path=cache_path,
    )

    full_grammar = _load_grammar_as_bnf(grammar_path)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(full_grammar=full_grammar)

    train_data, test_data, knn_indices = _load_knn(
        test_path, train_path, embedding_model, cache_dir, k, batch_size
    )

    _write_from_cache(
        test_data, train_data, knn_indices, system_prompt,
        model, cache, output_path,
    )


if __name__ == "__main__":
    fire.Fire({"predict": predict, "check": check, "collect": collect})
