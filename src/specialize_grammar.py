import asyncio
import json
import os
import random

import fire
from dotenv import load_dotenv
from tqdm import tqdm

from data import load_raw_data
from grammar_utils import (
    GENERIC_TERMINALS,
    has_terminal_reference,
    parse_minimal_grammar,
    reconstruct_minimal_grammar,
)
from llm_client import LLMClient, load_cache, save_cache

load_dotenv()

SYSTEM_PROMPT = (
    "You are a grammar specialization assistant. Given a natural language query "
    "and a grammar that contains generic terminals (ESCAPED_STRING for strings, "
    "NUMBER for numbers), predict the specific values that should replace "
    "these generic terminals.\n\n"
    "Output ONLY the specialized rules in the format:\n"
    "rule_name ::= alt1 | alt2 | ...\n\n"
    "For string values, use double-quoted strings like \"Frodo\".\n"
    "For number values, include the L suffix where appropriate (e.g., 1L for times/dates) "
    "or omit it for plain numbers (e.g., 30 for durations).\n"
    "Output one rule per line. Do not output anything else."
)


def has_generic_terminals(grammar_text: str) -> bool:
    parsed = parse_minimal_grammar(grammar_text)
    return any(
        has_terminal_reference(alt, GENERIC_TERMINALS)
        for alts in parsed.values()
        for alt in alts
    )


def extract_generic_rules(grammar_text: str) -> dict[str, list[str]]:
    parsed = parse_minimal_grammar(grammar_text)
    result = {}
    for name, alts in parsed.items():
        generic_alts = [a for a in alts if has_terminal_reference(a, GENERIC_TERMINALS)]
        if generic_alts:
            result[name] = alts
    return result


def replace_generic_rules(
    grammar_text: str, predicted_rules: dict[str, list[str]]
) -> str:
    parsed = parse_minimal_grammar(grammar_text)
    for name, alts in predicted_rules.items():
        if name in parsed:
            parsed[name] = alts
    return reconstruct_minimal_grammar(parsed)


def build_icl_examples(
    train_data: list[dict],
    train_generic_data: list[dict],
    n_icl_examples: int,
    seed: int,
) -> list[dict]:
    candidates = []
    for ex, generic_ex in zip(train_data, train_generic_data):
        generic_grammar = generic_ex["minimal_grammar"]
        generic_rules = extract_generic_rules(generic_grammar)

        if not generic_rules:
            continue

        specialized_parsed = parse_minimal_grammar(ex["minimal_grammar"])
        gold_rules = {name: specialized_parsed[name] for name in generic_rules}

        candidates.append(
            {
                "query": ex["query"],
                "grammar": generic_grammar,
                "gold_rules": gold_rules,
            }
        )

    rng = random.Random(seed)
    if len(candidates) > n_icl_examples:
        candidates = rng.sample(candidates, n_icl_examples)

    return candidates


def _format_user_message(query: str, grammar: str) -> str:
    return f"Query: {query}\n\nGrammar:\n{grammar}"


def _format_assistant_message(gold_rules: dict[str, list[str]]) -> str:
    return reconstruct_minimal_grammar(gold_rules)


def _build_messages(
    query: str,
    grammar: str,
    icl_examples: list[dict],
) -> list[dict]:
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    for ex in icl_examples:
        messages.append(
            {
                "role": "user",
                "content": _format_user_message(ex["query"], ex["grammar"]),
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": _format_assistant_message(ex["gold_rules"]),
            }
        )

    messages.append(
        {
            "role": "user",
            "content": _format_user_message(query, grammar),
        }
    )
    return messages


async def _process_example(
    ex: dict,
    llm: LLMClient,
    icl_examples: list[dict],
    cache: dict,
    semaphore: asyncio.Semaphore,
    pbar: tqdm,
) -> tuple[dict, bool]:
    grammar = ex["minimal_grammar"]
    generic_rules = extract_generic_rules(grammar)

    if not generic_rules:
        pbar.update(1)
        return ex, False

    messages = _build_messages(ex["query"], grammar, icl_examples)
    response = await llm.call(messages, cache, semaphore)

    predicted = parse_minimal_grammar(response)
    specialized_grammar = replace_generic_rules(grammar, predicted)

    pbar.update(1)
    return {**ex, "minimal_grammar": specialized_grammar}, True


async def _specialize_async(
    test_data: list[dict],
    llm: LLMClient,
    icl_examples: list[dict],
    cache: dict,
    max_concurrent: int,
) -> tuple[list[dict], int, int]:
    semaphore = asyncio.Semaphore(max_concurrent)
    pbar = tqdm(total=len(test_data), desc="Specializing")

    tasks = [
        _process_example(ex, llm, icl_examples, cache, semaphore, pbar)
        for ex in test_data
    ]
    outcomes = await asyncio.gather(*tasks)
    pbar.close()

    results = []
    n_specialized = 0
    n_skipped = 0
    for result_dict, was_specialized in outcomes:
        results.append(result_dict)
        if was_specialized:
            n_specialized += 1
        else:
            n_skipped += 1

    return results, n_specialized, n_skipped


def specialize(
    test_path: str = "data/smcalflow/test_generic.json",
    train_path: str = "data/smcalflow/train.json",
    train_generic_path: str = "data/smcalflow/train_generic.json",
    output_path: str = "outputs/predicted_grammars/specialized.json",
    model: str = "anthropic/claude-opus-4.6",
    n_icl_examples: int = 128,
    seed: int = 42,
    cache_path: str = "cache/specialize_cache.json",
    max_concurrent: int = 10,
    api: str = "openrouter",
):
    train_data = load_raw_data(train_path)
    train_generic_data = load_raw_data(train_generic_path)
    test_data = load_raw_data(test_path)

    assert len(train_data) == len(train_generic_data), (
        f"Training data length mismatch: {len(train_data)} vs {len(train_generic_data)}"
    )

    print(f"Building ICL examples from {len(train_data)} training examples...")
    icl_examples = build_icl_examples(train_data, train_generic_data, n_icl_examples, seed)
    print(f"Selected {len(icl_examples)} ICL examples")

    cache = load_cache(cache_path)
    print(f"Loaded cache with {len(cache)} entries")

    llm = LLMClient(api=api, model=model)

    results, n_specialized, n_skipped = asyncio.run(
        _specialize_async(test_data, llm, icl_examples, cache, max_concurrent)
    )

    save_cache(cache, cache_path)

    output = {"data": results}
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nDone: {n_specialized} specialized, {n_skipped} skipped (no generic terminals)")
    print(f"Output saved to {output_path}")


if __name__ == "__main__":
    fire.Fire(specialize)
