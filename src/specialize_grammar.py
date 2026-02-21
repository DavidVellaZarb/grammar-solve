import hashlib
import json
import os
import random
import re

import fire
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
from openai.types.chat import ChatCompletionMessageParam

from data import load_raw_data
from grammar_utils import (
    GENERIC_TERMINALS,
    has_terminal_reference,
    parse_minimal_grammar,
    reconstruct_minimal_grammar,
)

load_dotenv()

SYSTEM_PROMPT = (
    "You are a grammar specialization assistant. Given a natural language query "
    "and grammar rules that use generic terminals (ESCAPED_STRING for strings, "
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


def _is_quoted_string(alt: str) -> bool:
    return bool(re.match(r'^".*"$', alt.strip()))



def derive_generic_rule(rule_name: str, alts: list[str]) -> str | None:
    if rule_name == "string":
        if all(_is_quoted_string(a) for a in alts):
            return "ESCAPED_STRING"
        return None

    if rule_name == "number":
        generic_parts = []
        has_l = False
        has_no_l = False
        for alt in alts:
            stripped = alt.strip().strip('"')
            if re.match(r'^-?\d+(\.\d+)?L$', stripped):
                has_l = True
            elif re.match(r'^-?\d+(\.\d+)?$', stripped):
                has_no_l = True
            else:
                # Non-numeric alternative (e.g., "(longToNum (Acouple))") — skip
                return None
        if has_l:
            generic_parts.append('NUMBER "L"')
        if has_no_l:
            generic_parts.append("NUMBER")
        if generic_parts:
            return " | ".join(generic_parts)
        return None

    return None


def replace_generic_rules(
    grammar_text: str, predicted_rules: dict[str, list[str]]
) -> str:
    parsed = parse_minimal_grammar(grammar_text)
    for name, alts in predicted_rules.items():
        if name in parsed:
            parsed[name] = alts
    return reconstruct_minimal_grammar(parsed)


def build_icl_examples(
    train_data: list[dict], n_icl_examples: int = 64, seed: int = 42
) -> list[dict]:
    candidates = []
    for ex in train_data:
        grammar = ex["minimal_grammar"]
        parsed = parse_minimal_grammar(grammar)

        generic_rules = {}
        for name, alts in parsed.items():
            derived = derive_generic_rule(name, alts)
            if derived is not None:
                generic_rules[name] = derived

        if not generic_rules:
            continue

        gold_rules = {name: parsed[name] for name in generic_rules}

        candidates.append(
            {
                "query": ex["query"],
                "generic_rules": {
                    name: [a.strip() for a in generic_rhs.split(" | ")]
                    for name, generic_rhs in generic_rules.items()
                },
                "gold_rules": gold_rules,
            }
        )

    rng = random.Random(seed)
    if len(candidates) > n_icl_examples:
        candidates = rng.sample(candidates, n_icl_examples)

    return candidates


def _format_user_message(query: str, generic_rules: dict) -> str:
    rules_str = "\n".join(
        f'{name} ::= {" | ".join(alts)}' for name, alts in generic_rules.items()
    )
    return f"Query: {query}\n\nRules to specialize:\n{rules_str}"


def _format_assistant_message(gold_rules: dict[str, list[str]]) -> str:
    return reconstruct_minimal_grammar(gold_rules)


def _build_messages(
    query: str,
    generic_rules: dict,
    icl_examples: list[dict],
) -> list[ChatCompletionMessageParam]:
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    for ex in icl_examples:
        messages.append(
            {
                "role": "user",
                "content": _format_user_message(ex["query"], ex["generic_rules"]),
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
            "content": _format_user_message(query, generic_rules),
        }
    )
    return messages


def _cache_key(messages: list[ChatCompletionMessageParam]) -> str:
    serialized = json.dumps(messages, sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()


def _load_cache(cache_path: str) -> dict:
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict, cache_path: str) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)


def _call_llm(
    client: OpenAI, model: str, messages: list[ChatCompletionMessageParam], cache: dict, cache_path: str
) -> str:
    key = _cache_key(messages)
    if key in cache:
        return cache[key]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=1024,
    )
    result = (response.choices[0].message.content or "").strip()
    cache[key] = result
    _save_cache(cache, cache_path)
    return result


def specialize(
    test_path: str = "data/smcalflow/test_generic.json",
    train_path: str = "data/smcalflow/train.json",
    output_path: str = "outputs/predicted_grammars/specialized.json",
    model: str = "anthropic/claude-sonnet-4.6",
    n_icl_examples: int = 128,
    seed: int = 42,
    cache_path: str = "cache/specialize_cache.json",
):
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    train_data = load_raw_data(train_path)
    test_data = load_raw_data(test_path)

    print(f"Building ICL examples from {len(train_data)} training examples...")
    icl_examples = build_icl_examples(train_data, n_icl_examples, seed)
    print(f"Selected {len(icl_examples)} ICL examples")

    cache = _load_cache(cache_path)
    print(f"Loaded cache with {len(cache)} entries")

    results = []
    n_specialized = 0
    n_skipped = 0

    for ex in tqdm(test_data, desc="Specializing"):
        grammar = ex["minimal_grammar"]
        generic_rules = extract_generic_rules(grammar)

        if not generic_rules:
            results.append(ex)
            n_skipped += 1
            continue

        messages = _build_messages(
            ex["query"], generic_rules, icl_examples
        )
        response = _call_llm(client, model, messages, cache, cache_path)

        predicted = parse_minimal_grammar(response)
        specialized_grammar = replace_generic_rules(grammar, predicted)

        results.append(
            {
                **ex,
                "minimal_grammar": specialized_grammar,
            }
        )
        n_specialized += 1

    output = {"data": results}
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nDone: {n_specialized} specialized, {n_skipped} skipped (no generic terminals)")
    print(f"Output saved to {output_path}")


if __name__ == "__main__":
    fire.Fire(specialize)
