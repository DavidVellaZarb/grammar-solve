import json

from datasets import Dataset

SYSTEM_PROMPT_WITH_GRAMMAR = (
    "You are a semantic parser. Given a user query and a grammar, produce the "
    "program that parses the query according to the grammar. Output only the "
    "program, nothing else."
)

SYSTEM_PROMPT_WITHOUT_GRAMMAR = (
    "You are a semantic parser. Given a user query, produce the program that "
    "parses the query. Output only the program, nothing else."
)


def format_prompt_messages(example: dict, include_grammar: bool = True) -> list[dict]:
    if include_grammar:
        system_prompt = SYSTEM_PROMPT_WITH_GRAMMAR
        user_content = (
            f"Query: {example['query']}\n\n"
            f"Grammar:\n{example['minimal_grammar']}"
        )
    else:
        system_prompt = SYSTEM_PROMPT_WITHOUT_GRAMMAR
        user_content = f"Query: {example['query']}"

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def load_raw_data(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)["data"]


def load_data(path: str, include_grammar: bool = True) -> Dataset:
    raw = load_raw_data(path)

    records = []
    for ex in raw:
        records.append(
            {
                "prompt": format_prompt_messages(ex, include_grammar=include_grammar),
                "completion": [
                    {"role": "assistant", "content": ex["program"]},
                ],
            }
        )
    return Dataset.from_list(records)
