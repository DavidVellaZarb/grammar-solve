import json

from datasets import Dataset

SYSTEM_PROMPT = (
    "You are a semantic parser. Given a user query and a grammar, produce the "
    "program that parses the query according to the grammar. Output only the "
    "program, nothing else."
)


def format_prompt_messages(example: dict) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Query: {example['query']}\n\n"
                f"Grammar:\n{example['minimal_grammar']}"
            ),
        },
    ]


def load_raw_data(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)["data"]


def load_data(path: str) -> Dataset:
    raw = load_raw_data(path)

    records = []
    for ex in raw:
        records.append(
            {
                "prompt": format_prompt_messages(ex),
                "completion": [
                    {"role": "assistant", "content": ex["program"]},
                ],
            }
        )
    return Dataset.from_list(records)
