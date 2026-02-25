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

SYSTEM_PROMPT_GRAMMAR = (
    "You are a semantic parser. Given a user query, produce the minimal grammar "
    "for the query. Output only the grammar, nothing else."
)

SYSTEM_PROMPT_GRAMMAR_PROGRAM = (
    "You are a semantic parser. Given a user query, first produce the minimal "
    "grammar for the query, then produce the corresponding program. Separate "
    "the grammar and program with a blank line and the label 'Program:'. "
    "Output only the grammar and program, nothing else."
)


def format_prompt_messages(
    example: dict, include_grammar: bool = True, task: str = "program"
) -> list[dict]:
    if task == "grammar":
        system_prompt = SYSTEM_PROMPT_GRAMMAR
        user_content = f"Query: {example['query']}"
    elif task == "grammar_program":
        system_prompt = SYSTEM_PROMPT_GRAMMAR_PROGRAM
        user_content = f"Query: {example['query']}"
    elif include_grammar:
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


def load_data(
    path: str, include_grammar: bool = True, task: str = "program"
) -> Dataset:
    raw = load_raw_data(path)

    records = []
    for ex in raw:
        if task == "grammar_program":
            completion = f"{ex['minimal_grammar']}\n\nProgram:\n{ex['program']}"
        elif task == "grammar":
            completion = ex["minimal_grammar"]
        else:
            completion = ex["program"]

        records.append(
            {
                "prompt": format_prompt_messages(
                    ex, include_grammar=include_grammar, task=task
                ),
                "completion": [
                    {"role": "assistant", "content": completion},
                ],
            }
        )
    return Dataset.from_list(records)
