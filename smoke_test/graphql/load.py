from __future__ import annotations

import re

import fire

from smoke_test.common import (
    COMMON_GENERIC_TERMINALS,
    clean_code_block,
    collapse_ws,
    conversation_pair,
    first_text,
    iter_hf_records,
    iter_string_values,
    repo_path,
    write_smoke_splits,
)

DATASET = "weaviate/WeaviateGraphQLGorilla"
DOMAIN = "graphql"
GRAMMAR = repo_path("smoke_test", DOMAIN, "graphql.lark")
GENERIC_TERMINALS = COMMON_GENERIC_TERMINALS | {"VARIABLE"}


def _looks_like_graphql(text: str) -> bool:
    stripped = clean_code_block(text)
    return "{" in stripped and (
        stripped.startswith("{")
        or bool(re.search(r"\b(query|mutation|subscription)\b", stripped))
    )


def _extract_graphql(example: dict) -> tuple[str, str] | None:
    conv_question, conv_program = conversation_pair(example)
    question = first_text(
        example,
        [
            "question",
            "prompt",
            "instruction",
            "nlcommand",
            "natural_language_query",
            "query_text",
            "text",
        ],
    ) or conv_question
    program = None
    if conv_program and _looks_like_graphql(conv_program):
        program = clean_code_block(conv_program)
    for field in [
        "graphql",
        "graphql_query",
        "program",
        "target",
        "output",
        "answer",
        "response",
        "query",
    ]:
        if program is not None:
            break
        value = first_text(example, [field])
        if value and _looks_like_graphql(value):
            program = clean_code_block(value)
            break
    if program is None:
        for value in iter_string_values(example):
            if _looks_like_graphql(value):
                program = clean_code_block(value)
                break
    if not question or not program:
        return None

    schema = first_text(example, ["schema", "database_schema", "context", "input"])
    query = question.strip()
    if schema and schema.strip() not in query and schema.strip() not in program:
        query = f"{query}\n\nSchema:\n{collapse_ws(schema)}"
    return query, program


def load(
    output_dir: str = "data/smoke_test/graphql",
    max_scan: int = 100_000,
) -> None:
    dataset = iter_hf_records(DATASET, split_names=("train", "validation", "test"))
    write_smoke_splits(
        domain=DOMAIN,
        dataset=dataset,
        extract=_extract_graphql,
        output_dir=output_dir,
        grammar_path=GRAMMAR,
        start="document",
        generic_terminals=GENERIC_TERMINALS,
        max_scan=max_scan,
        min_program_chars=4,
        max_program_chars=6_000,
    )


if __name__ == "__main__":
    fire.Fire(load)
