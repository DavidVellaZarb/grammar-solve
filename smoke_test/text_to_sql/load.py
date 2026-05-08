from __future__ import annotations

import re

import fire

from smoke_test.common import (
    COMMON_GENERIC_TERMINALS,
    collapse_ws,
    conversation_pair,
    first_text,
    iter_hf_records,
    iter_string_values,
    repo_path,
    write_smoke_splits,
)

DATASET = "SuperMax991/spider-text2sql"
DOMAIN = "text_to_sql"
GRAMMAR = repo_path("smoke_test", DOMAIN, "text_to_sql.lark")
GENERIC_TERMINALS = COMMON_GENERIC_TERMINALS | {
    "BACKTICK_IDENTIFIER",
    "BRACKET_IDENTIFIER",
    "DOUBLE_QUOTED_STRING",
    "SINGLE_QUOTED_STRING",
}


def _looks_like_sql(text: str) -> bool:
    return bool(re.search(r"\b(select|with)\b", text, flags=re.IGNORECASE))


def _extract_sql(example: dict) -> tuple[str, str] | None:
    conv_question, conv_program = conversation_pair(example)
    question = first_text(
        example,
        [
            "question",
            "utterance",
            "natural_language_query",
            "instruction",
            "prompt",
            "text",
        ],
    ) or conv_question
    sql = None
    if conv_program and _looks_like_sql(conv_program):
        sql = conv_program
    for field in ["sql", "SQL", "query", "program", "target", "output", "answer", "response"]:
        if sql is not None:
            break
        value = first_text(example, [field])
        if value and _looks_like_sql(value):
            sql = value
            break
    if sql is None:
        for value in iter_string_values(example):
            if _looks_like_sql(value):
                sql = value
                break
    if not question or not sql:
        return None

    schema = first_text(
        example,
        ["schema", "db_schema", "database_schema", "context", "input", "table_info"],
    )
    query = question.strip()
    if schema and schema.strip() not in query:
        query = f"{query}\n\nSchema:\n{collapse_ws(schema)}"
    return query, sql


def load(
    output_dir: str = "data/smoke_test/text_to_sql",
    max_scan: int = 100_000,
    specialize_terminals: bool = False,
) -> None:
    dataset = iter_hf_records(DATASET, split_names=("train", "validation", "test"))
    write_smoke_splits(
        domain=DOMAIN,
        dataset=dataset,
        extract=_extract_sql,
        output_dir=output_dir,
        grammar_path=GRAMMAR,
        start="query",
        generic_terminals=frozenset() if specialize_terminals else GENERIC_TERMINALS,
        max_scan=max_scan,
        min_program_chars=10,
        max_program_chars=4_000,
    )


if __name__ == "__main__":
    fire.Fire(load)
