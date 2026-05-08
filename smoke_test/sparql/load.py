from __future__ import annotations

import re

import fire

from smoke_test.common import (
    COMMON_GENERIC_TERMINALS,
    clean_code_block,
    conversation_pair,
    first_text,
    iter_hf_records,
    iter_string_values,
    repo_path,
    write_smoke_splits,
)

DATASET = "Orange/lc_quad2-sparqltotext"
DOMAIN = "sparql"
GRAMMAR = repo_path("smoke_test", DOMAIN, "sparql.lark")
GENERIC_TERMINALS = COMMON_GENERIC_TERMINALS | {"LANGTAG"}


def _looks_like_sparql(text: str) -> bool:
    text = clean_code_block(text)
    return bool(
        re.search(r"\b(select|ask|construct|describe|prefix)\b", text, flags=re.IGNORECASE)
    ) and ("{" in text or "where" in text.lower())


def _extract_sparql(example: dict) -> tuple[str, str] | None:
    conv_question, conv_program = conversation_pair(example)
    question = first_text(
        example,
        [
            "question",
            "paraphrased_question",
            "NNQT_question",
            "corrected_question",
            "intermediary_question",
            "prompt",
            "instruction",
        ],
    ) or conv_question
    sparql = None
    if conv_program and _looks_like_sparql(conv_program):
        sparql = clean_code_block(conv_program)
    for field in [
        "simplified_query",
        "sparql_wikidata",
        "sparql_dbpedia18",
        "sparql_query",
        "sparql",
        "query",
        "target",
        "output",
        "answer",
    ]:
        if sparql is not None:
            break
        value = first_text(example, [field])
        if value and _looks_like_sparql(value):
            sparql = clean_code_block(value)
            break
    if sparql is None:
        for value in iter_string_values(example):
            if _looks_like_sparql(value):
                sparql = clean_code_block(value)
                break
    if not question or not sparql:
        return None
    return question, sparql


def load(
    output_dir: str = "data/smoke_test/sparql",
    max_scan: int = 100_000,
    specialize_terminals: bool = False,
) -> None:
    dataset = iter_hf_records(DATASET, split_names=("train", "valid", "validation", "test"))
    write_smoke_splits(
        domain=DOMAIN,
        dataset=dataset,
        extract=_extract_sparql,
        output_dir=output_dir,
        grammar_path=GRAMMAR,
        start="query",
        generic_terminals=frozenset() if specialize_terminals else GENERIC_TERMINALS,
        max_scan=max_scan,
        min_program_chars=20,
        max_program_chars=6_000,
    )


if __name__ == "__main__":
    fire.Fire(load)
