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

DATASET = "NOKHAB-Lab/LLM_4_TestBench"
DOMAIN = "vhdl"
GRAMMAR = repo_path("smoke_test", DOMAIN, "vhdl.lark")
GENERIC_TERMINALS = COMMON_GENERIC_TERMINALS | {
    "BASED_NUMBER",
    "BIT_STRING",
    "RAW_TOKEN",
}


def _looks_like_vhdl(text: str) -> bool:
    return bool(
        re.search(
            r"\b(library|use|entity|architecture|signal|process|std_logic)\b",
            text,
            flags=re.IGNORECASE,
        )
    ) and ";" in text


def _extract_vhdl(example: dict) -> tuple[str, str] | None:
    conv_question, conv_program = conversation_pair(example)
    question = first_text(
        example,
        ["instruction", "prompt", "question", "description", "nl", "task"],
    ) or conv_question
    context = first_text(example, ["input", "context", "design", "module", "entity", "source"])

    program = None
    if conv_program and _looks_like_vhdl(conv_program):
        program = conv_program
    for field in [
        "output",
        "response",
        "answer",
        "target",
        "testbench",
        "vhdl",
        "code",
        "program",
    ]:
        if program is not None:
            break
        value = first_text(example, [field])
        if value and _looks_like_vhdl(value):
            program = value
            break
    if program is None:
        for value in iter_string_values(example):
            if _looks_like_vhdl(value):
                program = value
                break
    if not question or not program:
        return None

    query = question.strip()
    if context and context.strip() not in query and context.strip() not in program:
        query = f"{query}\n\nContext:\n{collapse_ws(context)}"
    return query, program


def load(
    output_dir: str = "data/smoke_test/vhdl",
    max_scan: int = 100_000,
    specialize_terminals: bool = False,
) -> None:
    dataset = iter_hf_records(DATASET, split_names=("train", "validation", "test"))
    write_smoke_splits(
        domain=DOMAIN,
        dataset=dataset,
        extract=_extract_vhdl,
        output_dir=output_dir,
        grammar_path=GRAMMAR,
        start="source_file",
        generic_terminals=frozenset() if specialize_terminals else GENERIC_TERMINALS,
        max_scan=max_scan,
        min_program_chars=20,
        max_program_chars=10_000,
    )


if __name__ == "__main__":
    fire.Fire(load)
