from __future__ import annotations

import fire

from smoke_test.common import (
    COMMON_GENERIC_TERMINALS,
    canonical_xml,
    conversation_pair,
    first_text,
    iter_hf_records,
    iter_string_values,
    repo_path,
    write_smoke_splits,
)

DATASET = "TIGER-Lab/VisCode-Multi-679K"
DOMAIN = "restricted_graphics"
GRAMMAR = repo_path("smoke_test", DOMAIN, "restricted_graphics.lark")
GENERIC_TERMINALS = COMMON_GENERIC_TERMINALS | {"ATTR_VALUE"}


def _is_svg(example: dict) -> bool:
    language = first_text(example, ["language", "lang", "programming_language", "task"])
    return language is None or language.lower() in {"svg", "restricted_graphics"}


def _extract_svg(example: dict) -> tuple[str, str] | None:
    if not _is_svg(example):
        return None
    conv_query, conv_program = conversation_pair(example)
    query = first_text(
        example,
        ["instruction", "prompt", "question", "description", "query", "nl", "input"],
    ) or conv_query
    program = first_text(
        example,
        ["code", "program", "output", "answer", "response", "target"],
    ) or conv_program
    if not program:
        for value in iter_string_values(example):
            if "<svg" in value.lower():
                program = value
                break
    if not query or not program:
        return None
    program = canonical_xml(program)
    if "<svg" not in program.lower() or "</svg>" not in program.lower():
        return None
    return query, program


def load(
    output_dir: str = "data/smoke_test/restricted_graphics",
    max_scan: int = 250_000,
) -> None:
    dataset = iter_hf_records(DATASET, split_names=("train",))
    write_smoke_splits(
        domain=DOMAIN,
        dataset=dataset,
        extract=_extract_svg,
        output_dir=output_dir,
        grammar_path=GRAMMAR,
        start="document",
        generic_terminals=GENERIC_TERMINALS,
        max_scan=max_scan,
        min_program_chars=20,
        max_program_chars=8_000,
    )


if __name__ == "__main__":
    fire.Fire(load)
