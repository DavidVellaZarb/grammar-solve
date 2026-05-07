from __future__ import annotations

import json

import fire

from smoke_test.common import (
    COMMON_GENERIC_TERMINALS,
    canonical_json,
    conversation_pair,
    first_text,
    iter_string_values,
    iter_hf_records,
    repo_path,
    write_smoke_splits,
)

DATASET = "TIGER-Lab/VisCode-Multi-679K"
DOMAIN = "vega_lite"
GRAMMAR = repo_path("smoke_test", DOMAIN, "vega_lite.lark")
GENERIC_TERMINALS = COMMON_GENERIC_TERMINALS | {"SIGNED_NUMBER"}


def _is_vega_lite(example: dict) -> bool:
    language = first_text(example, ["language", "lang", "programming_language", "task"])
    return language is None or language.lower().replace("_", "-") in {
        "vega-lite",
        "vegalite",
        "vega lite",
    }


def _extract_vega_lite(example: dict) -> tuple[str, str] | None:
    if not _is_vega_lite(example):
        return None
    conv_query, conv_program = conversation_pair(example)
    query = first_text(
        example,
        ["instruction", "prompt", "question", "description", "query", "nl", "input"],
    ) or conv_query
    program = first_text(
        example,
        ["code", "program", "output", "answer", "response", "target", "vega_lite", "spec"],
    ) or conv_program
    if not program:
        for value in iter_string_values(example):
            if '"mark"' in value or "'mark'" in value:
                program = value
                break
    if not query or not program:
        return None
    program = canonical_json(program)
    try:
        loaded = json.loads(program)
    except Exception:
        return None
    spec_keys = {"mark", "encoding", "layer", "concat", "vconcat", "hconcat", "facet", "spec", "repeat"}
    if not isinstance(loaded, dict) or not spec_keys.intersection(loaded):
        return None
    return query, program


def load(
    output_dir: str = "data/smoke_test/vega_lite",
    max_scan: int = 1_000_000,
) -> None:
    dataset = iter_hf_records(DATASET, split_names=("train",))
    write_smoke_splits(
        domain=DOMAIN,
        dataset=dataset,
        extract=_extract_vega_lite,
        output_dir=output_dir,
        grammar_path=GRAMMAR,
        start="value",
        generic_terminals=GENERIC_TERMINALS,
        max_scan=max_scan,
        min_program_chars=20,
        max_program_chars=8_000,
    )


if __name__ == "__main__":
    fire.Fire(load)
