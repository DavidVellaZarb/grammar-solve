from __future__ import annotations

import fire

from smoke_test.common import (
    first_text,
    iter_hf_records,
    repo_path,
    write_smoke_splits,
)

DATASET = "liupf/ChEBI-20-MM"
DOMAIN = "selfies"
GRAMMAR = repo_path("smoke_test", DOMAIN, "selfies.lark")
GENERIC_TERMINALS = frozenset(
    {
        "ATOM_TOKEN",
        "BOND_TOKEN",
        "BRANCH_TOKEN",
        "RING_TOKEN",
        "SELFIES_TOKEN",
        "SPECIAL_TOKEN",
    }
)


def _extract_selfies(example: dict) -> tuple[str, str] | None:
    query = first_text(example, ["description", "caption", "question", "prompt", "input"])
    program = first_text(example, ["SELFIES", "selfies", "target", "program", "output"])
    if not query or not program:
        return None
    if "[" not in program or "]" not in program:
        return None
    return query, program


def load(
    output_dir: str = "data/smoke_test/selfies",
    max_scan: int = 50_000,
) -> None:
    dataset = iter_hf_records(DATASET, split_names=("train", "validation", "test"))
    write_smoke_splits(
        domain=DOMAIN,
        dataset=dataset,
        extract=_extract_selfies,
        output_dir=output_dir,
        grammar_path=GRAMMAR,
        start="selfies",
        generic_terminals=GENERIC_TERMINALS,
        max_scan=max_scan,
        min_program_chars=4,
        max_program_chars=4_000,
        normalize_repetition=False,
        position_aware_spacing=False,
    )


if __name__ == "__main__":
    fire.Fire(load)

