from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from smoke_test.common import COMMON_GENERIC_TERMINALS


DOMAINS = (
    "text_to_sql",
    "sparql",
    "graphql",
    "vega_lite",
    "vhdl",
    "restricted_graphics",
    "selfies",
)

DOMAIN_EXTRA_TERMINALS = {
    "sparql": frozenset({"LANGTAG"}),
    "graphql": frozenset({"VARIABLE"}),
    "vega_lite": frozenset({"SIGNED_NUMBER"}),
    "restricted_graphics": frozenset({"ATTR_VALUE"}),
    "selfies": frozenset(
        {
            "ATOM_TOKEN",
            "BOND_TOKEN",
            "BRANCH_TOKEN",
            "RING_TOKEN",
            "SELFIES_TOKEN",
            "SPECIAL_TOKEN",
        }
    ),
}

SPLITS = ("train", "valid", "test")


def generic_terminals_for_domain(domain: str) -> frozenset[str]:
    return COMMON_GENERIC_TERMINALS | DOMAIN_EXTRA_TERMINALS.get(domain, frozenset())


def find_generic_placeholders(grammar: str, terminals: frozenset[str]) -> set[str]:
    grammar = _mask_quoted_literals(grammar)
    found: set[str] = set()
    for terminal in terminals:
        pattern = rf'(?<![A-Za-z0-9_"]){re.escape(terminal)}(?![A-Za-z0-9_"])'
        if re.search(pattern, grammar):
            found.add(terminal)
    return found


def _mask_quoted_literals(grammar: str) -> str:
    chars: list[str] = []
    in_quote = False
    escaped = False
    for ch in grammar:
        if in_quote:
            if escaped:
                escaped = False
                chars.append(" ")
            elif ch == "\\":
                escaped = True
                chars.append(" ")
            elif ch == '"':
                in_quote = False
                chars.append(ch)
            else:
                chars.append(" ")
            continue

        chars.append(ch)
        if ch == '"':
            in_quote = True
    return "".join(chars)


def validate_domain(data_root: Path, domain: str) -> list[str]:
    terminals = generic_terminals_for_domain(domain)
    errors: list[str] = []
    for split in SPLITS:
        path = data_root / domain / f"{split}.json"
        if not path.exists():
            errors.append(f"{path}: missing")
            continue

        with path.open() as f:
            rows = json.load(f)["data"]

        for i, row in enumerate(rows):
            found = find_generic_placeholders(row.get("minimal_grammar", ""), terminals)
            if found:
                errors.append(
                    f"{path}: example {i} contains generic placeholders: "
                    + ", ".join(sorted(found))
                )
                break
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate that smoke-test v2 grammars specialize generic terminals."
    )
    parser.add_argument("--data-root", default="data/smoke_test_v2")
    parser.add_argument("domains", nargs="*", default=list(DOMAINS))
    args = parser.parse_args()

    data_root = Path(args.data_root)
    errors: list[str] = []
    for domain in args.domains:
        errors.extend(validate_domain(data_root, domain))

    if errors:
        raise SystemExit(
            "Smoke-test v2 specialization validation failed:\n" + "\n".join(errors)
        )

    print(
        "Smoke-test v2 specialization validation passed for "
        + ", ".join(args.domains)
    )


if __name__ == "__main__":
    main()
