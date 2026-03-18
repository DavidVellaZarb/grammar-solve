import json
import re

import pytest

from grammar_parser import _build_parser, _detect_repetition_rules, _walk_tree
from grammar_utils import SPICE_GENERIC_TERMINALS
from load_spice import GRAMMAR_PATH, SKIP_RULES

TRAIN_PATH = "data/spice/train.json"
TEST_PATH = "data/spice/test.json"


def parse_grammar(text: str) -> dict[str, set[str]]:
    rules: dict[str, set[str]] = {}
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        name, _, rest = line.partition(" ::= ")
        alts = rest.split(" | ")
        rules[name] = set(alts)
    return rules


def extract_literals(grammar_text: str) -> list[str]:
    return re.findall(r'"([^"]+)"', grammar_text)


@pytest.fixture(scope="module")
def all_data():
    entries = []
    for path in [TRAIN_PATH, TEST_PATH]:
        with open(path) as f:
            entries.extend(json.load(f)["data"])
    return entries


def test_grammar_literals_in_program(all_data):
    failures = []
    for i, entry in enumerate(all_data):
        program = entry["program"]
        literals = extract_literals(entry["minimal_grammar"])
        for lit in literals:
            if lit not in program:
                failures.append(f"[{i}] literal {lit!r} not in program: {program[:100]}")
                break

    assert not failures, f"{len(failures)} failures:\n" + "\n".join(failures[:20])


def test_re_extraction_matches_stored(all_data):
    parser = _build_parser(GRAMMAR_PATH, start="netlist")
    rep_rules = _detect_repetition_rules(GRAMMAR_PATH)

    failures = []
    for i, entry in enumerate(all_data):
        try:
            tree = parser.parse(entry["program"])
        except Exception:
            continue

        rules: dict[str, list[str]] = {}
        element_types: dict[str, set[str]] = {}
        _walk_tree(tree, rules, repetition_rules=rep_rules,
                   element_types=element_types, position_aware_spacing=True)

        for elem_name, types in element_types.items():
            rules.setdefault(elem_name, [])
            for t in sorted(types):
                if t not in rules[elem_name]:
                    rules[elem_name].append(t)

        rules = {k: v for k, v in rules.items() if k not in SKIP_RULES}
        lines = [f"{name} ::= {' | '.join(alts)}" for name, alts in rules.items()]
        regenerated = "\n".join(lines)

        gen_rules = parse_grammar(regenerated)
        exp_rules = parse_grammar(entry["minimal_grammar"])

        if gen_rules != exp_rules:
            diff_lines = []
            for rule in sorted(set(gen_rules) | set(exp_rules)):
                gen_alts = gen_rules.get(rule, set())
                exp_alts = exp_rules.get(rule, set())
                if gen_alts != exp_alts:
                    if gen_alts - exp_alts:
                        diff_lines.append(f"  {rule}: regen extra: {gen_alts - exp_alts}")
                    if exp_alts - gen_alts:
                        diff_lines.append(f"  {rule}: stored extra: {exp_alts - gen_alts}")
            failures.append(f"[{i}]\n" + "\n".join(diff_lines))

    assert not failures, f"{len(failures)} failures:\n" + "\n".join(failures[:10])
