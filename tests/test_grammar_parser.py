import json

import pytest

from grammar_parser import extract_minimal_grammar

DATA_PATH = "data/smcalflow/train.json"
GRAMMAR_PATH = "grammars/smcalflow.lark"


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


@pytest.fixture(scope="module")
def train_data():
    with open(DATA_PATH) as f:
        return json.load(f)["data"]


def test_extract_matches_train_data(train_data):
    failures = []
    for i, entry in enumerate(train_data):
        generated = extract_minimal_grammar(
            entry["program"], grammar_path=GRAMMAR_PATH
        )
        gen_rules = parse_grammar(generated)
        exp_rules = parse_grammar(entry["minimal_grammar"])

        if gen_rules != exp_rules:
            diff_lines = []
            for rule in sorted(set(gen_rules) | set(exp_rules)):
                gen_alts = gen_rules.get(rule, set())
                exp_alts = exp_rules.get(rule, set())
                if gen_alts != exp_alts:
                    if gen_alts - exp_alts:
                        diff_lines.append(
                            f"  {rule}: generated has extra: {gen_alts - exp_alts}"
                        )
                    if exp_alts - gen_alts:
                        diff_lines.append(
                            f"  {rule}: expected has extra:  {exp_alts - gen_alts}"
                        )
            failures.append(f"[{i}] {entry['program'][:100]}...\n" + "\n".join(diff_lines))

    assert not failures, f"{len(failures)} failures:\n" + "\n".join(failures[:10])
