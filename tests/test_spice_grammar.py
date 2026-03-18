import json
import re

import pytest
from lark import Lark, Tree

from grammar_parser import _build_parser, _detect_repetition_rules, _walk_tree
from grammar_utils import parse_minimal_grammar
from load_spice import GRAMMAR_PATH, SKIP_RULES

TRAIN_PATH = "data/spice/train.json"
TEST_PATH = "data/spice/test.json"

COMPONENT_RULES = {
    "resistor", "capacitor", "inductor", "voltage_source", "current_source",
    "diode", "mosfet", "bjt", "jfet", "subcircuit_call", "coupled_inductor",
    "vccs", "vcvs", "cccs", "ccvs", "behavioral_source", "switch",
}

DOT_COMMAND_RULES = {
    "model_def", "subckt_def", "tran_cmd", "dc_cmd", "ac_cmd", "op_cmd",
    "param_cmd", "lib_cmd", "include_cmd", "options_cmd", "ic_cmd",
    "control_block", "print_cmd", "plot_cmd", "measure_cmd", "global_cmd",
    "temp_cmd", "nodeset_cmd", "save_cmd", "tf_cmd", "ends_cmd", "node_cmd",
}

STATEMENT_RULES = COMPONENT_RULES | DOT_COMMAND_RULES | {"fallback_line"}


def _collect_rule_names(tree) -> set[str]:
    names = set()
    if isinstance(tree, Tree):
        names.add(tree.data)
        for child in tree.children:
            names.update(_collect_rule_names(child))
    return names


def _escape_lark_literals(alt: str) -> str:
    def _escape(m: re.Match) -> str:
        return '"' + m.group(0)[1:-1].replace("\\", "\\\\") + '"'
    return re.sub(r'"[^"]*"', _escape, alt)


def minimal_grammar_to_lark(minimal_grammar_text: str) -> str:
    min_rules = parse_minimal_grammar(minimal_grammar_text)

    # Remove empty rules and strip their references from other rules
    while True:
        empty = {name for name, alts in min_rules.items()
                 if not alts or all(not a.strip() for a in alts)}
        if not empty:
            break
        for name in list(min_rules):
            if name in empty:
                del min_rules[name]
                continue
            new_alts = []
            for alt in min_rules[name]:
                parts = re.findall(r'"[^"]*"|\S+', alt)
                filtered = [p for p in parts if p not in empty]
                if filtered:
                    new_alts.append(" ".join(filtered))
            min_rules[name] = new_alts

    top_level = [r for r in min_rules if r in STATEMENT_RULES]

    lines = ["start: (statement NEWLINE)*"]
    if top_level:
        lines.append(f'?statement: {" | ".join(top_level)}')
    for name, alts in min_rules.items():
        escaped = [_escape_lark_literals(a) for a in alts]
        lines.append(f'{name}: {" | ".join(escaped)}')
    lines.append(r"NEWLINE: /\n/")
    lines.append(r"%ignore /[ \t]+/")
    return "\n".join(lines)


def extract_body(program: str) -> str:
    lines = program.split("\n")
    body = []
    for line in lines[1:]:
        if line.strip().lower() == ".end":
            break
        body.append(line)
    return "\n".join(body) + "\n" if body else ""


@pytest.fixture(scope="module")
def all_data():
    entries = []
    for path in [TRAIN_PATH, TEST_PATH]:
        with open(path) as f:
            entries.extend(json.load(f)["data"])
    return entries


def test_parse_program_with_minimal_grammar(all_data):
    failures = []
    for i, entry in enumerate(all_data):
        body = extract_body(entry["program"])
        if not body.strip():
            continue

        try:
            min_rules = parse_minimal_grammar(entry["minimal_grammar"])
            top_level = {r for r in min_rules if r in STATEMENT_RULES}
            lark_grammar = minimal_grammar_to_lark(entry["minimal_grammar"])
            parser = Lark(lark_grammar, start="start", parser="earley")
            tree = parser.parse(body)
            used = _collect_rule_names(tree) & top_level
            missing = top_level - used
            if missing:
                failures.append(f"[{i}] rules not matched: {missing}")
        except Exception as e:
            failures.append(f"[{i}] {str(e)[:200]}")

    assert not failures, (
        f"{len(failures)} failures:\n" + "\n".join(failures[:20])
    )


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

        gen_rules = _parse_grammar_to_set(regenerated)
        exp_rules = _parse_grammar_to_set(entry["minimal_grammar"])

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


def _parse_grammar_to_set(text: str) -> dict[str, set[str]]:
    rules: dict[str, set[str]] = {}
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        name, _, rest = line.partition(" ::= ")
        alts = rest.split(" | ")
        rules[name] = set(alts)
    return rules
