import json
import re
from dataclasses import dataclass
from functools import lru_cache

import fire
from lark import Lark, Token, Tree

from grammar_utils import GENERIC_TERMINALS


@dataclass
class RepetitionInfo:
    element: str
    separator: str
    is_terminal_sep: bool
    is_transparent: bool = False


@lru_cache(maxsize=4)
def _detect_repetition_rules(grammar_path: str) -> dict[str, list[RepetitionInfo]]:
    """Detect pure repetition rules: element (sep element)* or element (sep element)+"""
    with open(grammar_path) as f:
        text = f.read()

    transparent_rules: set[str] = set()
    repetition_rules: dict[str, list[RepetitionInfo]] = {}
    current_rule_clean: str | None = None

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("//") or stripped.startswith("%"):
            continue

        m = re.match(r'^(\??)([a-z_]\w*)\s*:\s*(.*)', line)
        if m:
            is_transparent = m.group(1) == '?'
            matched_name = m.group(2)
            current_rule_clean = matched_name
            rhs = m.group(3).strip()

            if is_transparent:
                transparent_rules.add(matched_name)

            if rhs:
                for alt in rhs.split('|'):
                    alt = alt.strip()
                    if alt:
                        _check_repetition_alt(alt, matched_name, repetition_rules)
            continue

        if line[0] in (' ', '\t') and current_rule_clean is not None:
            if stripped.startswith('|'):
                alt = stripped[1:].strip()
                if alt:
                    _check_repetition_alt(alt, current_rule_clean, repetition_rules)
            continue

        current_rule_clean = None

    for infos in repetition_rules.values():
        for info in infos:
            if info.element in transparent_rules:
                info.is_transparent = True

    return repetition_rules


def _check_repetition_alt(
    alt: str, rule_name: str, repetition_rules: dict[str, list[RepetitionInfo]]
) -> None:
    m = re.match(r'^(\w+)\s+\("([^"]+)"\s+\1\)\s*[*+]\s*$', alt)
    if m:
        repetition_rules.setdefault(rule_name, [])
        repetition_rules[rule_name].append(
            RepetitionInfo(element=m.group(1), separator=m.group(2), is_terminal_sep=False)
        )
        return

    m = re.match(r'^(\w+)\s+\(([A-Z_]+)\s+\1\)\s*[*+]\s*$', alt)
    if m:
        repetition_rules.setdefault(rule_name, [])
        repetition_rules[rule_name].append(
            RepetitionInfo(element=m.group(1), separator=m.group(2), is_terminal_sep=True)
        )
        return


@lru_cache(maxsize=4)
def _build_parser(grammar_path: str, start: str = "call") -> Lark:
    with open(grammar_path) as f:
        grammar_text = f.read()
    return Lark(
        grammar_text,
        start=start,
        parser="earley",
        keep_all_tokens=True,
    )


def _fix_ambiguity(tree: Tree) -> Tree:
    if not isinstance(tree, Tree):
        return tree

    if tree.data == "date" and len(tree.children) == 1:
        child = tree.children[0]
        if isinstance(child, Tree) and child.data == "day" and child.children:
            first = child.children[0]
            if isinstance(first, Token) and str(first) == "(adjustByPeriod":
                inner = list(child.children)
                if isinstance(inner[1], Tree) and inner[1].data == "day":
                    inner[1] = Tree("date", [inner[1]])
                tree.children = inner
                return _fix_ambiguity(tree)

    tree.children = [_fix_ambiguity(c) for c in tree.children]
    return tree


def _join_token_buf(tokens: list[Token]) -> str:
    if not tokens:
        return ""
    parts = [str(tokens[0])]
    for i in range(1, len(tokens)):
        prev, cur = tokens[i - 1], tokens[i]
        if (getattr(prev, "end_column", None) is not None
                and getattr(cur, "column", None) is not None):
            if getattr(prev, "end_line", 0) == getattr(cur, "line", 0):
                if prev.end_column < cur.column:
                    parts.append(" ")
            else:
                parts.append(" ")
        parts.append(str(cur))
    return "".join(parts)


def _reconstruct_alt(children: list, generic_terminals: frozenset[str] | None = None,
                     position_aware_spacing: bool = False) -> str:
    parts: list[str] = []
    token_buf: list = []

    for child in children:
        if isinstance(child, Tree):
            if token_buf:
                joined = _join_token_buf(token_buf) if position_aware_spacing else "".join(token_buf)
                parts.append('"' + joined + '"')
                token_buf = []
            parts.append(child.data)
        elif isinstance(child, Token):
            if generic_terminals and child.type in generic_terminals:
                if token_buf:
                    joined = _join_token_buf(token_buf) if position_aware_spacing else "".join(token_buf)
                    parts.append('"' + joined + '"')
                    token_buf = []
                parts.append(child.type)
            else:
                token_buf.append(child if position_aware_spacing else str(child))

    if token_buf:
        joined = _join_token_buf(token_buf) if position_aware_spacing else "".join(token_buf)
        parts.append('"' + joined + '"')

    return " ".join(parts)


def _match_repetition(tree: Tree, rep_infos: list[RepetitionInfo]) -> RepetitionInfo | None:
    if len(rep_infos) == 1:
        return rep_infos[0]

    for child in tree.children:
        if isinstance(child, Tree):
            for ri in rep_infos:
                if ri.element == child.data:
                    return ri
            for ri in rep_infos:
                if ri.is_transparent:
                    return ri
            break
        if isinstance(child, Token):
            for ri in rep_infos:
                if ri.element == child.type:
                    return ri
            break

    return rep_infos[0]


def _collect_separators(tree: Tree, rep_info: RepetitionInfo) -> set[str]:
    separators: set[str] = set()
    element = rep_info.element
    is_element_terminal = element.isupper()

    for child in tree.children:
        if not isinstance(child, Token):
            continue
        if is_element_terminal:
            if child.type != element:
                separators.add(str(child))
        elif rep_info.is_terminal_sep:
            if child.type == rep_info.separator:
                separators.add(str(child))
        else:
            if str(child) == rep_info.separator:
                separators.add(str(child))

    return separators


def _walk_tree(
    tree: Tree,
    rules: dict[str, list[str]],
    generic: bool = False,
    generic_terminals: frozenset[str] | None = None,
    repetition_rules: dict[str, list[RepetitionInfo]] | None = None,
    element_types: dict[str, set[str]] | None = None,
    position_aware_spacing: bool = False,
) -> None:
    if generic_terminals is None and generic:
        generic_terminals = GENERIC_TERMINALS

    rule_name = tree.data
    normalized = False

    if repetition_rules and rule_name in repetition_rules:
        rep_info = _match_repetition(tree, repetition_rules[rule_name])
        if rep_info:
            element = rep_info.element
            is_element_terminal = element.isupper()

            if not is_element_terminal or (generic_terminals and element in generic_terminals):
                normalized = True
                separators = _collect_separators(tree, rep_info)

                rules.setdefault(rule_name, [])
                for sep in sorted(separators):
                    recursive = f'{rule_name} "{sep}" {element}'
                    if recursive not in rules[rule_name]:
                        rules[rule_name].append(recursive)
                if element not in rules[rule_name]:
                    rules[rule_name].append(element)

                if rep_info.is_transparent and element_types is not None:
                    element_types.setdefault(element, set())
                    for child in tree.children:
                        if isinstance(child, Tree):
                            element_types[element].add(child.data)

    if not normalized:
        alt = _reconstruct_alt(tree.children, generic_terminals,
                               position_aware_spacing=position_aware_spacing)
        rules.setdefault(rule_name, [])
        if alt not in rules[rule_name]:
            rules[rule_name].append(alt)

    for child in tree.children:
        if isinstance(child, Tree):
            _walk_tree(child, rules, generic_terminals=generic_terminals,
                       repetition_rules=repetition_rules, element_types=element_types,
                       position_aware_spacing=position_aware_spacing)


def extract_minimal_grammar(
    program: str,
    grammar_path: str = "grammars/smcalflow.lark",
    generic: bool = False,
    start: str = "call",
    skip_rules: set[str] | None = None,
    generic_terminals: frozenset[str] | None = None,
    normalize_repetition: bool = True,
) -> str:
    if generic_terminals is None and generic:
        generic_terminals = GENERIC_TERMINALS

    parser = _build_parser(grammar_path, start)
    tree = parser.parse(program)
    if "smcalflow" in grammar_path:
        tree = _fix_ambiguity(tree)

    rep_rules = _detect_repetition_rules(grammar_path) if normalize_repetition else None
    element_types = {} if normalize_repetition else None

    rules: dict[str, list[str]] = {}
    _walk_tree(tree, rules, generic_terminals=generic_terminals,
               repetition_rules=rep_rules, element_types=element_types)

    if element_types:
        for elem_name, types in element_types.items():
            rules.setdefault(elem_name, [])
            for t in sorted(types):
                if t not in rules[elem_name]:
                    rules[elem_name].append(t)

    if skip_rules:
        rules = {k: v for k, v in rules.items() if k not in skip_rules}

    lines = [f"{name} ::= {' | '.join(alts)}" for name, alts in rules.items()]
    return "\n".join(lines)


def add_minimal_grammar(
    input_path: str,
    output_path: str,
    grammar_path: str = "grammars/smcalflow.lark",
    generic: bool = False,
    start: str = "call",
    skip_rules: set[str] | None = None,
    program_key: str = "program",
    generic_terminals: frozenset[str] | None = None,
    normalize_repetition: bool = True,
) -> None:
    if generic_terminals is None and generic:
        generic_terminals = GENERIC_TERMINALS

    with open(input_path) as f:
        data = json.load(f)

    parser = _build_parser(grammar_path, start)
    apply_fixup = "smcalflow" in grammar_path
    rep_rules = _detect_repetition_rules(grammar_path) if normalize_repetition else None

    for i, entry in enumerate(data["data"]):
        tree = parser.parse(entry[program_key])
        if apply_fixup:
            tree = _fix_ambiguity(tree)
        rules: dict[str, list[str]] = {}
        element_types = {} if normalize_repetition else None
        _walk_tree(tree, rules, generic_terminals=generic_terminals,
                   repetition_rules=rep_rules, element_types=element_types)

        if element_types:
            for elem_name, types in element_types.items():
                rules.setdefault(elem_name, [])
                for t in sorted(types):
                    if t not in rules[elem_name]:
                        rules[elem_name].append(t)

        if skip_rules:
            rules = {k: v for k, v in rules.items() if k not in skip_rules}
        lines = [f"{name} ::= {' | '.join(alts)}" for name, alts in rules.items()]
        entry["minimal_grammar"] = "\n".join(lines)

        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(data['data'])} entries")

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Wrote {len(data['data'])} entries to {output_path}")


if __name__ == "__main__":
    fire.Fire(
        {
            "extract": extract_minimal_grammar,
            "add_minimal_grammar": add_minimal_grammar,
        }
    )
