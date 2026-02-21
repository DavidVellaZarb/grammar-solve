import json
from functools import lru_cache

import fire
from lark import Lark, Token, Tree

_GENERIC_TERMINALS = frozenset({"ESCAPED_STRING", "NUMBER"})


@lru_cache(maxsize=1)
def _build_parser(grammar_path: str) -> Lark:
    with open(grammar_path) as f:
        grammar_text = f.read()
    return Lark(
        grammar_text,
        start="call",
        parser="earley",
        keep_all_tokens=True,
    )


def _fix_ambiguity(tree: Tree) -> Tree:
    if not isinstance(tree, Tree):
        return tree

    # Fix: date → day → (adjustByPeriod day period) ⇒ date → (adjustByPeriod date period)
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


def _reconstruct_alt(children: list, generic: bool) -> str:
    parts: list[str] = []
    token_buf: list[str] = []

    for child in children:
        if isinstance(child, Tree):
            if token_buf:
                parts.append('"' + "".join(token_buf) + '"')
                token_buf = []
            parts.append(child.data)
        elif isinstance(child, Token):
            if generic and child.type in _GENERIC_TERMINALS:
                if token_buf:
                    parts.append('"' + "".join(token_buf) + '"')
                    token_buf = []
                parts.append(child.type)
            else:
                token_buf.append(str(child))

    if token_buf:
        parts.append('"' + "".join(token_buf) + '"')

    return " ".join(parts)


def _walk_tree(tree: Tree, rules: dict[str, list[str]], generic: bool) -> None:
    rule_name = tree.data
    alt = _reconstruct_alt(tree.children, generic)

    if rule_name not in rules:
        rules[rule_name] = []
    if alt not in rules[rule_name]:
        rules[rule_name].append(alt)

    for child in tree.children:
        if isinstance(child, Tree):
            _walk_tree(child, rules, generic)


def extract_minimal_grammar(
    program: str,
    grammar_path: str = "grammars/smcalflow.lark",
    generic: bool = False,
) -> str:
    parser = _build_parser(grammar_path)
    tree = parser.parse(program)
    tree = _fix_ambiguity(tree)

    rules: dict[str, list[str]] = {}
    _walk_tree(tree, rules, generic)

    lines = [f"{name} ::= {' | '.join(alts)}" for name, alts in rules.items()]
    return "\n".join(lines)


def add_minimal_grammar(
    input_path: str,
    output_path: str,
    grammar_path: str = "grammars/smcalflow.lark",
    generic: bool = False,
) -> None:
    with open(input_path) as f:
        data = json.load(f)

    parser = _build_parser(grammar_path)

    for i, entry in enumerate(data["data"]):
        tree = parser.parse(entry["program"])
        tree = _fix_ambiguity(tree)
        rules: dict[str, list[str]] = {}
        _walk_tree(tree, rules, generic)
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
