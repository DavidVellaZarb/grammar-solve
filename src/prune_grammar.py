import json
import re
from pathlib import Path

import fire
from lark import Lark, Token, Tree

from grammar_parser import _fix_ambiguity
from grammar_utils import ENUM_TERMINALS, GENERIC_TERMINALS, parse_lark_grammar

_NAMED_TERMINALS = ENUM_TERMINALS | GENERIC_TERMINALS


def _build_parser(grammar_path: str) -> Lark:
    with open(grammar_path) as f:
        grammar_text = f.read()
    return Lark(grammar_text, start="call", parser="earley", keep_all_tokens=True)


def tokenize_alternative(text: str) -> list[tuple[str, str]]:
    tokens = []
    i = 0
    while i < len(text):
        if text[i] == '"':
            j = text.index('"', i + 1)
            tokens.append(("str", text[i + 1 : j]))
            i = j + 1
        elif text[i].isspace():
            i += 1
        elif re.match(r"\w", text[i]):
            m = re.match(r"\w+", text[i:])
            tokens.append(("ref", m.group()))
            i += m.end()
        else:
            i += 1
    return tokens


def normalize(alt_text: str) -> str:
    tokens = tokenize_alternative(alt_text)
    contents = [content for _, content in tokens]
    s = " ".join(contents)
    s = re.sub(r"(?<=\W)\s+(?=\W)", "", s)
    return s


def _reconstruct_lark_alt(children: list) -> str:
    parts = []
    token_buf = []
    for child in children:
        if isinstance(child, Tree):
            if token_buf:
                parts.append('"' + "".join(token_buf) + '"')
                token_buf = []
            parts.append(child.data)
        elif isinstance(child, Token):
            if child.type in _NAMED_TERMINALS:
                if token_buf:
                    parts.append('"' + "".join(token_buf) + '"')
                    token_buf = []
                parts.append(child.type)
            else:
                token_buf.append(str(child))
    if token_buf:
        parts.append('"' + "".join(token_buf) + '"')
    return " ".join(parts)


def _walk_tree(
    tree: Tree,
    used_alts: set[tuple[str, str]],
    used_terminal_values: dict[str, set[str]],
) -> None:
    alt = _reconstruct_lark_alt(tree.children)
    used_alts.add((tree.data, normalize(alt)))

    for child in tree.children:
        if isinstance(child, Tree):
            _walk_tree(child, used_alts, used_terminal_values)
        elif isinstance(child, Token) and child.type in ENUM_TERMINALS:
            used_terminal_values.setdefault(child.type, set()).add(str(child))


def find_used(
    grammar_path: str, dataset_paths: list[str],
) -> tuple[set[tuple[str, str]], dict[str, set[str]]]:
    parser = _build_parser(grammar_path)
    used_alts: set[tuple[str, str]] = set()
    used_terminal_values: dict[str, set[str]] = {}

    for path in dataset_paths:
        with open(path) as f:
            data = json.load(f)["data"]
        for entry in data:
            tree = parser.parse(entry["program"])
            tree = _fix_ambiguity(tree)
            _walk_tree(tree, used_alts, used_terminal_values)

    return used_alts, used_terminal_values


def find_prunable(
    grammar_path: str,
    used_alts: set[tuple[str, str]],
    used_terminal_values: dict[str, set[str]],
) -> tuple[set[tuple[str, str]], dict[str, set[str]]]:
    with open(grammar_path) as f:
        all_parsed = parse_lark_grammar(f.read())

    prunable_alts: set[tuple[str, str]] = set()
    prunable_terminal_values: dict[str, set[str]] = {}

    for name, alts in all_parsed.items():
        if name in ENUM_TERMINALS:
            unused = set()
            for alt in alts:
                value = alt.strip('"')
                if name not in used_terminal_values or value not in used_terminal_values[name]:
                    unused.add(value)
            if unused:
                prunable_terminal_values[name] = unused
        elif name.isupper():
            continue
        else:
            for alt in alts:
                if (name, normalize(alt)) not in used_alts:
                    prunable_alts.add((name, alt))

    return prunable_alts, prunable_terminal_values


def write_pruned_grammar(
    input_path: str,
    output_path: str,
    prunable_alts: set[tuple[str, str]],
    prunable_terminal_values: dict[str, set[str]],
):
    with open(input_path) as f:
        text = f.read()

    blocks = []
    current_def = None

    for line in text.splitlines():
        stripped = line.strip()

        if not stripped or stripped.startswith("//") or stripped.startswith("%"):
            if current_def:
                blocks.append(("def", current_def))
                current_def = None
            blocks.append(("pass", line))
            continue

        m = re.match(r"^(\w+)\s*:\s*(.*)", line)
        if m:
            if current_def:
                blocks.append(("def", current_def))
            name = m.group(1)
            is_enum_terminal = name in ENUM_TERMINALS
            rhs = m.group(2).strip()
            alts = []
            if rhs:
                for alt_text in rhs.split("|"):
                    alt_text = alt_text.strip()
                    if alt_text:
                        alts.append(alt_text)
            current_def = [name, is_enum_terminal, alts]
            continue

        if line[0] in (" ", "\t") and current_def is not None:
            if stripped.startswith("|"):
                alt_text = stripped[1:].strip()
                if alt_text:
                    current_def[2].append(alt_text)
            continue

        if current_def:
            blocks.append(("def", current_def))
            current_def = None
        blocks.append(("pass", line))

    if current_def:
        blocks.append(("def", current_def))

    output_lines = []
    for block_type, block_data in blocks:
        if block_type == "pass":
            output_lines.append(block_data)
            continue

        name, is_enum_terminal, alts = block_data

        if is_enum_terminal:
            kept = []
            for alt in alts:
                value = alt.strip('"')
                if name in prunable_terminal_values and value in prunable_terminal_values[name]:
                    continue
                kept.append(alt)
        else:
            kept = []
            for alt in alts:
                if (name, alt) in prunable_alts:
                    continue
                kept.append(alt)

        if not kept:
            continue

        indent = "\t" if is_enum_terminal else "    "
        output_lines.append(f"{name}: {kept[0]}")
        for alt in kept[1:]:
            output_lines.append(f"{indent}| {alt}")

    with open(output_path, "w") as f:
        f.write("\n".join(output_lines) + "\n")


def main(grammar_path: str, *dataset_paths: str, dry_run: bool = True):
    used_alts, used_terminal_values = find_used(grammar_path, list(dataset_paths))
    prunable_alts, prunable_terminal_values = find_prunable(
        grammar_path, used_alts, used_terminal_values
    )

    if dry_run:
        alts_by_rule: dict[str, list[str]] = {}
        for rule_name, alt in sorted(prunable_alts):
            alts_by_rule.setdefault(rule_name, []).append(alt)

        print("=== Prunable alternatives ===\n")
        for rule_name in sorted(alts_by_rule):
            alts = alts_by_rule[rule_name]
            print(f"{rule_name}:")
            for alt in sorted(alts):
                print(f"  {alt}")
            print()

        if prunable_terminal_values:
            print("=== Prunable terminal values ===\n")
            for terminal in sorted(prunable_terminal_values):
                values = prunable_terminal_values[terminal]
                print(f"{terminal}:")
                for value in sorted(values):
                    print(f'  "{value}"')
                print()

        print("=== Summary ===")
        print(f"Prunable alternatives: {len(prunable_alts)}")
        total_terminal_values = sum(
            len(v) for v in prunable_terminal_values.values()
        )
        print(f"Prunable terminal values: {total_terminal_values}")
    else:
        p = Path(grammar_path)
        output_path = str(p.with_stem(p.stem + "_pruned"))
        write_pruned_grammar(grammar_path, output_path, prunable_alts, prunable_terminal_values)

        print(f"Wrote pruned grammar to {output_path}")
        print(f"Pruned alternatives: {len(prunable_alts)}")
        total_terminal_values = sum(
            len(v) for v in prunable_terminal_values.values()
        )
        print(f"Pruned terminal values: {total_terminal_values}")


if __name__ == "__main__":
    fire.Fire(main)
