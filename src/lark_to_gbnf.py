import re

import fire

from grammar_utils import parse_lark_grammar

BUILTIN_TERMINAL_DEFS = {
    "NUMBER": r'[0-9]+ ("." [0-9]*)? (("e" | "E") ("+" | "-")? [0-9]+)?',
    "ESCAPED_STRING": r'"\"" [^"]* "\""',
}


def _tokenize_alt(alt: str) -> list[tuple[str, str]]:
    tokens = []
    i = 0
    while i < len(alt):
        if alt[i] == '"':
            j = i + 1
            while j < len(alt):
                if alt[j] == '\\':
                    j += 2
                    continue
                if alt[j] == '"':
                    break
                j += 1
            j += 1
            tokens.append(("literal", alt[i:j]))
            i = j
        elif alt[i] in (" ", "\t"):
            i += 1
        elif alt[i] == "_" or alt[i].isalpha():
            j = i
            while j < len(alt) and (alt[j] == "_" or alt[j].isalnum()):
                j += 1
            tokens.append(("ref", alt[i:j]))
            i = j
        else:
            i += 1
    return tokens


def _alt_to_gbnf(alt: str) -> str:
    tokens = _tokenize_alt(alt)
    if not tokens:
        return '""'
    parts = []
    for i, (_, val) in enumerate(tokens):
        if i > 0:
            parts.append("ws")
        parts.append(val)
    return " ".join(parts)


def _find_imports(grammar_text: str) -> set[str]:
    imports = set()
    for line in grammar_text.splitlines():
        m = re.match(r"^%import\s+common\.(\w+)", line.strip())
        if m:
            imports.add(m.group(1))
    return imports


def lark_to_gbnf(
    grammar_path: str,
    start: str = "call",
    generic_terminals: set[str] | None = None,
) -> str:
    with open(grammar_path) as f:
        raw_text = f.read()

    rules = parse_lark_grammar(raw_text)
    imports = _find_imports(raw_text)

    if generic_terminals is None:
        generic_terminals = imports - {"WS"}

    lines = [
        f"root ::= ws {start} ws",
        r"ws ::= [ \t\n]*",
    ]

    for name, alts in rules.items():
        gbnf_alts = [_alt_to_gbnf(a) for a in alts]
        lines.append(f'{name} ::= {" | ".join(gbnf_alts)}')

    for term in sorted(generic_terminals):
        if term in BUILTIN_TERMINAL_DEFS and term not in rules:
            lines.append(f"{term} ::= {BUILTIN_TERMINAL_DEFS[term]}")

    return "\n".join(lines)


def convert(
    grammar_path: str, start: str = "call", output_path: str | None = None
) -> None:
    gbnf = lark_to_gbnf(grammar_path, start=start)
    if output_path:
        with open(output_path, "w") as f:
            f.write(gbnf)
        print(f"Saved GBNF to {output_path}")
    else:
        print(gbnf)


if __name__ == "__main__":
    fire.Fire(convert)
