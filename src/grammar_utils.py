import re

GENERIC_TERMINALS = frozenset({"ESCAPED_STRING", "NUMBER"})
ENUM_TERMINALS = {"DAY", "MONTH", "WEEK", "HOLIDAY", "SEASON", "OP"}

VERILOG_GENERIC_TERMINALS = frozenset({
    "IDENTIFIER",
    "SIZED_NUMBER", "BASED_NUMBER", "DECIMAL_NUMBER", "REAL_NUMBER", "XZ_LITERAL",
    "ESCAPED_STRING",
    "MACRO_USAGE", "SYSTEM_TASK", "PREPROC_DIRECTIVE",
})

SPICE_GENERIC_TERMINALS = frozenset({
    "IDENTIFIER",
    "COMP_R", "COMP_C", "COMP_L", "COMP_V", "COMP_I",
    "COMP_D", "COMP_M", "COMP_Q", "COMP_J", "COMP_X", "COMP_K",
    "SI_VALUE", "NUMBER", "EXPRESSION",
    "NODE_ZERO",
})


def has_terminal_reference(alt: str, terminals: set[str]) -> bool:
    stripped = re.sub(r'"[^"]*"', "", alt)
    return any(re.search(rf"\b{t}\b", stripped) for t in terminals)


def parse_lark_grammar(text: str) -> dict[str, list[str]]:
    rules: dict[str, list[str]] = {}
    current_rule = None

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("//") or stripped.startswith("%"):
            continue

        if line[0] in (" ", "\t") and current_rule is not None:
            if stripped.startswith("|"):
                alt = stripped[1:].strip()
                if alt:
                    rules[current_rule].append(alt)
            continue

        m = re.match(r"^(\w+)\s*:\s*(.*)", line)
        if m:
            current_rule = m.group(1)
            rhs = m.group(2).strip()

            if current_rule not in rules:
                rules[current_rule] = []

            if rhs:
                for alt in rhs.split("|"):
                    alt = alt.strip()
                    if alt:
                        rules[current_rule].append(alt)

    return rules


def filter_rules(
    rules: dict[str, list[str]], exclude: set[str]
) -> dict[str, list[str]]:
    filtered = {}
    for name, alts in rules.items():
        if name in exclude:
            continue
        valid = [a for a in alts if not has_terminal_reference(a, exclude)]
        if valid:
            filtered[name] = valid
    return filtered


def parse_minimal_grammar(text: str) -> dict[str, list[str]]:
    rules: dict[str, list[str]] = {}
    for line in text.split("\n"):
        line = line.strip()
        if not line or "::=" not in line:
            continue
        name, rhs = line.split("::=", 1)
        alts = [a.strip() for a in rhs.split(" | ")]
        rules[name.strip()] = [a for a in alts if a]
    return rules


def reconstruct_minimal_grammar(rules: dict[str, list[str]]) -> str:
    return "\n".join(f'{name} ::= {" | ".join(alts)}' for name, alts in rules.items())
