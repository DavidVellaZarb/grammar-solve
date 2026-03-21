import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pytest
from lark import Lark

from grammar_utils import parse_minimal_grammar

# ---------------------------------------------------------------------------
# Domain configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DomainConfig:
    name: str
    data_paths: tuple[str, ...]
    start: str
    scaffold: Callable[[dict[str, list[str]]], str]
    prepare_program: Callable[[dict], str] | None = None
    prepare_literal: Callable[[str], str] | None = None
    max_entries: int = 0  # 0 = all
    max_fail_rate: float = 0.0
    extra_rules: dict[str, list[str]] | None = None


def _no_scaffold(_rules: dict[str, list[str]]) -> str:
    return ""


def _smiles_scaffold(rules: dict[str, list[str]]) -> str:
    candidates = ["atom", "bond", "branch", "ring_closure"]
    present = [c for c in candidates if c in rules]
    if not present:
        return ""
    return f'smiles: ({" | ".join(present)})+'


def _find_root_rules(rules: dict[str, list[str]]) -> list[str]:
    """Return rules not referenced as tokens in any other rule's alternatives."""
    referenced: set[str] = set()
    for alts in rules.values():
        for alt in alts:
            # Strip quoted literals, then find bare words
            stripped = re.sub(r'"[^"]*"', "", alt)
            referenced.update(re.findall(r"\b([a-z_]\w*)\b", stripped))
    return sorted(r for r in rules if r not in referenced)


def _auto_scaffold(rules: dict[str, list[str]]) -> str:
    roots = _find_root_rules(rules)
    if not roots:
        roots = sorted(rules.keys())
    return f'_start: ({" | ".join(roots)})+'


OPENSCAD_TOP_RULES = [
    "assignment", "module_call", "module_def", "function_def",
    "if_statement", "for_statement", "let_statement",
    "use_statement", "include_statement",
]


def _openscad_scaffold(rules: dict[str, list[str]]) -> str:
    present = [r for r in OPENSCAD_TOP_RULES if r in rules]
    if not present:
        return _auto_scaffold(rules)
    return f'_start: ({" | ".join(present)})+'


def _spice_scaffold(rules: dict[str, list[str]]) -> str:
    all_rules = sorted(rules.keys())
    if not all_rules:
        return ""
    return f'_start: ({" | ".join(all_rules)})+'


def _spice_prepare(entry: dict) -> str:
    """Strip title line and .end from SPICE netlists."""
    lines = entry["program"].split("\n")
    body = []
    for line in lines[1:]:
        if line.strip().lower() == ".end":
            break
        body.append(line)
    return "\n".join(body) + "\n" if body else ""


def _openscad_prepare(entry: dict) -> str:
    """Strip comments from OpenSCAD programs."""
    text = entry["program"]
    # Block comments
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    # Line comments
    text = re.sub(r"//[^\n]*", "", text)
    return text


def _verilog_prepare(entry: dict) -> str:
    """Strip module header and endmodule from Verilog programs."""
    text = entry["program"]
    # Strip module header: "module name(...);" — only in eval data
    text = re.sub(r"^module\s+\w+\s*\([^)]*\)\s*;", "", text, flags=re.DOTALL)
    # Strip endmodule
    text = re.sub(r"\bendmodule\b\s*$", "", text)
    return text


_OPENSCAD_KEYWORD_RE = re.compile(r"\b(module|function|use|include)(?=\w)")


def _openscad_split_keywords(text: str) -> str:
    """Insert spaces after OpenSCAD keywords concatenated with identifiers."""
    return _OPENSCAD_KEYWORD_RE.sub(r"\1 ", text)


DOMAINS = [
    DomainConfig(
        name="smcalflow",
        data_paths=("data/smcalflow/train.json", "data/smcalflow/test.json"),
        start="call",
        scaffold=_no_scaffold,
        max_entries=500,
    ),
    DomainConfig(
        name="smiles",
        data_paths=("data/smiles/train.json", "data/smiles/test.json"),
        start="smiles",
        scaffold=_smiles_scaffold,
        max_entries=500,
    ),
    DomainConfig(
        name="spice",
        data_paths=("data/spice/train.json", "data/spice/test.json"),
        start="_start",
        scaffold=_spice_scaffold,
        prepare_program=_spice_prepare,
        max_entries=500,
        max_fail_rate=0.02,
    ),
    DomainConfig(
        name="openscad",
        data_paths=("data/openscad/train.json", "data/openscad/test.json"),
        start="_start",
        scaffold=_openscad_scaffold,
        prepare_program=_openscad_prepare,
        prepare_literal=_openscad_split_keywords,
        max_entries=200,
        max_fail_rate=0.05,
    ),
    DomainConfig(
        name="verilog",
        data_paths=(
            "data/mg_verilog/train_detailed.json",
            "data/verilog_eval/VerilogEval_Human_gold.json",
        ),
        start="_start",
        scaffold=_spice_scaffold,
        prepare_program=_verilog_prepare,
        max_entries=200,
        max_fail_rate=0.08,
        extra_rules={"port_dir": ['"input"', '"output"', '"inout"']},
    ),
    DomainConfig(
        name="geoquery",
        data_paths=("data/geoquery/train.json", "data/geoquery/test.json"),
        start="query",
        scaffold=lambda _rules: 'query: "answer(" answer_type ")"',
        max_entries=500,
    ),
    DomainConfig(
        name="overnight",
        data_paths=("data/overnight/train.json", "data/overnight/test.json"),
        start="list_value",
        scaffold=_auto_scaffold,
        max_entries=500,
    ),
]

# Filter to domains whose data files exist
domain_params = [
    pytest.param(d, id=d.name)
    for d in DOMAINS
    if all(Path(p).exists() for p in d.data_paths)
]

# ---------------------------------------------------------------------------
# Minimal grammar → Lark conversion
# ---------------------------------------------------------------------------

_Q_TERMINAL = '_Q: /"/\n'


def _remove_empty_rules(rules: dict[str, list[str]]) -> None:
    """Remove rules with no alternatives and strip their references."""
    while True:
        empty = {
            name for name, alts in rules.items()
            if not alts or all(not a.strip() for a in alts)
        }
        if not empty:
            break
        for name in list(rules):
            if name in empty:
                del rules[name]
                continue
            new_alts = []
            for alt in rules[name]:
                words = alt.split()
                filtered = [w for w in words if w.strip('"') not in empty]
                if filtered:
                    new_alts.append(" ".join(filtered))
            rules[name] = new_alts


def _to_lark(minimal_grammar_text: str, config: DomainConfig) -> str:
    rules = parse_minimal_grammar(minimal_grammar_text)

    # Add extra rules for known missing definitions
    if config.extra_rules:
        for name, alts in config.extra_rules.items():
            if name not in rules and any(
                name in alt for alts_ in rules.values() for alt in alts_
            ):
                rules[name] = list(alts)

    # Strip NEWLINE references (whitespace is already ignored via %ignore)
    for name in list(rules):
        rules[name] = [
            re.sub(r"\bNEWLINE\b", "", alt).strip()
            for alt in rules[name]
        ]
        rules[name] = [a for a in rules[name] if a]
    rules = {k: v for k, v in rules.items() if v}

    # Remove empty rules and strip their references from other rules
    _remove_empty_rules(rules)

    # Make the start rule known so cross-references (e.g. SMILES branch→smiles) work
    if config.start not in rules:
        rules[config.start] = []

    needs_q = False
    lark_lines: list[str] = []

    for name, alts in rules.items():
        converted_alts: list[str] = []
        for alt in alts:
            tokens = _convert_alt(alt, config.prepare_literal, rules)
            if not tokens:
                continue
            if "_Q" in tokens:
                needs_q = True
            converted_alts.append(" ".join(tokens))
        if converted_alts:
            lark_lines.append(f'{name}: {" | ".join(converted_alts)}')

    scaffold = config.scaffold(rules)
    parts: list[str] = []
    if scaffold:
        parts.append(scaffold)
    parts.extend(lark_lines)
    if needs_q:
        parts.append(_Q_TERMINAL.strip())
    parts.append(r"%ignore /\s+/")
    return "\n".join(parts)


def _convert_alt(
    alt: str,
    prepare_literal: Callable[[str], str] | None,
    rules: dict[str, list[str]],
) -> list[str]:
    """Convert a minimal grammar alternative to Lark tokens.

    Uses known rule names as anchors to distinguish rule references from
    literal text. Handles ``""...""`` escaped strings (SMCalFlow).
    """
    # Handle ""..."" escaped strings first
    first = alt.find('""')
    if first != -1:
        last = alt.rfind('""', first + 2)
        if last > first:
            tokens: list[str] = []
            tokens.extend(
                _process_words(alt[:first], rules, prepare_literal)
            )
            inner = alt[first + 2 : last]
            if prepare_literal:
                inner = prepare_literal(inner)
            tokens.extend(["_Q"] + _split_literal(inner) + ["_Q"])
            tokens.extend(
                _process_words(alt[last + 2 :], rules, prepare_literal)
            )
            return tokens

    return _process_words(alt, rules, prepare_literal)


def _process_words(
    text: str,
    rules: dict[str, list[str]],
    prepare_literal: Callable[[str], str] | None,
) -> list[str]:
    """Parse alt text into Lark tokens using known rule names as anchors.

    Splits by whitespace. Words matching a defined rule name (after
    stripping ``"`` delimiters) become rule references. Everything else
    accumulates into a literal buffer, which is joined, stripped of outer
    grammar-delimiter quotes, and token-split.
    """
    rule_names = set(rules.keys())
    words = text.split()
    tokens: list[str] = []
    literal_buf: list[str] = []

    for word in words:
        bare = word.strip('"')
        if bare in rule_names:
            if literal_buf:
                tokens.extend(
                    _flush_literal_buf(literal_buf, prepare_literal)
                )
                literal_buf = []
            tokens.append(bare)
        else:
            literal_buf.append(word)

    if literal_buf:
        tokens.extend(_flush_literal_buf(literal_buf, prepare_literal))

    return tokens


def _flush_literal_buf(
    buf: list[str], prepare_literal: Callable[[str], str] | None
) -> list[str]:
    """Join buffered literal words, strip outer grammar quotes, and token-split."""
    text = " ".join(buf)
    # Strip leading/trailing grammar-delimiter quotes
    if text.startswith('"'):
        text = text[1:]
    if text.endswith('"'):
        text = text[:-1]
    if prepare_literal:
        text = prepare_literal(text)
    return _split_literal(text)


def _split_literal(text: str) -> list[str]:
    """Split text into individual token literals for Lark."""
    if not text:
        return []
    # Split into word chars and individual punctuation
    pieces = re.findall(r"\w+|\S", text)
    result: list[str] = []
    for p in pieces:
        if p == '"':
            result.append("_Q")
        else:
            escaped = p.replace("\\", "\\\\")
            result.append(f'"{escaped}"')
    return result


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def _load_entries(config: DomainConfig) -> list[dict]:
    entries = []
    for path in config.data_paths:
        with open(path) as f:
            entries.extend(json.load(f)["data"])
    return entries


@pytest.mark.parametrize("config", domain_params)
def test_minimal_grammar_parses_program(config: DomainConfig):
    entries = _load_entries(config)
    if config.max_entries:
        entries = entries[: config.max_entries]

    failures: list[str] = []
    tested = 0

    for i, entry in enumerate(entries):
        program = (
            config.prepare_program(entry)
            if config.prepare_program
            else entry["program"]
        )
        if not program.strip():
            continue
        tested += 1

        try:
            grammar = _to_lark(entry["minimal_grammar"], config)
            parser = Lark(grammar, start=config.start, parser="earley")
            parser.parse(program)
        except Exception as e:
            failures.append(f"[{i}] {str(e)[:200]}")

    fail_rate = len(failures) / tested if tested else 0
    assert fail_rate <= config.max_fail_rate, (
        f"{len(failures)}/{tested} failures ({fail_rate:.1%}):\n"
        + "\n".join(failures[:20])
    )
