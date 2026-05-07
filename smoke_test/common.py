from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

from datasets import load_dataset
from lark import Lark

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from grammar_parser import _detect_repetition_rules, _walk_tree  # noqa: E402

TRAIN_SIZE = 1000
VALID_SIZE = 100
TEST_SIZE = 200
TOTAL_SIZE = TRAIN_SIZE + VALID_SIZE + TEST_SIZE

COMMON_GENERIC_TERMINALS = frozenset(
    {
        "ATTR_VALUE",
        "BACKTICK_IDENTIFIER",
        "BASED_NUMBER",
        "BIT_STRING",
        "BLOCK_STRING",
        "BRACKET_IDENTIFIER",
        "DOUBLE_QUOTED_STRING",
        "ESCAPED_STRING",
        "FLOAT",
        "IDENTIFIER",
        "INT",
        "IRIREF",
        "NAME",
        "NUMBER",
        "PARAMETER",
        "PNAME",
        "PNAME_NS",
        "RAW_TOKEN",
        "SIGNED_NUMBER",
        "SINGLE_QUOTED_STRING",
        "STRING",
        "TEXT",
        "UNQUOTED_VALUE",
        "VAR",
        "VARIABLE",
    }
)


def repo_path(*parts: str) -> Path:
    return REPO_ROOT.joinpath(*parts)


def clean_code_block(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).replace("\r\n", "\n").replace("\r", "\n").strip()
    fenced = re.search(r"```[^\n`]*\n(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    text = re.sub(
        r"^\s*(?:answer|program|query|sql|sparql|graphql|json|vhdl|svg|selfies)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text.strip()


def collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def strip_sql_semicolon(text: str) -> str:
    return re.sub(r"\s*;\s*$", "", text.strip())


def _extract_balanced(text: str, opener: str, closer: str) -> str:
    start = text.find(opener)
    if start < 0:
        return text
    depth = 0
    in_string = False
    quote = ""
    escape = False
    for i, ch in enumerate(text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                in_string = False
            continue
        if ch in {"'", '"'}:
            in_string = True
            quote = ch
            continue
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]


def canonical_json(text: Any) -> str:
    text = clean_code_block(text)
    if not text.lstrip().startswith(("{", "[")):
        text = _extract_balanced(text, "{", "}")
    try:
        return json.dumps(json.loads(text), sort_keys=True, separators=(",", ":"))
    except Exception:
        return collapse_ws(text)


def canonical_xml(text: Any) -> str:
    text = clean_code_block(text)
    lower = text.lower()
    idx = lower.find("<svg")
    if idx >= 0:
        text = text[idx:]
    text = re.sub(r"<\?xml[^>]*\?>", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"<!DOCTYPE[^>]*>", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL).strip()
    text = re.sub(r">\s+<", "><", text)
    return collapse_ws(text)


def canonical_vhdl(text: Any) -> str:
    text = clean_code_block(text)
    match = re.search(
        r"(?is)\b(?:library|use|entity|architecture|package|configuration|signal|process)\b",
        text,
    )
    if match:
        text = text[match.start() :]
    text = re.sub(r"--[^\n]*", "", text)
    return collapse_ws(text)


def canonical_program(domain: str, text: Any) -> str:
    text = clean_code_block(text)
    if domain == "vega_lite":
        return canonical_json(text)
    if domain == "restricted_graphics":
        return canonical_xml(text)
    if domain == "selfies":
        return re.sub(r"\s+", "", text)
    if domain == "vhdl":
        return canonical_vhdl(text)
    if domain == "text_to_sql":
        return strip_sql_semicolon(collapse_ws(text))
    if domain in {"sparql", "graphql"}:
        return collapse_ws(text)
    return collapse_ws(text)


def value_at_path(example: dict[str, Any], path: str) -> Any:
    current: Any = example
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def first_text(example: dict[str, Any], candidates: list[str]) -> str | None:
    for name in candidates:
        value = value_at_path(example, name)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (dict, list)):
            encoded = json.dumps(value, ensure_ascii=False)
            if encoded.strip():
                return encoded
    return None


def iter_string_values(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        if value.strip():
            yield value.strip()
    elif isinstance(value, dict):
        for child in value.values():
            yield from iter_string_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_string_values(child)


def conversation_pair(example: dict[str, Any]) -> tuple[str | None, str | None]:
    for name in ["messages", "conversations", "conversation", "dialogue"]:
        turns = value_at_path(example, name)
        if not isinstance(turns, list):
            continue
        user_text = None
        assistant_text = None
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            role = str(
                turn.get("role")
                or turn.get("from")
                or turn.get("speaker")
                or turn.get("author")
                or ""
            ).lower()
            content = (
                turn.get("content")
                or turn.get("value")
                or turn.get("text")
                or turn.get("message")
            )
            if not isinstance(content, str) or not content.strip():
                continue
            if role in {"user", "human", "prompt"} and user_text is None:
                user_text = content.strip()
            elif role in {"assistant", "gpt", "model", "bot"} and assistant_text is None:
                assistant_text = content.strip()
        if user_text or assistant_text:
            return user_text, assistant_text
    return None, None


def iter_hf_records(
    dataset_name: str,
    *,
    config_name: str | None = None,
    split_names: tuple[str, ...] = ("train", "validation", "valid", "test"),
    streaming: bool = True,
    trust_remote_code: bool = False,
) -> Iterator[dict[str, Any]]:
    found_split = False
    for split in split_names:
        try:
            if config_name:
                ds = load_dataset(
                    dataset_name,
                    config_name,
                    split=split,
                    streaming=streaming,
                    trust_remote_code=trust_remote_code,
                )
            else:
                ds = load_dataset(
                    dataset_name,
                    split=split,
                    streaming=streaming,
                    trust_remote_code=trust_remote_code,
                )
        except Exception as exc:
            print(f"Skipping split {split!r}: {exc}")
            continue
        found_split = True
        for example in ds:
            yield dict(example)
    if not found_split:
        raise RuntimeError(f"No usable splits found for Hugging Face dataset {dataset_name!r}")


def build_parser(grammar_path: Path, start: str) -> Lark:
    with grammar_path.open() as f:
        grammar_text = f.read()
    return Lark(grammar_text, start=start, parser="earley", keep_all_tokens=True)


def minimal_grammar_for_program(
    program: str,
    parser: Lark,
    grammar_path: Path,
    *,
    generic_terminals: frozenset[str] = COMMON_GENERIC_TERMINALS,
    skip_rules: frozenset[str] = frozenset(),
    normalize_repetition: bool = True,
    position_aware_spacing: bool = True,
) -> str:
    tree = parser.parse(program)
    repetition_rules = (
        _detect_repetition_rules(str(grammar_path)) if normalize_repetition else None
    )
    element_types: dict[str, set[str]] | None = {} if normalize_repetition else None
    rules: dict[str, list[str]] = {}
    _walk_tree(
        tree,
        rules,
        generic_terminals=generic_terminals,
        repetition_rules=repetition_rules,
        element_types=element_types,
        position_aware_spacing=position_aware_spacing,
    )
    if element_types:
        for elem_name, types in element_types.items():
            rules.setdefault(elem_name, [])
            for rule_type in sorted(types):
                if rule_type not in rules[elem_name]:
                    rules[elem_name].append(rule_type)
    if skip_rules:
        rules = {name: alts for name, alts in rules.items() if name not in skip_rules}
    return "\n".join(f"{name} ::= {' | '.join(alts)}" for name, alts in rules.items())


ExtractFn = Callable[[dict[str, Any]], tuple[str, str] | None]


def write_smoke_splits(
    *,
    domain: str,
    dataset: Iterable[dict[str, Any]],
    extract: ExtractFn,
    output_dir: str | Path,
    grammar_path: str | Path,
    start: str,
    generic_terminals: frozenset[str] = COMMON_GENERIC_TERMINALS,
    skip_rules: frozenset[str] = frozenset(),
    max_scan: int = 100_000,
    max_program_chars: int = 8_000,
    min_program_chars: int = 1,
    normalize_repetition: bool = True,
    position_aware_spacing: bool = True,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    grammar_file = Path(grammar_path)
    if not grammar_file.is_absolute():
        grammar_file = REPO_ROOT / grammar_file
    parser = build_parser(grammar_file, start=start)

    records: list[dict[str, str]] = []
    failures: list[dict[str, Any]] = []
    scanned = 0

    for raw in dataset:
        scanned += 1
        if scanned > max_scan:
            break

        pair = extract(raw)
        if pair is None:
            if len(failures) < 200:
                failures.append({"index": scanned - 1, "reason": "missing fields"})
            continue

        query, program = pair
        query = str(query).strip()
        program = canonical_program(domain, program)
        if not query or not program:
            if len(failures) < 200:
                failures.append({"index": scanned - 1, "reason": "empty query/program"})
            continue
        if len(program) < min_program_chars or len(program) > max_program_chars:
            if len(failures) < 200:
                failures.append(
                    {
                        "index": scanned - 1,
                        "reason": "program length outside bounds",
                        "chars": len(program),
                    }
                )
            continue

        try:
            minimal_grammar = minimal_grammar_for_program(
                program,
                parser,
                grammar_file,
                generic_terminals=generic_terminals,
                skip_rules=skip_rules,
                normalize_repetition=normalize_repetition,
                position_aware_spacing=position_aware_spacing,
            )
        except Exception as exc:
            if len(failures) < 200:
                failures.append(
                    {
                        "index": scanned - 1,
                        "reason": "parse failure",
                        "error": str(exc)[:500],
                        "program": program[:500],
                    }
                )
            continue

        if not minimal_grammar.strip():
            if len(failures) < 200:
                failures.append({"index": scanned - 1, "reason": "empty grammar"})
            continue

        records.append(
            {
                "query": query,
                "program": program,
                "minimal_grammar": minimal_grammar,
            }
        )
        if len(records) % 100 == 0:
            print(f"{domain}: collected {len(records)}/{TOTAL_SIZE} examples")
        if len(records) >= TOTAL_SIZE:
            break

    if failures:
        with (output_path / "parse_failures.json").open("w") as f:
            json.dump({"data": failures}, f, indent=2)

    if len(records) < TOTAL_SIZE:
        raise RuntimeError(
            f"{domain}: collected {len(records)} parseable examples after scanning "
            f"{scanned}; need {TOTAL_SIZE}. Inspect {output_path / 'parse_failures.json'}."
        )

    splits = {
        "train": records[:TRAIN_SIZE],
        "valid": records[TRAIN_SIZE : TRAIN_SIZE + VALID_SIZE],
        "test": records[TRAIN_SIZE + VALID_SIZE : TOTAL_SIZE],
    }
    for split_name, rows in splits.items():
        with (output_path / f"{split_name}.json").open("w") as f:
            json.dump({"data": rows}, f, indent=2)
        print(f"{domain}: wrote {len(rows)} examples to {output_path / f'{split_name}.json'}")
