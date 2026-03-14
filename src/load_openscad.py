import json
import os
import re
import statistics

import fire
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from grammar_parser import _build_parser, _detect_repetition_rules, _walk_tree
from grammar_utils import OPENSCAD_GENERIC_TERMINALS

SKIP_RULES = {"program"}

GRAMMAR_PATH = "grammars/openscad.lark"


def _extract_grammar(
    code: str,
    parser,
    generic_terminals: frozenset[str] | None = None,
    normalize_repetition: bool = True,
) -> str | None:
    try:
        tree = parser.parse(code)
    except Exception:
        return None

    rep_rules = _detect_repetition_rules(GRAMMAR_PATH) if normalize_repetition else None
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

    rules = {k: v for k, v in rules.items() if k not in SKIP_RULES}
    if not rules:
        return None
    lines = [f"{name} ::= {' | '.join(alts)}" for name, alts in rules.items()]
    return "\n".join(lines)


def load(
    output_dir: str = "data/openscad",
    test_size: float = 0.1,
    valid_size: float = 0.1,
    seed: int = 42,
    max_examples: int = 0,
    generic: bool = False,
    normalize_repetition: bool = True,
) -> None:
    print("Loading thingiverse-openscad dataset from HuggingFace...")
    ds = load_dataset("redcathode/thingiverse-openscad")
    data = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]
    if max_examples > 0:
        data = data.select(range(min(max_examples, len(data))))
    print(f"Loaded {len(data)} examples")

    print("Building OpenSCAD parser...")
    parser = _build_parser(GRAMMAR_PATH, start="program")

    generic_terminals = OPENSCAD_GENERIC_TERMINALS if generic else None
    print(f"Extracting minimal grammars{' (generic)' if generic else ''}...")

    successes = []
    failures = []
    code_lengths: list[int] = []

    for i, example in tqdm(enumerate(data), total=len(data), desc="Processing"):
        query = example.get("fakeprompt") or example.get("prompt") or example.get("description")
        raw_scad = example.get("scad") or example.get("code") or ""
        blocks = re.findall(r"```\n(.*?)```", raw_scad, re.DOTALL)
        code = blocks[-1].strip() if blocks else raw_scad.strip()

        if not query or not code:
            failures.append({"index": i, "reason": "empty description or code"})
            continue

        grammar = _extract_grammar(
            code, parser,
            generic_terminals=generic_terminals,
            normalize_repetition=normalize_repetition,
        )

        if grammar is not None:
            code_lengths.append(len(code.split()))
            successes.append({
                "query": query,
                "minimal_grammar": grammar,
                "program": code,
            })
        else:
            failures.append({"index": i, "reason": "parse failure"})

    print(f"\nParse results: {len(successes)}/{len(data)} succeeded "
          f"({100 * len(successes) / len(data):.1f}%)")
    print(f"Failures: {len(failures)}")

    if code_lengths:
        print(f"\nCode token stats:")
        print(f"  Min: {min(code_lengths)}")
        print(f"  Max: {max(code_lengths)}")
        print(f"  Mean: {statistics.mean(code_lengths):.1f}")
        print(f"  Median: {statistics.median(code_lengths):.1f}")

    os.makedirs(output_dir, exist_ok=True)

    if failures:
        fail_path = os.path.join(output_dir, "parse_failures.json")
        with open(fail_path, "w") as f:
            json.dump(failures, f, indent=2)
        print(f"Wrote {len(failures)} failures to {fail_path}")

    if not successes:
        print("No successful parses. Exiting.")
        return

    print("\nCreating train/valid/test splits...")
    indices = list(range(len(successes)))
    test_n = int(len(successes) * test_size)
    valid_n = int(len(successes) * valid_size)

    train_idx, test_idx = train_test_split(
        indices, test_size=test_n, random_state=seed
    )
    train_idx, valid_idx = train_test_split(
        train_idx, test_size=valid_n, random_state=seed
    )

    splits = {"train": train_idx, "valid": valid_idx, "test": test_idx}
    for split_name, idxs in splits.items():
        print(f"  {split_name}: {len(idxs)} examples")

    suffix = "_generic" if generic else ""
    for split_name, idxs in splits.items():
        split_data = [successes[i] for i in idxs]
        out_path = os.path.join(output_dir, f"{split_name}{suffix}.json")
        with open(out_path, "w") as f:
            json.dump({"data": split_data}, f, indent=2)
        print(f"Wrote {len(split_data)} entries to {out_path}")

    print("\nDone!")


if __name__ == "__main__":
    fire.Fire({"load": load})
