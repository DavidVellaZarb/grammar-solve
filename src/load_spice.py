import json
import os
import re
import statistics

import fire
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from grammar_parser import _build_parser, _detect_repetition_rules, _walk_tree
from grammar_utils import SPICE_GENERIC_TERMINALS

SKIP_RULES = {"netlist", "netlist_body", "title_line", "end_line", "comment_line",
              "subckt_body"}

GRAMMAR_PATH = "grammars/spice.lark"


def _preprocess_netlist(raw: str) -> str:
    """Merge continuation lines (+), strip comments (*), normalize whitespace."""
    lines = raw.splitlines()
    merged: list[str] = []

    for line in lines:
        stripped = line.rstrip()
        if not stripped:
            continue
        if stripped.startswith("*"):
            continue
        if stripped.startswith("+") and merged:
            merged[-1] = merged[-1] + " " + stripped[1:].strip()
        else:
            merged.append(stripped)

    has_end = any(l.strip().lower() == ".end" for l in merged)
    if not has_end:
        merged.append(".end")

    result = "\n".join(re.sub(r"[ \t]+", " ", l) for l in merged)
    return result


def _extract_grammar(
    netlist: str,
    parser,
    generic_terminals: frozenset[str] | None = None,
    normalize_repetition: bool = True,
) -> str | None:
    """Preprocess, parse, walk tree, return BNF minimal grammar."""
    preprocessed = _preprocess_netlist(netlist)

    non_empty_lines = [l for l in preprocessed.splitlines()
                       if l.strip() and l.strip().lower() != ".end"]
    if len(non_empty_lines) <= 1:
        return None

    try:
        tree = parser.parse(preprocessed)
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
    lines = [f"{name} ::= {' | '.join(alts)}" for name, alts in rules.items()]
    return "\n".join(lines)


def load(
    output_dir: str = "data/spice",
    test_size: float = 0.1,
    valid_size: float = 0.1,
    seed: int = 42,
    max_examples: int = 0,
    generic: bool = False,
    normalize_repetition: bool = True,
) -> None:
    print("Loading Masala-CHAI dataset from HuggingFace...")
    ds = load_dataset("bhatvineet/masala-chai")
    data = ds["all"] if "all" in ds else ds[list(ds.keys())[0]]
    if max_examples > 0:
        data = data.select(range(min(max_examples, len(data))))
    print(f"Loaded {len(data)} examples")

    print("Building SPICE parser...")
    parser = _build_parser(GRAMMAR_PATH, start="netlist")

    generic_terminals = SPICE_GENERIC_TERMINALS if generic else None
    print(f"Extracting minimal grammars{' (generic)' if generic else ''}...")

    successes = []
    failures = []
    netlist_lengths: list[int] = []

    for i, example in tqdm(enumerate(data), total=len(data), desc="Processing"):
        query = example["description"]
        netlist = example["spice"]

        if not query or not netlist:
            failures.append({"index": i, "reason": "empty description or netlist"})
            continue

        grammar = _extract_grammar(
            netlist, parser,
            generic_terminals=generic_terminals,
            normalize_repetition=normalize_repetition,
        )

        if grammar is not None:
            preprocessed = _preprocess_netlist(netlist)
            netlist_lengths.append(len(preprocessed.split()))
            successes.append({
                "query": query,
                "minimal_grammar": grammar,
                "program": preprocessed,
            })
        else:
            failures.append({"index": i, "reason": "parse failure"})

    print(f"\nParse results: {len(successes)}/{len(data)} succeeded "
          f"({100 * len(successes) / len(data):.1f}%)")
    print(f"Failures: {len(failures)}")

    if netlist_lengths:
        print(f"\nNetlist token stats:")
        print(f"  Min: {min(netlist_lengths)}")
        print(f"  Max: {max(netlist_lengths)}")
        print(f"  Mean: {statistics.mean(netlist_lengths):.1f}")
        print(f"  Median: {statistics.median(netlist_lengths):.1f}")

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
