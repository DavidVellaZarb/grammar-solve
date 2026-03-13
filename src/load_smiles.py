import json
import os

import fire
from datasets import load_dataset
from tqdm import tqdm

from lark import Lark

from grammar_parser import _walk_tree

SKIP_RULES = {"smiles"}

GRAMMAR_PATH = "grammars/smiles.lark"


def _extract_grammar(smiles_str: str, parser) -> str | None:
    try:
        tree = parser.parse(smiles_str)
    except Exception:
        return None

    rules: dict[str, list[str]] = {}
    _walk_tree(tree, rules)
    rules = {k: v for k, v in rules.items() if k not in SKIP_RULES}
    lines = [f"{name} ::= {' | '.join(alts)}" for name, alts in rules.items()]
    return "\n".join(lines)


def load(
    output_dir: str = "data/smiles",
    max_examples: int = 0,
) -> None:
    print("Loading ChEBI-20 dataset from HuggingFace...")
    ds = load_dataset("liupf/ChEBI-20-MM")
    print(f"Splits: {list(ds.keys())}")

    print("Building SMILES parser...")
    with open(GRAMMAR_PATH) as f:
        grammar_text = f.read()
    parser = Lark(grammar_text, start="smiles", parser="lalr", keep_all_tokens=True)

    os.makedirs(output_dir, exist_ok=True)

    split_map = {"train": "train", "validation": "valid", "test": "test"}

    for split_name, out_name in split_map.items():
        if split_name not in ds:
            print(f"Warning: split '{split_name}' not found, skipping")
            continue

        data = ds[split_name]
        if max_examples > 0:
            data = data.select(range(min(max_examples, len(data))))
        print(f"\nProcessing {split_name} ({len(data)} examples)...")

        successes = []
        failures = []

        for i, example in tqdm(enumerate(data), total=len(data), desc=f"  Parsing {split_name}"):
            query = example["description"]
            smiles = example["SMILES"]

            grammar = _extract_grammar(smiles, parser)
            if grammar is not None:
                successes.append({
                    "query": query,
                    "minimal_grammar": grammar,
                    "program": smiles,
                })
            else:
                failures.append({
                    "index": i,
                    "description": query,
                    "SMILES": smiles,
                })

        print(f"  Parse results: {len(successes)}/{len(data)} succeeded "
              f"({100 * len(successes) / len(data):.1f}%)")

        out_path = os.path.join(output_dir, f"{out_name}.json")
        with open(out_path, "w") as f:
            json.dump({"data": successes}, f, indent=2)
        print(f"  Wrote {len(successes)} entries to {out_path}")

        if failures:
            fail_path = os.path.join(output_dir, f"parse_failures_{out_name}.json")
            with open(fail_path, "w") as f:
                json.dump(failures, f, indent=2)
            print(f"  Wrote {len(failures)} failures to {fail_path}")

    print("\nDone!")


if __name__ == "__main__":
    fire.Fire({"load": load})
