import json
import os
import urllib.request

import fire
from tqdm import tqdm

from lark import Lark

from grammar_parser import _walk_tree

SKIP_RULES = {"query"}

GRAMMAR_PATH = "grammars/geoquery.lark"

BASE_URL = "https://raw.githubusercontent.com/berlino/grammar-prompting/main/data/geoquery/iid_split"


def _download_file(url: str, path: str) -> None:
    if os.path.exists(path):
        return
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, path)


def _extract_grammar(program: str, parser) -> str | None:
    try:
        tree = parser.parse(program)
    except Exception:
        return None

    rules: dict[str, list[str]] = {}
    _walk_tree(tree, rules)
    rules = {k: v for k, v in rules.items() if k not in SKIP_RULES}
    lines = [f"{name} ::= {' | '.join(alts)}" for name, alts in rules.items()]
    return "\n".join(lines)


def load(
    output_dir: str = "data/geoquery",
    max_examples: int = 0,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Download data files
    raw_dir = os.path.join(output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    for fname in ("train.src", "train.tgt", "dev.src", "dev.tgt", "test.src", "test.tgt"):
        _download_file(f"{BASE_URL}/{fname}", os.path.join(raw_dir, fname))

    print("Building GeoQuery parser...")
    with open(GRAMMAR_PATH) as f:
        grammar_text = f.read()
    parser = Lark(grammar_text, start="query", parser="earley", keep_all_tokens=True)

    split_map = {"train": "train", "dev": "valid", "test": "test"}

    for split_name, out_name in split_map.items():
        src_path = os.path.join(raw_dir, f"{split_name}.src")
        tgt_path = os.path.join(raw_dir, f"{split_name}.tgt")

        with open(src_path) as f:
            queries = [line.strip() for line in f if line.strip()]
        with open(tgt_path) as f:
            programs = [line.strip() for line in f if line.strip()]

        assert len(queries) == len(programs), (
            f"Mismatch: {len(queries)} queries vs {len(programs)} programs in {split_name}"
        )

        if max_examples > 0:
            queries = queries[:max_examples]
            programs = programs[:max_examples]

        print(f"\nProcessing {split_name} ({len(queries)} examples)...")

        successes = []
        failures = []

        for i, (query, program) in tqdm(
            enumerate(zip(queries, programs)), total=len(queries), desc=f"  Parsing {split_name}"
        ):
            grammar = _extract_grammar(program, parser)
            if grammar is not None:
                successes.append({
                    "query": query,
                    "minimal_grammar": grammar,
                    "program": program,
                })
            else:
                failures.append({
                    "index": i,
                    "query": query,
                    "program": program,
                })

        print(f"  Parse results: {len(successes)}/{len(queries)} succeeded "
              f"({100 * len(successes) / len(queries):.1f}%)")

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
