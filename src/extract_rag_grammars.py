import json
import os

import fire

from grammar_utils import extract_grammar_from_output


def extract(predicted_path: str, output_path: str) -> None:
    with open(predicted_path) as f:
        examples = json.load(f)["data"]

    out = []
    n_skipped = 0
    for ex in examples:
        grammar = ex.get("minimal_grammar")
        if grammar is None:
            n_skipped += 1
            continue
        out.append({**ex, "minimal_grammar": extract_grammar_from_output(grammar)})

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"data": out}, f, indent=2)
    print(f"Wrote {len(out)} examples to {output_path} ({n_skipped} skipped, no grammar)")


if __name__ == "__main__":
    fire.Fire({"extract": extract})
