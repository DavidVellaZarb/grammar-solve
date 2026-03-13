import json
import os
import re

import fire
from datasets import Dataset
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split

from grammar_parser import _build_parser, _detect_repetition_rules, _walk_tree
from grammar_utils import VERILOG_GENERIC_TERMINALS

DESCRIPTION_LEVELS = {
    "high_level": "high_level_global_summary",
    "detailed": "detailed_global_summary",
    "block": "block_summary",
}

SKIP_RULES = {"start", "module", "list_of_ports", "parameter_list", "port_item",
              "port_declaration", "port_dir"}

GRAMMAR_PATH = "grammars/verilog.lark"


def _parse_description(text: str) -> tuple[str, str]:
    inst_match = re.search(r"<</SYS>>\s*(.*?)\s*\[/INST\]", text, re.DOTALL)
    if not inst_match:
        return text.strip(), ""

    content = inst_match.group(1).strip()

    header_split = re.split(r"Module header:\s*", content, maxsplit=1)
    if len(header_split) == 2:
        desc_part = header_split[0].strip()
        module_header = header_split[1].strip()
    else:
        desc_part = content
        module_header = ""

    desc_part = re.sub(
        r"^Implement the Verilog module based on the following (?:description|block level summaries)\."
        r"\s*Assume that signals are positive clock/clk edge triggered unless otherwise stated\.\s*",
        "",
        desc_part,
        flags=re.DOTALL,
    ).strip()

    return desc_part, module_header


def _extract_grammar(
    full_module: str,
    parser,
    generic_terminals: frozenset[str] | None = None,
    normalize_repetition: bool = True,
) -> str | None:
    try:
        tree = parser.parse(full_module)
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
    output_dir: str = "data/mg_verilog",
    test_size: float = 0.1,
    valid_size: float = 0.1,
    seed: int = 42,
    max_examples: int = 0,
    generic: bool = False,
    normalize_repetition: bool = True,
) -> None:
    print("Loading MG-Verilog dataset from HuggingFace...")
    arrow_path = hf_hub_download(
        "GaTech-EIC/MG-Verilog",
        "merged_dataset/data-00000-of-00001.arrow",
        repo_type="dataset",
    )
    data = Dataset.from_file(arrow_path)
    if max_examples > 0:
        data = data.select(range(min(max_examples, len(data))))
    print(f"Loaded {len(data)} examples")

    print("Parsing descriptions...")
    codes = data["code"]
    descriptions = data["description"]

    examples = []
    for i, (code_raw, desc) in enumerate(zip(codes, descriptions)):
        code = code_raw.strip()

        parsed_descs = {}
        module_header = None
        for level_key, field_name in DESCRIPTION_LEVELS.items():
            nl_desc, header = _parse_description(desc[field_name])
            parsed_descs[level_key] = nl_desc
            if header and module_header is None:
                module_header = header

        if not module_header:
            if code.startswith("module"):
                module_header = code.split(";")[0] + ";"
            else:
                module_header = ""

        examples.append({
            "index": i,
            "code": code,
            "module_header": module_header,
            "descriptions": parsed_descs,
        })

    print("Building Verilog parser...")
    parser = _build_parser(GRAMMAR_PATH, start="module")

    generic_terminals = VERILOG_GENERIC_TERMINALS if generic else None
    print(f"Extracting minimal grammars{' (generic)' if generic else ''}...")
    successes = []
    failures = []
    for ex in examples:
        header = ex["module_header"]
        code = ex["code"]

        if header:
            if not code.rstrip().endswith("endmodule"):
                full_module = header + "\n" + code + "\nendmodule"
            else:
                full_module = header + "\n" + code
        else:
            full_module = code

        grammar = _extract_grammar(full_module, parser, generic_terminals=generic_terminals,
                                   normalize_repetition=normalize_repetition)
        if grammar is not None:
            ex["minimal_grammar"] = grammar
            successes.append(ex)
        else:
            failures.append(ex)

        total = len(successes) + len(failures)
        if total % 1000 == 0:
            print(f"  Processed {total}/{len(examples)} "
                  f"({len(successes)} ok, {len(failures)} failed)")

    print(f"\nParse results: {len(successes)}/{len(examples)} succeeded "
          f"({100*len(successes)/len(examples):.1f}%)")
    print(f"Failures: {len(failures)}")

    os.makedirs(output_dir, exist_ok=True)
    if failures:
        fail_path = os.path.join(output_dir, "parse_failures.json")
        fail_data = [{"index": f["index"], "module_header": f["module_header"],
                       "code": f["code"][:500]} for f in failures]
        with open(fail_path, "w") as f:
            json.dump(fail_data, f, indent=2)
        print(f"Wrote {len(failures)} failures to {fail_path}")

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

    for split_name, idxs in splits.items():
        split_examples = [successes[i] for i in idxs]

        for level_key in DESCRIPTION_LEVELS:
            out_data = []
            for ex in split_examples:
                out_data.append({
                    "query": ex["descriptions"][level_key],
                    "module_header": ex["module_header"],
                    "minimal_grammar": ex["minimal_grammar"],
                    "program": ex["code"],
                })

            suffix = "_generic" if generic else ""
            out_path = os.path.join(output_dir, f"{split_name}_{level_key}{suffix}.json")
            with open(out_path, "w") as f:
                json.dump({"data": out_data}, f, indent=2)
            print(f"Wrote {len(out_data)} entries to {out_path}")

    print("\nDone!")


if __name__ == "__main__":
    fire.Fire({"load": load})
