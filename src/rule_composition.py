import json
import math
import os
from collections import Counter

import fire
from tqdm import tqdm

from grammar_parser import extract_minimal_grammar
from grammar_utils import (
    GENERIC_TERMINALS,
    VERILOG_GENERIC_TERMINALS,
    parse_minimal_grammar,
)

VERILOG_SKIP_RULES = {
    "start", "module", "list_of_ports", "parameter_list", "port_item",
    "port_declaration", "port_dir",
}

DATASET_CONFIGS = {
    "smcalflow": {
        "train_path": "data/smcalflow/train.json",
        "test_path": "data/smcalflow/test.json",
        "grammar_path": "grammars/smcalflow.lark",
        "start": "call",
        "skip_rules": None,
        "generic_terminals": GENERIC_TERMINALS,
    },
    "verilog": {
        "train_path": "data/mg_verilog/train_detailed.json",
        "test_path": "data/verilog_eval/VerilogEval_Human.jsonl",
        "grammar_path": "grammars/verilog.lark",
        "start": "module",
        "skip_rules": VERILOG_SKIP_RULES,
        "generic_terminals": VERILOG_GENERIC_TERMINALS,
    },
}


def _load_programs(path: str, dataset: str) -> list[str]:
    if path.endswith(".jsonl"):
        programs = []
        with open(path) as f:
            for line in f:
                entry = json.loads(line)
                programs.append(entry["prompt"] + entry["canonical_solution"])
        return programs

    with open(path) as f:
        data = json.load(f)["data"]

    if dataset == "verilog":
        return [
            entry["module_header"] + "\n" + entry["program"]
            for entry in data
        ]
    else:
        return [entry["program"] for entry in data]


def _extract_rule_alternatives(
    programs: list[str],
    grammar_path: str,
    start: str,
    skip_rules: set[str] | None,
    generic_terminals: frozenset[str] | None,
    label: str,
) -> tuple[Counter, int, int]:
    counts: Counter = Counter()
    failures = 0

    for program in tqdm(programs, desc=f"  [{label}]"):
        try:
            text = extract_minimal_grammar(
                program,
                grammar_path=grammar_path,
                start=start,
                skip_rules=skip_rules,
                generic_terminals=generic_terminals,
            )
            rules = parse_minimal_grammar(text)
            alts = set()
            for name, alternatives in rules.items():
                for alt in alternatives:
                    alts.add(f"{name} ::= {alt}")
            counts.update(alts)
        except Exception:
            failures += 1

    print(f"  [{label}] Done: {len(programs)} programs, {failures} failures")
    return counts, len(programs), failures


def _compute_metrics(
    train_counts: Counter,
    test_counts: Counter,
    n_train: int,
    n_test: int,
    top_k: int,
) -> dict:
    all_rules = set(train_counts.keys()) | set(test_counts.keys())

    all_rules_data = []
    for rule in sorted(all_rules):
        tc = train_counts.get(rule, 0)
        tec = test_counts.get(rule, 0)
        train_df = tc / n_train if n_train > 0 else 0
        test_df = tec / n_test if n_test > 0 else 0
        ratio = test_df / (train_df + 1e-7)
        gap = test_df - train_df
        novelty = test_df * math.log(test_df / (train_df + 1e-7)) if test_df > 0 else 0.0

        all_rules_data.append({
            "rule": rule,
            "train_count": tc,
            "test_count": tec,
            "train_df": round(train_df, 6),
            "test_df": round(test_df, 6),
            "ratio": round(ratio, 4),
            "gap": round(gap, 6),
            "novelty": round(novelty, 6),
        })

    test_only = [r for r in all_rules_data if r["train_count"] == 0 and r["test_count"] > 0]
    test_only.sort(key=lambda r: r["test_count"], reverse=True)

    train_only = [r for r in all_rules_data if r["test_count"] == 0 and r["train_count"] > 0]
    train_only.sort(key=lambda r: r["train_count"], reverse=True)

    shared = [r for r in all_rules_data if r["train_count"] > 0 and r["test_count"] > 0]
    rare_in_train = sorted(shared, key=lambda r: r["novelty"], reverse=True)[:top_k]

    return {
        "test_only_rules": test_only,
        "train_only_rules": train_only,
        "rare_in_train_rules": rare_in_train,
        "all_rules": sorted(all_rules_data, key=lambda r: r["novelty"], reverse=True),
    }


def analyze(
    dataset: str,
    train_path: str | None = None,
    test_path: str | None = None,
    grammar_path: str | None = None,
    top_k: int = 20,
    output_path: str | None = None,
) -> None:
    if dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset}. Choose from {list(DATASET_CONFIGS.keys())}")

    config = DATASET_CONFIGS[dataset]
    train_path = train_path or config["train_path"]
    test_path = test_path or config["test_path"]
    grammar_path = grammar_path or config["grammar_path"]
    start = config["start"]
    skip_rules = config["skip_rules"]
    generic_terminals = config["generic_terminals"]

    if output_path is None:
        output_path = f"results/analysis/{dataset}_rule_composition.json"

    print(f"Rule composition analysis: {dataset}")
    print(f"  Train: {train_path}")
    print(f"  Test:  {test_path}")
    print(f"  Grammar: {grammar_path}")
    print()

    print("Loading programs...")
    train_programs = _load_programs(train_path, dataset)
    test_programs = _load_programs(test_path, dataset)
    print(f"  Train: {len(train_programs)}, Test: {len(test_programs)}")
    print()

    print("Extracting rule alternatives from train...")
    train_counts, n_train, train_failures = _extract_rule_alternatives(
        train_programs, grammar_path, start, skip_rules, generic_terminals, "train"
    )
    print()

    print("Extracting rule alternatives from test...")
    test_counts, n_test, test_failures = _extract_rule_alternatives(
        test_programs, grammar_path, start, skip_rules, generic_terminals, "test"
    )
    print()

    print("Computing metrics...")
    metrics = _compute_metrics(train_counts, test_counts, n_train, n_test, top_k)

    summary = {
        "n_train": n_train,
        "n_test": n_test,
        "train_failures": train_failures,
        "test_failures": test_failures,
        "total_unique_rules": len(metrics["all_rules"]),
        "test_only_count": len(metrics["test_only_rules"]),
        "train_only_count": len(metrics["train_only_rules"]),
        "shared_count": len(metrics["all_rules"]) - len(metrics["test_only_rules"]) - len(metrics["train_only_rules"]),
    }

    output = {
        "config": {
            "dataset": dataset,
            "train_path": train_path,
            "test_path": test_path,
            "grammar_path": grammar_path,
            "top_k": top_k,
        },
        "summary": summary,
        "test_only_rules": metrics["test_only_rules"],
        "train_only_rules": metrics["train_only_rules"],
        "rare_in_train_rules": metrics["rare_in_train_rules"],
        "all_rules": metrics["all_rules"],
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print()
    print(f"=== {dataset} Rule Composition Summary ===")
    print(f"Train programs: {n_train} ({train_failures} failures)")
    print(f"Test programs:  {n_test} ({test_failures} failures)")
    print(f"Total unique rule alternatives: {summary['total_unique_rules']}")
    print(f"Shared: {summary['shared_count']}, Test-only: {summary['test_only_count']}, Train-only: {summary['train_only_count']}")
    print()

    if metrics["test_only_rules"]:
        print(f"Top test-only rules (unseen in train):")
        for r in metrics["test_only_rules"][:10]:
            print(f"  {r['rule']}  (test_count={r['test_count']}, test_df={r['test_df']})")
        print()

    if metrics["train_only_rules"]:
        print(f"Top train-only rules (unseen in test):")
        for r in metrics["train_only_rules"][:10]:
            print(f"  {r['rule']}  (train_count={r['train_count']}, train_df={r['train_df']})")
        print()

    if metrics["rare_in_train_rules"]:
        print(f"Top {top_k} rare-in-train rules (by novelty):")
        for r in metrics["rare_in_train_rules"][:10]:
            print(f"  {r['rule']}  (train_df={r['train_df']}, test_df={r['test_df']}, novelty={r['novelty']})")
        print()

    print(f"Output written to {output_path}")


if __name__ == "__main__":
    fire.Fire({"analyze": analyze})
