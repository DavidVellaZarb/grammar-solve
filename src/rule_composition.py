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
        "train_generic_path": "data/smcalflow/train_generic.json",
        "test_generic_path": "data/smcalflow/test_generic.json",
        "grammar_path": "grammars/smcalflow.lark",
        "start": "call",
        "skip_rules": None,
        "generic_terminals": GENERIC_TERMINALS,
    },
    "verilog": {
        "train_path": "data/mg_verilog/train_detailed.json",
        "test_path": "data/verilog_eval/VerilogEval_Human.jsonl",
        "train_generic_path": "data/mg_verilog/train_detailed_generic.json",
        "test_generic_path": "data/verilog_eval/VerilogEval_Human.jsonl",
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


def _load_data(path: str) -> list[dict]:
    if path.endswith(".jsonl"):
        with open(path) as f:
            return [json.loads(line) for line in f]
    with open(path) as f:
        return json.load(f)["data"]


def _load_rule_sets(
    data: list[dict], dataset: str, config: dict,
) -> list[set[str]]:
    has_grammar = "minimal_grammar" in data[0] and data[0]["minimal_grammar"]

    if has_grammar:
        rule_sets = []
        for entry in data:
            rules = parse_minimal_grammar(entry["minimal_grammar"])
            alts = set()
            for name, alternatives in rules.items():
                for alt in alternatives:
                    alts.add(f"{name} ::= {alt}")
            rule_sets.append(alts)
        return rule_sets

    programs = []
    for entry in data:
        if "canonical_solution" in entry:
            programs.append(entry["prompt"] + entry["canonical_solution"])
        elif dataset == "verilog" and "module_header" in entry:
            programs.append(entry["module_header"] + "\n" + entry["program"])
        else:
            programs.append(entry["program"])

    rule_sets = []
    failures = 0
    for program in tqdm(programs, desc="  Extracting grammars"):
        try:
            text = extract_minimal_grammar(
                program,
                grammar_path=config["grammar_path"],
                start=config["start"],
                skip_rules=config["skip_rules"],
                generic_terminals=config["generic_terminals"],
            )
            rules = parse_minimal_grammar(text)
            alts = set()
            for name, alternatives in rules.items():
                for alt in alternatives:
                    alts.add(f"{name} ::= {alt}")
            rule_sets.append(alts)
        except Exception:
            failures += 1
            rule_sets.append(set())

    if failures:
        print(f"  {failures}/{len(programs)} extraction failures")
    return rule_sets


def analyze_knn(
    dataset: str,
    rule_composition_path: str | None = None,
    train_path: str | None = None,
    test_path: str | None = None,
    model_name: str = "BAAI/bge-large-en-v1.5",
    k_values: list[int] | None = None,
    min_test_df: float = 0.05,
    max_train_df: float = 0.15,
    cache_dir: str = "cache/knn",
    output_path: str | None = None,
) -> None:
    from knn import _find_knn, _load_or_compute_embeddings
    from sentence_transformers import SentenceTransformer

    if k_values is None:
        k_values = [1, 4, 8, 16]

    if dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset}. Choose from {list(DATASET_CONFIGS.keys())}")

    config = DATASET_CONFIGS[dataset]

    if rule_composition_path is None:
        rule_composition_path = f"results/analysis/{dataset}_rule_composition.json"

    with open(rule_composition_path) as f:
        composition = json.load(f)

    rare_rules = [
        r for r in composition["rare_in_train_rules"]
        if r["test_df"] >= min_test_df and r["train_df"] < max_train_df
    ]
    print(f"Rules with test_df >= {min_test_df} and train_df < {max_train_df}: {len(rare_rules)}")
    for r in rare_rules:
        print(f"  {r['rule']}  (train_df={r['train_df']}, test_df={r['test_df']})")
    print()

    train_path = train_path or config.get("train_generic_path") or config["train_path"]
    test_path = test_path or config.get("test_generic_path") or config["test_path"]

    print(f"Loading data...")
    print(f"  Train: {train_path}")
    print(f"  Test:  {test_path}")
    train_data = _load_data(train_path)
    test_data = _load_data(test_path)
    print(f"  Train: {len(train_data)}, Test: {len(test_data)}")
    print()

    print("Building rule sets...")
    train_rule_sets = _load_rule_sets(train_data, dataset, config)
    test_rule_sets = _load_rule_sets(test_data, dataset, config)
    print()

    train_queries = [
        entry.get("query") or entry.get("prompt", "")
        for entry in train_data
    ]
    test_queries = [
        entry.get("query") or entry.get("prompt", "")
        for entry in test_data
    ]

    print(f"Computing embeddings (model: {model_name})...")
    model = SentenceTransformer(model_name)
    train_embeddings = _load_or_compute_embeddings(
        train_queries, model, cache_dir, model_name,
    )
    test_embeddings = _load_or_compute_embeddings(
        test_queries, model, cache_dir, model_name,
    )
    print()

    max_k = max(k_values)
    print(f"Finding {max_k}-NN...")
    knn_indices = _find_knn(test_embeddings, train_embeddings, max_k)
    print()

    print(f"Analyzing rule recovery...")
    results_rules = []
    for rule_info in rare_rules:
        rule = rule_info["rule"]
        test_indices = [i for i, rs in enumerate(test_rule_sets) if rule in rs]
        total = len(test_indices)

        per_k = {}
        for k in sorted(k_values):
            recovered = 0
            for i in test_indices:
                neighbors = knn_indices[i, :k]
                if any(rule in train_rule_sets[idx] for idx in neighbors):
                    recovered += 1
            per_k[str(k)] = {
                "recovered": recovered,
                "total": total,
                "recall": round(recovered / total, 4) if total > 0 else 0.0,
            }

        results_rules.append({
            "rule": rule,
            "train_df": rule_info["train_df"],
            "test_df": rule_info["test_df"],
            "test_count": total,
            "results": per_k,
        })

    output = {
        "config": {
            "dataset": dataset,
            "train_path": train_path,
            "test_path": test_path,
            "model_name": model_name,
            "k_values": k_values,
            "min_test_df": min_test_df,
            "max_train_df": max_train_df,
        },
        "summary": {
            "n_rules_analyzed": len(rare_rules),
            "n_train": len(train_data),
            "n_test": len(test_data),
        },
        "rules": results_rules,
    }

    if output_path is None:
        output_path = f"results/analysis/{dataset}_knn.json"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print()
    print(f"=== KNN Rule Recovery ({dataset}) ===")
    print(f"Rules analyzed: {len(rare_rules)}")
    print()
    for r in results_rules:
        print(f"  {r['rule']}")
        print(f"    train_df={r['train_df']}, test_df={r['test_df']}, test_count={r['test_count']}")
        for k_str, res in r["results"].items():
            print(f"    k={k_str}: {res['recovered']}/{res['total']} = {res['recall']:.1%}")
        print()

    print("  Aggregate recall (macro avg across rules):")
    for k in sorted(k_values):
        recalls = [r["results"][str(k)]["recall"] for r in results_rules]
        avg = sum(recalls) / len(recalls) if recalls else 0.0
        print(f"    k={k}: {avg:.1%}")
    print()

    print(f"Output written to {output_path}")


if __name__ == "__main__":
    fire.Fire({"analyze": analyze, "analyze_knn": analyze_knn})
