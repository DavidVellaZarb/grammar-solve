import json

import fire

from data import load_raw_data
from grammar_utils import extract_grammar_from_output, parse_minimal_grammar


def compare_grammars(
    predicted_grammar: str, gold_grammar: str
) -> dict:
    pred = parse_minimal_grammar(predicted_grammar)
    gold = parse_minimal_grammar(gold_grammar)

    added_rules = []
    missing_rules = []

    all_names = set(pred) | set(gold)
    for name in sorted(all_names):
        pred_alts = set(pred.get(name, []))
        gold_alts = set(gold.get(name, []))

        for alt in sorted(pred_alts - gold_alts):
            added_rules.append(f"{name} ::= {alt}")
        for alt in sorted(gold_alts - pred_alts):
            missing_rules.append(f"{name} ::= {alt}")

    return {
        "exact_match": len(added_rules) == 0 and len(missing_rules) == 0,
        "added_rules": added_rules,
        "missing_rules": missing_rules,
    }


def evaluate(
    predicted_path: str,
    gold_path: str,
    write: bool = False,
):
    predicted_data = load_raw_data(predicted_path)
    gold_data = load_raw_data(gold_path)

    assert len(predicted_data) == len(gold_data), (
        f"Predicted has {len(predicted_data)} entries but gold has {len(gold_data)}"
    )

    exact_match_count = 0
    relaxed_match_count = 0
    only_added_count = 0
    only_missing_count = 0
    both_count = 0
    only_added_totals = []
    only_missing_totals = []
    both_added_totals = []
    both_missing_totals = []

    for pred_ex, gold_ex in zip(predicted_data, gold_data):
        assert pred_ex["query"] == gold_ex["query"], (
            f"Query mismatch: {pred_ex['query']!r} vs {gold_ex['query']!r}"
        )

        result = compare_grammars(
            extract_grammar_from_output(pred_ex["minimal_grammar"]),
            extract_grammar_from_output(gold_ex["minimal_grammar"]),
        )

        if write:
            pred_ex["added_rules"] = result["added_rules"]
            pred_ex["missing_rules"] = result["missing_rules"]
            pred_ex["exact_match"] = result["exact_match"]
            pred_ex["relaxed_match"] = (
                len(result["added_rules"]) <= 1 and len(result["missing_rules"]) <= 1
            )

        has_added = len(result["added_rules"]) > 0
        has_missing = len(result["missing_rules"]) > 0

        if len(result["added_rules"]) <= 1 and len(result["missing_rules"]) <= 1:
            relaxed_match_count += 1

        if not has_added and not has_missing:
            exact_match_count += 1
        elif has_added and not has_missing:
            only_added_count += 1
            only_added_totals.append(len(result["added_rules"]))
        elif not has_added and has_missing:
            only_missing_count += 1
            only_missing_totals.append(len(result["missing_rules"]))
        else:
            both_count += 1
            both_added_totals.append(len(result["added_rules"]))
            both_missing_totals.append(len(result["missing_rules"]))

    total = len(predicted_data)
    metrics = {
        "total": total,
        "exact_match": exact_match_count / total,
        "relaxed_match": relaxed_match_count / total,
        "only_added": {
            "proportion": only_added_count / total,
            "avg_added": (
                sum(only_added_totals) / len(only_added_totals)
                if only_added_totals
                else 0.0
            ),
        },
        "only_missing": {
            "proportion": only_missing_count / total,
            "avg_missing": (
                sum(only_missing_totals) / len(only_missing_totals)
                if only_missing_totals
                else 0.0
            ),
        },
        "both": {
            "proportion": both_count / total,
            "avg_added": (
                sum(both_added_totals) / len(both_added_totals)
                if both_added_totals
                else 0.0
            ),
            "avg_missing": (
                sum(both_missing_totals) / len(both_missing_totals)
                if both_missing_totals
                else 0.0
            ),
        },
    }

    print(f"Total examples: {total}")
    print(f"Exact match: {metrics['exact_match']:.4f} ({exact_match_count}/{total})")
    print(f"Relaxed match: {metrics['relaxed_match']:.4f} ({relaxed_match_count}/{total})")
    print(
        f"Only added: {metrics['only_added']['proportion']:.4f} "
        f"({only_added_count}/{total}), "
        f"avg added: {metrics['only_added']['avg_added']:.2f}"
    )
    print(
        f"Only missing: {metrics['only_missing']['proportion']:.4f} "
        f"({only_missing_count}/{total}), "
        f"avg missing: {metrics['only_missing']['avg_missing']:.2f}"
    )
    print(
        f"Both added+missing: {metrics['both']['proportion']:.4f} "
        f"({both_count}/{total}), "
        f"avg added: {metrics['both']['avg_added']:.2f}, "
        f"avg missing: {metrics['both']['avg_missing']:.2f}"
    )

    if write:
        with open(predicted_path, "w") as f:
            json.dump({"metrics": metrics, "data": predicted_data}, f, indent=2)
        print(f"\nAnnotated results written to {predicted_path}")


if __name__ == "__main__":
    fire.Fire(evaluate)
