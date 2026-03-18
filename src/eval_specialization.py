import json
import os

import fire

from data import load_raw_data
from grammar_utils import parse_minimal_grammar
from specialize_grammar import extract_generic_rules, has_generic_terminals


def evaluate(
    predicted_path: str = "outputs/predicted_grammars/specialized.json",
    gold_path: str = "data/smcalflow/test.json",
    generic_path: str = "data/smcalflow/test_generic.json",
    output_path: str | None = "results/specialization",
):
    predicted_data = load_raw_data(predicted_path)
    gold_data = load_raw_data(gold_path)
    generic_data = load_raw_data(generic_path)

    assert len(predicted_data) == len(gold_data) == len(generic_data), (
        f"Data length mismatch: predicted={len(predicted_data)}, "
        f"gold={len(gold_data)}, generic={len(generic_data)}"
    )

    total = 0
    correct = 0
    string_total = 0
    string_correct = 0
    number_total = 0
    number_correct = 0
    details = []

    n_skipped = 0
    for i, (pred_ex, gold_ex, gen_ex) in enumerate(
        zip(predicted_data, gold_data, generic_data)
    ):
        if pred_ex["minimal_grammar"] is None:
            n_skipped += 1
            continue

        generic_grammar = gen_ex["minimal_grammar"]
        if not has_generic_terminals(generic_grammar):
            continue

        generic_rules = extract_generic_rules(generic_grammar)
        pred_parsed = parse_minimal_grammar(pred_ex["minimal_grammar"])
        gold_parsed = parse_minimal_grammar(gold_ex["minimal_grammar"])

        all_correct = True
        rule_results = {}

        for rule_name in generic_rules:
            pred_alts = set(pred_parsed.get(rule_name, []))
            gold_alts = set(gold_parsed.get(rule_name, []))
            match = pred_alts == gold_alts

            rule_results[rule_name] = {
                "predicted": sorted(pred_alts),
                "gold": sorted(gold_alts),
                "correct": match,
            }

            if not match:
                all_correct = False

            if rule_name == "string":
                string_total += 1
                if match:
                    string_correct += 1
            elif rule_name == "number":
                number_total += 1
                if match:
                    number_correct += 1

        total += 1
        if all_correct:
            correct += 1

        details.append(
            {
                "index": i,
                "query": gen_ex["query"],
                "gold_grammar": gold_ex["minimal_grammar"],
                "correct": all_correct,
                "rules": rule_results,
            }
        )

    if n_skipped:
        print(f"WARNING: Skipped {n_skipped} examples with missing grammar predictions")
    overall_acc = correct / total if total else 0
    string_acc = string_correct / string_total if string_total else 0
    number_acc = number_correct / number_total if number_total else 0

    print(f"Overall accuracy: {correct}/{total} = {overall_acc:.1%}")
    if string_total > 0:
        print(f"String accuracy:  {string_correct}/{string_total} = {string_acc:.1%}")
    if number_total > 0:
        print(f"Number accuracy:  {number_correct}/{number_total} = {number_acc:.1%}")

    if output_path:
        os.makedirs(output_path, exist_ok=True)

        accuracy = {
            "overall": overall_acc,
            "string": string_acc,
            "number": number_acc,
            "overall_correct": correct,
            "overall_total": total,
            "string_correct": string_correct,
            "string_total": string_total,
            "number_correct": number_correct,
            "number_total": number_total,
        }
        with open(os.path.join(output_path, "accuracy.json"), "w") as f:
            json.dump(accuracy, f, indent=2)

        with open(os.path.join(output_path, "details.json"), "w") as f:
            json.dump(details, f, indent=2)

        print(f"\nResults saved to {output_path}/")


if __name__ == "__main__":
    fire.Fire(evaluate)
