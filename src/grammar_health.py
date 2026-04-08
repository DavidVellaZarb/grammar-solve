import json
import os
import re

import fire
import matplotlib.pyplot as plt
import numpy as np

from grammar_utils import extract_grammar_from_output, parse_minimal_grammar


def _find_referenced_rules(alternatives: list[str]) -> set[str]:
    refs = set()
    for alt in alternatives:
        tokens = re.findall(r'(?:^|(?<=\s))([a-z_][a-z0-9_]*)(?=\s|$)', alt)
        refs.update(tokens)
    return refs


def _analyze_grammar(grammar_text: str) -> dict:
    extracted = extract_grammar_from_output(grammar_text)
    rules = parse_minimal_grammar(extracted)
    if not rules:
        return {"incomplete": [], "unreachable": []}

    defined = set(rules.keys())
    start_rule = next(iter(rules))

    all_referenced = set()
    for alts in rules.values():
        all_referenced.update(_find_referenced_rules(alts))

    incomplete = sorted(all_referenced - defined)
    unreachable = sorted(defined - all_referenced - {start_rule})

    return {"incomplete": incomplete, "unreachable": unreachable}


def analyze(input_path: str, output_path: str):
    with open(input_path) as f:
        data = json.load(f)["data"]

    results = []
    only_incomplete = 0
    only_unreachable = 0
    both_count = 0
    neither_count = 0

    for ex in data:
        grammar = ex.get("minimal_grammar") or ""
        analysis = _analyze_grammar(grammar)
        has_incomplete = len(analysis["incomplete"]) > 0
        has_unreachable = len(analysis["unreachable"]) > 0
        if has_incomplete and has_unreachable:
            both_count += 1
        elif has_incomplete:
            only_incomplete += 1
        elif has_unreachable:
            only_unreachable += 1
        else:
            neither_count += 1
        results.append({
            "query": ex["query"],
            "incomplete_rules": analysis["incomplete"],
            "unreachable_rules": analysis["unreachable"],
        })

    total = len(data)

    def pct(n):
        return n / total * 100 if total else 0

    output = {
        "total": total,
        "only_incomplete": {"count": only_incomplete, "percentage": pct(only_incomplete)},
        "only_unreachable": {"count": only_unreachable, "percentage": pct(only_unreachable)},
        "both": {"count": both_count, "percentage": pct(both_count)},
        "neither": {"count": neither_count, "percentage": pct(neither_count)},
        "examples": results,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(
        f"{input_path}: only_incomplete={pct(only_incomplete):.1f}%, "
        f"only_unreachable={pct(only_unreachable):.1f}%, "
        f"both={pct(both_count):.1f}%, "
        f"neither={pct(neither_count):.1f}%"
    )


def plot(result_files: list[str], labels: list[str], output_path: str = "outputs/analysis/grammar_health/plot.png"):
    only_incomplete = []
    only_unreachable = []
    both_vals = []
    neither_vals = []
    for path in result_files:
        with open(path) as f:
            data = json.load(f)
        only_incomplete.append(data["only_incomplete"]["percentage"])
        only_unreachable.append(data["only_unreachable"]["percentage"])
        both_vals.append(data["both"]["percentage"])
        neither_vals.append(data["neither"]["percentage"])

    x = np.arange(len(labels))
    width = 0.5

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.8), 5))
    bottom = np.zeros(len(labels))

    for vals, label in [
        (neither_vals, "Neither"),
        (only_incomplete, "Only incomplete"),
        (only_unreachable, "Only unreachable"),
        (both_vals, "Both"),
    ]:
        bars = ax.bar(x, vals, width, bottom=bottom, label=label)
        for bar, val, b in zip(bars, vals, bottom):
            if val >= 5:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    b + val / 2,
                    f"{val:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                    fontweight="bold",
                )
        bottom = bottom + np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Grammars (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    fire.Fire()
