from __future__ import annotations

import json
import os
from pathlib import Path

import fire
import matplotlib.pyplot as plt

DOMAINS = [
    ("text_to_sql", "Text-to-SQL"),
    ("sparql", "SPARQL"),
    ("graphql", "GraphQL"),
    ("vega_lite", "Vega-Lite"),
    ("vhdl", "VHDL"),
    ("restricted_graphics", "Restricted Graphics"),
    ("selfies", "SELFIES"),
]

METHODS = [
    ("baseline", "Baseline", "#4C72B0"),
    ("gold", "Gold Grammar", "#55A868"),
]


def _load_accuracy(path: Path) -> float | None:
    if not path.exists():
        return None
    with path.open() as f:
        data = json.load(f)
    return float(data["accuracy"])


def plot(
    model_alias: str = "qwen3-4b",
    results_root: str = "results/smoke_test",
    output_path: str | None = None,
    strict: bool = True,
) -> None:
    results_dir = Path(results_root) / model_alias
    output_path = output_path or f"outputs/analysis/smoke_test/{model_alias}_baseline_vs_gold.png"

    missing: list[str] = []
    rows: list[tuple[str, list[float | None]]] = []
    for domain, label in DOMAINS:
        values = []
        for method, _, _ in METHODS:
            path = results_dir / domain / f"{method}.json"
            acc = _load_accuracy(path)
            if acc is None:
                missing.append(str(path))
            values.append(acc)
        rows.append((label, values))

    if strict and missing:
        raise FileNotFoundError(
            "Missing smoke-test result files:\n" + "\n".join(missing)
        )

    n = len(rows)
    n_cols = 4
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 8), sharey=True)
    axes_flat = list(axes.flatten())

    for ax, (label, values) in zip(axes_flat, rows):
        x = range(len(METHODS))
        heights = [v if v is not None else 0.0 for v in values]
        colors = [color for _, _, color in METHODS]
        bars = ax.bar(x, heights, color=colors, width=0.62)
        ax.set_title(label, fontweight="bold")
        ax.set_xticks(list(x))
        ax.set_xticklabels([method_label for _, method_label, _ in METHODS], rotation=15, ha="right")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values, strict=True):
            text = "missing" if value is None else f"{value:.1%}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(bar.get_height() + 0.025, 0.98),
                text,
                ha="center",
                va="bottom",
                fontsize=9,
            )

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    axes_flat[0].set_ylabel("Canonical exact match")
    fig.suptitle(f"Smoke Test Results ({model_alias})", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    fire.Fire(plot)
