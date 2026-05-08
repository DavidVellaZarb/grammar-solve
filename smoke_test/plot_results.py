from __future__ import annotations

import json
import os
from pathlib import Path

import fire
import matplotlib.pyplot as plt

from smoke_test.metrics import clamp01, compute_domain_metrics, metric_label, metric_order

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


def _select_methods(methods: str | None = None) -> list[tuple[str, str, str]]:
    if not methods:
        return METHODS

    by_name = {method: (method, label, color) for method, label, color in METHODS}
    selected: list[tuple[str, str, str]] = []
    unknown: list[str] = []
    for method in methods.split(","):
        key = method.strip()
        if not key:
            continue
        if key not in by_name:
            unknown.append(key)
            continue
        selected.append(by_name[key])

    if unknown:
        raise ValueError(
            "Unknown methods: "
            + ", ".join(unknown)
            + ". Expected one or more of: "
            + ", ".join(by_name)
        )
    if not selected:
        raise ValueError("At least one method must be selected")
    return selected


def _load_accuracy(path: Path) -> float | None:
    if not path.exists():
        return None
    with path.open() as f:
        data = json.load(f)
    return float(data["accuracy"])


def _load_results(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def plot(
    model_alias: str = "qwen3-4b",
    results_root: str = "results/smoke_test",
    output_path: str | None = None,
    metric_set: str = "exact",
    summary_path: str | None = None,
    strict: bool = True,
    methods: str | None = None,
) -> None:
    selected_methods = _select_methods(methods)
    if metric_set not in {"exact", "all"}:
        raise ValueError("metric_set must be 'exact' or 'all'")
    if metric_set == "all":
        return plot_all_metrics(
            model_alias=model_alias,
            results_root=results_root,
            output_path=output_path,
            summary_path=summary_path,
            strict=strict,
            methods=methods,
        )

    results_dir = Path(results_root) / model_alias
    output_path = output_path or f"outputs/analysis/smoke_test/{model_alias}_baseline_vs_gold.png"

    missing: list[str] = []
    rows: list[tuple[str, list[float | None]]] = []
    for domain, label in DOMAINS:
        values = []
        for method, _, _ in selected_methods:
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
        x = range(len(selected_methods))
        heights = [v if v is not None else 0.0 for v in values]
        colors = [color for _, _, color in selected_methods]
        bars = ax.bar(x, heights, color=colors, width=0.62)
        ax.set_title(label, fontweight="bold")
        ax.set_xticks(list(x))
        ax.set_xticklabels(
            [method_label for _, method_label, _ in selected_methods],
            rotation=15,
            ha="right",
        )
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


def plot_all_metrics(
    model_alias: str = "qwen3-4b",
    results_root: str = "results/smoke_test",
    output_path: str | None = None,
    summary_path: str | None = None,
    strict: bool = True,
    methods: str | None = None,
) -> None:
    selected_methods = _select_methods(methods)
    results_dir = Path(results_root) / model_alias
    output_path = (
        output_path
        or f"outputs/analysis/smoke_test/{model_alias}_baseline_vs_gold_all_metrics.png"
    )
    summary_path = (
        summary_path
        or f"outputs/analysis/smoke_test/{model_alias}_baseline_vs_gold_all_metrics.json"
    )

    missing: list[str] = []
    rows: list[tuple[str, str, list[str], dict[str, dict[str, float] | None]]] = []
    metric_labels: dict[str, str] = {}
    domain_summaries: dict[str, dict[str, dict[str, float] | None]] = {}
    summary: dict[str, object] = {
        "model_alias": model_alias,
        "results_root": str(results_dir),
        "methods": [method for method, _, _ in selected_methods],
        "metric_labels": metric_labels,
        "domains": domain_summaries,
    }

    for domain, label in DOMAINS:
        domain_metrics = metric_order(domain)
        values_by_method: dict[str, dict[str, float] | None] = {}
        metric_labels.update({m: metric_label(m) for m in domain_metrics})
        for method, _, _ in selected_methods:
            path = results_dir / domain / f"{method}.json"
            data = _load_results(path)
            if data is None:
                missing.append(str(path))
                values_by_method[method] = None
                continue
            values_by_method[method] = compute_domain_metrics(domain, data.get("results", []))
        rows.append((domain, label, domain_metrics, values_by_method))
        domain_summaries[domain] = {
            method: values_by_method[method] for method, _, _ in selected_methods
        }

    if strict and missing:
        raise FileNotFoundError(
            "Missing smoke-test result files:\n" + "\n".join(missing)
        )

    n_cols = 4
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 9.5), sharey=True)
    axes_flat = list(axes.flatten())
    max_metrics = max((len(metrics) for _, _, metrics, _ in rows), default=1)

    for ax, (_, label, metrics, values_by_method) in zip(axes_flat, rows):
        offset = (max_metrics - len(metrics)) / 2
        x = [offset + i for i in range(len(metrics))]
        bar_width = 0.34
        offsets = [
            (i - (len(selected_methods) - 1) / 2) * bar_width
            for i in range(len(selected_methods))
        ]
        for i, (method, method_label, color) in enumerate(selected_methods):
            method_values = values_by_method.get(method)
            heights = [
                clamp01(float(method_values[m])) if method_values is not None else 0.0
                for m in metrics
            ]
            bars = ax.bar(
                [pos + offsets[i] for pos in x],
                heights,
                width=bar_width,
                color=color,
                label=method_label,
            )
            for bar, value in zip(bars, heights, strict=True):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    min(value + 0.025, 0.98),
                    "missing" if method_values is None else f"{value:.1%}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=0,
                )

        ax.set_title(label, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([metric_label(m) for m in metrics], rotation=22, ha="right")
        ax.set_xlim(-0.5, max_metrics - 0.5)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.25)

    for ax in axes_flat[len(rows):]:
        ax.set_visible(False)

    axes_flat[0].set_ylabel("Score")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.985, 0.98))
    fig.suptitle(f"Smoke Test Results ({model_alias}) - All Metrics", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved plot to {output_path}")
    print(f"Saved metrics summary to {summary_path}")


if __name__ == "__main__":
    fire.Fire(plot)
