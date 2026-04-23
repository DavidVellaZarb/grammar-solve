import json
import os
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np

from bootstrap import bootstrap_ci

DATASETS = [
    {
        "name": "VerilogEval",
        "dir": "verilog",
        "pass_at_k": True,
    },
    {
        "name": "SMCalFlow",
        "dir": "smcalflow",
        "metrics": ["accuracy"],
        "metric_labels": {"accuracy": "Exact Match"},
        "per_example_fields": {"accuracy": "match"},
    },
    {
        "name": "GeoQuery",
        "dir": "geoquery",
        "metrics": ["exact_match", "execution_accuracy"],
        "metric_labels": {"exact_match": "Exact Match", "execution_accuracy": "Exec. Accuracy"},
        "per_example_fields": {"exact_match": "exact_match", "execution_accuracy": "execution_match"},
    },
    {
        "name": "Overnight",
        "dir": "overnight",
        "metrics": ["exact_match", "execution_accuracy"],
        "metric_labels": {"exact_match": "Exact Match", "execution_accuracy": "Exec. Accuracy"},
        "per_example_fields": {"exact_match": "exact_match", "execution_accuracy": "execution_match"},
    },
    {
        "name": "SPICE",
        "dir": "spice",
        "metrics": ["ged_similarity", "component_f1"],
        "metric_labels": {"ged_similarity": "GED Similarity", "component_f1": "Component F1"},
        "per_example_fields": {"ged_similarity": "ged_similarity", "component_f1": "component_f1"},
    },
    {
        "name": "SMILES",
        "dir": "smiles",
        "metrics": ["fingerprint_similarity", "validity", "canonical_exact_match"],
        "metric_labels": {"fingerprint_similarity": "FP Similarity", "validity": "Validity", "canonical_exact_match": "Exact Match"},
        "per_example_fields": {"fingerprint_similarity": "fingerprint_similarity", "validity": "valid", "canonical_exact_match": "canonical_match"},
    },
    {
        "name": "OpenSCAD",
        "dir": "openscad",
        "metrics": ["iou", "syntax_validity"],
        "metric_labels": {"iou": "Volumetric IoU", "syntax_validity": "Syntax Validity"},
        "per_example_fields": {"iou": "iou", "syntax_validity": "valid"},
    },
]

METHODS = [
    ("baseline", "Baseline"),
    ("rag", "Ours (RAG)"),
    ("gold", "Gold Grammar"),
]

COLORS = ["#4C72B0", "#DD8452", "#55A868"]


def _load_if_exists(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def plot_model_panel(
    model_alias: str,
    results_dir: str = "results",
    output_path: str | None = None,
    n_bootstrap: int = 1000,
):
    output_path = output_path or f"outputs/analysis/{model_alias}_panel.png"

    def _n_groups(ds):
        return 3 if ds.get("pass_at_k") else len(ds["metrics"])

    def _has_data(ds):
        ds_dir = f"{results_dir}/{ds['dir']}/{model_alias}"
        return any(os.path.exists(f"{ds_dir}/{m}.json") for m, _ in METHODS)

    datasets = [ds for ds in DATASETS if _has_data(ds)]
    n = len(datasets)
    widths = [_n_groups(ds) for ds in datasets]

    if n <= 1:
        split = n
    else:
        total = sum(widths)
        best_k, best_score = None, None
        for k in range(1, n):
            diff = abs(sum(widths[:k]) - (total - sum(widths[:k])))
            tie = abs(k - (n + 1) // 2)
            score = (diff, tie)
            if best_score is None or score < best_score:
                best_k, best_score = k, score
        split = best_k

    row1 = datasets[:split]
    row2 = datasets[split:]
    widths1 = widths[:split]
    widths2 = widths[split:]

    def _pad(ws, target):
        gap = target - sum(ws)
        if gap <= 0:
            return ws, 0
        left = gap / 2
        right = gap - left
        return [left] + ws + [right], 1

    target_w = max(sum(widths1), sum(widths2)) if widths2 else sum(widths1)
    widths1_full, col_offset1 = _pad(widths1, target_w)
    widths2_full, col_offset2 = _pad(widths2, target_w) if widths2 else ([], 0)

    n_rows = 2 if widths2 else 1
    fig_h = 10 if n_rows == 2 else 5.5
    fig = plt.figure(figsize=(22, fig_h))
    gs = fig.add_gridspec(n_rows, len(widths1_full), width_ratios=widths1_full, hspace=0.35, wspace=0.3)
    gs2 = fig.add_gridspec(n_rows, len(widths2_full), width_ratios=widths2_full, hspace=0.35, wspace=0.3) if widths2 else None

    axes = []
    for j in range(len(widths1)):
        axes.append(fig.add_subplot(gs[0, j + col_offset1]))
    for j in range(len(widths2)):
        axes.append(fig.add_subplot(gs2[1, j + col_offset2]))

    bar_width = 0.22

    for idx, ds in enumerate(datasets):
        ax = axes[idx]
        ds_dir = f"{results_dir}/{ds['dir']}/{model_alias}"

        methods_present = []
        method_data = []
        method_colors = []
        for method_key, method_label in METHODS:
            data = _load_if_exists(f"{ds_dir}/{method_key}.json")
            if data is not None:
                methods_present.append(method_label)
                method_data.append(data)
                method_colors.append(COLORS[len(methods_present) - 1])

        if not methods_present:
            ax.set_title(f"{ds['name']} (no data)")
            ax.set_visible(False)
            continue

        if ds.get("pass_at_k"):
            k_values = []
            for data in method_data:
                for k in data:
                    if k.startswith("pass@") and k not in k_values:
                        k_values.append(k)
            k_values.sort(key=lambda x: int(x.split("@")[1]))

            num_k = len(k_values)
            num_methods = len(methods_present)

            for i, (data, label, color) in enumerate(zip(method_data, methods_present, method_colors)):
                values = [data.get(k, 0.0) for k in k_values]
                x_pos = [j + i * bar_width - (num_methods - 1) * bar_width / 2 for j in range(num_k)]
                bars = ax.bar(x_pos, values, bar_width, label=label, color=color)

            ax.set_xticks(range(num_k))
            ax.set_xticklabels(k_values)
            ax.set_ylabel("Pass Rate")
        else:
            metrics = ds["metrics"]
            metric_labels = ds["metric_labels"]
            per_example_fields = ds["per_example_fields"]
            num_metrics = len(metrics)
            num_methods = len(methods_present)

            for i, (data, label, color) in enumerate(zip(method_data, methods_present, method_colors)):
                means = []
                errs_low = []
                errs_high = []
                for m in metrics:
                    field = per_example_fields[m]
                    if "results" in data:
                        per_ex = [r[field] for r in data["results"] if r.get(field) is not None]
                        stats = bootstrap_ci(per_ex, n_bootstrap=n_bootstrap)
                        means.append(stats["mean"])
                        errs_low.append(stats["mean"] - stats["ci_low"])
                        errs_high.append(stats["ci_high"] - stats["mean"])
                    else:
                        means.append(data.get(m, 0.0))
                        errs_low.append(0)
                        errs_high.append(0)

                x_pos = [j + i * bar_width - (num_methods - 1) * bar_width / 2 for j in range(num_metrics)]
                bars = ax.bar(x_pos, means, bar_width, yerr=[errs_low, errs_high],
                              capsize=2, label=label, color=color)

            ax.set_xticks(range(num_metrics))
            ax.set_xticklabels([metric_labels.get(m, m) for m in metrics])
            ax.set_ylabel("Score")

        ax.set_ylim(0, 1.0)
        ax.set_title(ds["name"], fontweight="bold")
        is_row_start = idx == 0 or idx == split
        if not is_row_start:
            ax.set_ylabel("")
            ax.set_yticklabels([])
        if idx == 0:
            ax.legend(fontsize=8, loc="upper right")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def plot_all(results_dir: str = "results", output_dir: str = "outputs/analysis"):
    for model in ["qwen2.5-7b", "llama-3.1-8b"]:
        plot_model_panel(model, results_dir=results_dir,
                         output_path=f"{output_dir}/{model}_panel.png")


if __name__ == "__main__":
    fire.Fire({"panel": plot_model_panel, "all": plot_all})
