import json
import os
from pathlib import Path

import fire
import matplotlib.pyplot as plt


def plot_accuracies(
    results_dir: str = "results",
    models: list[str] | None = None,
    output_path: str = "results/results.png",
    title: str | None = None,
    test_labels: dict[str, str] | None = None,
    model_labels: dict[str, str] | None = None,
):
    """Plot evaluation accuracies as a grouped bar chart.

    Args:
        results_dir: Parent directory containing model subdirectories.
        models: Subdirectory names to include. None means all subdirs.
        output_path: Where to save the figure.
        title: Chart title. Auto-generated if None.
        test_labels: Mapping from JSON filename stem to display label.
        model_labels: Mapping from subdirectory name to display label.
    """
    results_path = Path(results_dir)
    test_labels = test_labels or {}
    model_labels = model_labels or {}

    if models:
        model_dirs = [results_path / m for m in models]
    else:
        model_dirs = sorted(
            [d for d in results_path.iterdir() if d.is_dir()],
            key=lambda d: d.name,
        )

    if not model_dirs:
        raise FileNotFoundError(f"No model directories found in {results_dir}")

    data = {}
    all_test_names = []

    for model_dir in model_dirs:
        model_name = model_dir.name
        data[model_name] = {}
        for result_file in sorted(model_dir.glob("*.json")):
            with open(result_file) as f:
                result = json.load(f)
            test_name = result_file.stem
            data[model_name][test_name] = result["accuracy"]
            if test_name not in all_test_names:
                all_test_names.append(test_name)

    if not all_test_names:
        raise FileNotFoundError("No result files found")

    num_tests = len(all_test_names)
    num_models = len(model_dirs)
    bar_width = 0.8 / num_models if num_models > 1 else 0.5

    fig, ax = plt.subplots(figsize=(max(6, num_tests * 2), 5))

    for i, model_dir in enumerate(model_dirs):
        model_name = model_dir.name
        display_name = model_labels.get(model_name, model_name)
        accuracies = [data[model_name].get(t, 0.0) for t in all_test_names]

        if num_models > 1:
            x_positions = [j + i * bar_width - (num_models - 1) * bar_width / 2 for j in range(num_tests)]
        else:
            x_positions = list(range(num_tests))

        bars = ax.bar(x_positions, accuracies, bar_width, label=display_name)

        for bar, acc in zip(bars, accuracies):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{acc:.2%}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    x_labels = [test_labels.get(t) or t for t in all_test_names]
    ax.set_xticks(range(num_tests))
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.15)

    if title:
        ax.set_title(title)
    elif num_models == 1:
        ax.set_title(f"Results — {model_labels.get(model_dirs[0].name, model_dirs[0].name)}")
    else:
        ax.set_title("Evaluation Results")

    if num_models > 1:
        ax.legend()

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved plot to {output_path}")


def plot_pass_at_k(
    result_files: list[str],
    labels: list[str] | None = None,
    output_path: str = "results/pass_at_k.png",
    title: str | None = None,
):
    """Plot pass@k metrics from multiple result JSON files as a grouped bar chart.

    Each result file should contain keys like "pass@1", "pass@3", "pass@5".

    Args:
        result_files: Paths to result JSON files.
        labels: Display labels for each file. Uses filename stems if None.
        output_path: Where to save the figure.
        title: Chart title.
    """
    labels = labels or [Path(f).stem for f in result_files]

    all_results = []
    all_k_values = []
    for path in result_files:
        with open(path) as f:
            data = json.load(f)
        metrics = {k: v for k, v in data.items() if k.startswith("pass@")}
        all_results.append(metrics)
        for k in metrics:
            if k not in all_k_values:
                all_k_values.append(k)

    all_k_values.sort(key=lambda x: int(x.split("@")[1]))

    num_k = len(all_k_values)
    num_methods = len(result_files)
    bar_width = 0.8 / num_methods if num_methods > 1 else 0.5

    fig, ax = plt.subplots(figsize=(max(6, num_k * 2.5), 5))

    for i, (metrics, label) in enumerate(zip(all_results, labels)):
        values = [metrics.get(k, 0.0) for k in all_k_values]
        if num_methods > 1:
            x_positions = [
                j + i * bar_width - (num_methods - 1) * bar_width / 2
                for j in range(num_k)
            ]
        else:
            x_positions = list(range(num_k))

        bars = ax.bar(x_positions, values, bar_width, label=label)

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.1%}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xticks(range(num_k))
    ax.set_xticklabels(all_k_values)
    ax.set_ylabel("Pass Rate")
    ax.set_ylim(0, min(1.15, max(
        v for m in all_results for v in m.values()
    ) * 1.3 + 0.05))

    if title:
        ax.set_title(title)
    else:
        ax.set_title("Functional Correctness (pass@k)")

    if num_methods > 1:
        ax.legend()

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    fire.Fire({"plot_accuracies": plot_accuracies, "plot_pass_at_k": plot_pass_at_k})
