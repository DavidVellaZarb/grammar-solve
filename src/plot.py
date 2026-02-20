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


if __name__ == "__main__":
    fire.Fire(plot_accuracies)
