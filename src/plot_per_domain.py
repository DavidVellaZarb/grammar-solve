"""Per-domain multi-panel diagrams for the paper.

Renders two multi-panel figures comparing three model sizes (Qwen3.5-4B,
Qwen3.5-9B, Gemma-4-31B) across our three configurations (no-grammar baseline,
RAG-predicted grammars, gold grammars), faceted by domain.

- semantic_parsing.pdf: SMCalFlow, GeoQuery, Overnight
- hdl.pdf: Verilog, SPICE
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np

from bootstrap import bootstrap_ci


MODELS = [
    ("qwen3-5-4b", "Qwen-4B"),
    ("qwen3-5-9b", "Qwen-9B"),
    ("gemma-4-31b", "Gemma-31B"),
]

CONFIGS = [
    ("baseline", "Baseline"),
    ("rag", "Ours (RAG)"),
    ("gold", "Gold Grammar"),
]

COLORS = ["#4C72B0", "#DD8452", "#55A868"]

PER_EXAMPLE_FIELD = {
    "smcalflow": "match",
    "geoquery": "execution_match",
    "overnight": "execution_match",
    "spice": "ged_similarity",
}

SEMANTIC_PANELS = [
    ("smcalflow", "SMCalFlow"),
    ("geoquery", "GeoQuery"),
    ("overnight", "Overnight"),
]

HDL_PANELS = [
    ("verilog", "Verilog"),
    ("spice", "SPICE"),
]


def _per_example_from_json(path: str, field: str) -> list[float]:
    with open(path) as f:
        data = json.load(f)
    out: list[float] = []
    for r in data.get("results", []):
        v = r.get(field)
        if v is None:
            continue
        out.append(float(v))
    return out


def _per_problem_pass1_from_verilog(samples_results_path: str) -> list[float]:
    by_task: dict[str, list[bool]] = defaultdict(list)
    with open(samples_results_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            by_task[obj["task_id"]].append(bool(obj.get("passed", False)))
    return [sum(passes) / len(passes) for passes in by_task.values() if passes]


def _collect_values(domain: str, model: str, config: str, results_dir: str) -> list[float]:
    base = Path(results_dir) / domain / model
    if domain == "verilog":
        path = base / f"{config}_samples.jsonl_results.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        return _per_problem_pass1_from_verilog(str(path))

    path = base / f"{config}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return _per_example_from_json(str(path), PER_EXAMPLE_FIELD[domain])


def _plot_panels(
    panels: list[tuple[str, str]],
    y_label,  # str -> shared y-axis; list[str] -> per-panel y-axis
    output_path: str,
    results_dir: str,
    n_bootstrap: int,
    figsize: tuple[float, float],
) -> None:
    per_panel = isinstance(y_label, (list, tuple))
    n = len(panels)
    fig, axes = plt.subplots(
        1, n,
        sharey=not per_panel,
        figsize=figsize,
        gridspec_kw={"wspace": 0.35 if per_panel else 0.15},
    )
    if n == 1:
        axes = [axes]

    bar_width = 0.25
    offsets = np.array([-bar_width, 0.0, bar_width])
    x_positions = np.arange(len(MODELS), dtype=float)

    for ax_idx, (domain_dir, display_name) in enumerate(panels):
        ax = axes[ax_idx]

        for cfg_idx, (config_key, config_label) in enumerate(CONFIGS):
            means: list[float] = []
            errs_low: list[float] = []
            errs_high: list[float] = []
            for model_key, _ in MODELS:
                values = _collect_values(domain_dir, model_key, config_key, results_dir)
                stats = bootstrap_ci(values, n_bootstrap=n_bootstrap)
                means.append(stats["mean"])
                errs_low.append(max(0.0, stats["mean"] - stats["ci_low"]))
                errs_high.append(max(0.0, stats["ci_high"] - stats["mean"]))

            xs = x_positions + offsets[cfg_idx]
            ax.bar(
                xs,
                means,
                bar_width,
                yerr=[errs_low, errs_high],
                capsize=2,
                color=COLORS[cfg_idx],
                label=config_label,
                error_kw=dict(elinewidth=0.8),
            )

        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            [m[1] for m in MODELS],
            fontsize=7,
            rotation=15,
            ha="right",
            rotation_mode="anchor",
        )
        ax.set_xlabel(display_name, fontsize=9)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlim(-0.5, len(MODELS) - 0.5)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if per_panel:
            ax.set_ylabel(y_label[ax_idx], fontsize=9)

    if not per_panel:
        axes[0].set_ylabel(y_label, fontsize=9)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=len(CONFIGS),
        frameon=False,
        fontsize=9,
        columnspacing=2.0,
        handlelength=1.5,
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", format="pdf")
    plt.close()
    print(f"Saved {output_path}")


def semantic_parsing(
    results_dir: str = "results",
    output_dir: str = "outputs/analysis/per-domain",
    n_bootstrap: int = 1000,
) -> None:
    _plot_panels(
        SEMANTIC_PANELS,
        y_label="Accuracy",
        output_path=f"{output_dir}/semantic_parsing.pdf",
        results_dir=results_dir,
        n_bootstrap=n_bootstrap,
        figsize=(6.3, 2.5),
    )


def hdl(
    results_dir: str = "results",
    output_dir: str = "outputs/analysis/per-domain",
    n_bootstrap: int = 1000,
) -> None:
    _plot_panels(
        HDL_PANELS,
        y_label=["pass@1", "GED Similarity"],
        output_path=f"{output_dir}/hdl.pdf",
        results_dir=results_dir,
        n_bootstrap=n_bootstrap,
        figsize=(6.3, 2.5),
    )


def all_(
    results_dir: str = "results",
    output_dir: str = "outputs/analysis/per-domain",
    n_bootstrap: int = 1000,
) -> None:
    semantic_parsing(results_dir=results_dir, output_dir=output_dir, n_bootstrap=n_bootstrap)
    hdl(results_dir=results_dir, output_dir=output_dir, n_bootstrap=n_bootstrap)


if __name__ == "__main__":
    fire.Fire({
        "semantic_parsing": semantic_parsing,
        "hdl": hdl,
        "all": all_,
    })
