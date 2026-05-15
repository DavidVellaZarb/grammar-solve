"""Per-model multi-panel diagram for the paper.

Given a single model alias, renders a five-panel figure comparing the three
configurations (no-grammar baseline, RAG-predicted grammars, gold grammars)
across SMCalFlow / GeoQuery / Overnight / Verilog / SPICE. Saved as both
panel.pdf and panel.png in outputs/analysis/models/{alias}/.

Per-domain metrics:
- SMCalFlow: exact match
- GeoQuery:  exact match, execution accuracy
- Overnight: exact match, execution accuracy
- Verilog:   pass@1, pass@3, pass@5 (per-problem unbiased estimator, bootstrapped)
- SPICE:     GED similarity, component F1

All bars are 95% bootstrap CIs over the same per-example / per-problem unit
that defines the metric.
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path

import fire
import json
import matplotlib.pyplot as plt
import numpy as np

from bootstrap import bootstrap_ci
from plot_per_domain import COLORS, CONFIGS, _per_example_from_json
from verilog_eval.evaluation import estimate_pass_at_k


PANELS: list[tuple[str, str, list[tuple[str, str, object]]]] = [
    ("smcalflow", "SMCalFlow", [
        ("Exact Match", "per_ex", "match"),
    ]),
    ("geoquery", "GeoQuery", [
        ("Exact Match", "per_ex", "exact_match"),
        ("Exec. Acc.",  "per_ex", "execution_match"),
    ]),
    ("overnight", "Overnight", [
        ("Exact Match", "per_ex", "exact_match"),
        ("Exec. Acc.",  "per_ex", "execution_match"),
    ]),
    ("verilog", "Verilog", [
        ("pass@1", "verilog", 1),
        ("pass@3", "verilog", 3),
        ("pass@5", "verilog", 5),
    ]),
    ("spice", "SPICE", [
        ("GED Sim.", "per_ex", "ged_similarity"),
        ("Comp. F1", "per_ex", "component_f1"),
    ]),
]

WIDTH_RATIOS = [len(metrics) for _, _, metrics in PANELS]

DEFAULT_ALIASES = ("qwen3-5-4b", "qwen3-4b", "gemma-3-12b", "gemma-3-27b")


def _warn(msg: str) -> None:
    print(f"[plot_per_model] {msg}", file=sys.stderr)


def _passes_by_task(path: str, cache: dict) -> dict[str, list[bool]]:
    if path in cache:
        return cache[path]
    by_task: dict[str, list[bool]] = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            by_task[obj["task_id"]].append(bool(obj.get("passed", False)))
    cache[path] = dict(by_task)
    return cache[path]


def _per_problem_pass_at_k_from_verilog(path: str, k: int, cache: dict) -> list[float]:
    by_task = _passes_by_task(path, cache)
    values: list[float] = []
    min_n = None
    for passes in by_task.values():
        if not passes:
            continue
        n = len(passes)
        c = sum(passes)
        values.append(float(estimate_pass_at_k(n, [c], k)[0]))
        min_n = n if min_n is None else min(min_n, n)
    if min_n is not None and min_n < k:
        _warn(
            f"verilog {path}: min samples/task={min_n} < k={k}; pass@{k} will be degenerate"
        )
    return values


def _collect(
    domain: str,
    alias: str,
    config: str,
    extractor: tuple[str, object],
    results_dir: str,
    verilog_cache: dict,
) -> list[float] | None:
    kind, arg = extractor
    if kind == "per_ex":
        path = Path(results_dir) / domain / alias / f"{config}.json"
        if not path.exists():
            _warn(f"missing {path}; skipping bar")
            return None
        return _per_example_from_json(str(path), str(arg))
    if kind == "verilog":
        path = (
            Path(results_dir) / domain / alias /
            f"{config}_samples.jsonl_results.jsonl"
        )
        if not path.exists():
            _warn(f"missing {path}; skipping bar (need n_samples>=5 verilog eval)")
            return None
        return _per_problem_pass_at_k_from_verilog(str(path), int(arg), verilog_cache)
    raise ValueError(f"unknown extractor kind: {kind}")


def per_model(
    model_alias: str,
    results_dir: str = "results",
    output_dir: str = "outputs/analysis/models",
    n_bootstrap: int = 1000,
    figsize: tuple[float, float] = (13.0, 2.8),
) -> None:
    """Render the per-model multi-panel figure to {output_dir}/{model_alias}/panel.{pdf,png}."""
    fig, axes = plt.subplots(
        1,
        len(PANELS),
        sharey=True,
        figsize=figsize,
        gridspec_kw={"width_ratios": WIDTH_RATIOS, "wspace": 0.18},
    )

    bar_width = 0.25
    offsets = np.array([-bar_width, 0.0, bar_width])
    verilog_cache: dict = {}
    any_bar_drawn = False

    for ax_idx, (domain, display_name, metrics) in enumerate(PANELS):
        ax = axes[ax_idx]
        num_metrics = len(metrics)
        x_positions = np.arange(num_metrics, dtype=float)

        for cfg_idx, (config_key, config_label) in enumerate(CONFIGS):
            means: list[float] = []
            errs_low: list[float] = []
            errs_high: list[float] = []
            skip_mask: list[bool] = []

            for _, kind, arg in metrics:
                values = _collect(
                    domain, model_alias, config_key, (kind, arg),
                    results_dir, verilog_cache,
                )
                if values is None:
                    skip_mask.append(True)
                    means.append(0.0)
                    errs_low.append(0.0)
                    errs_high.append(0.0)
                    continue
                skip_mask.append(False)
                stats = bootstrap_ci(values, n_bootstrap=n_bootstrap)
                means.append(stats["mean"])
                errs_low.append(max(0.0, stats["mean"] - stats["ci_low"]))
                errs_high.append(max(0.0, stats["ci_high"] - stats["mean"]))

            xs = x_positions + offsets[cfg_idx]
            xs_kept = [x for x, skip in zip(xs, skip_mask) if not skip]
            means_kept = [m for m, skip in zip(means, skip_mask) if not skip]
            low_kept = [l for l, skip in zip(errs_low, skip_mask) if not skip]
            high_kept = [h for h, skip in zip(errs_high, skip_mask) if not skip]
            if not xs_kept:
                continue
            any_bar_drawn = True
            ax.bar(
                xs_kept,
                means_kept,
                bar_width,
                yerr=[low_kept, high_kept],
                capsize=2,
                color=COLORS[cfg_idx],
                label=config_label,
                error_kw=dict(elinewidth=0.8),
            )

        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            [label for label, _, _ in metrics],
            fontsize=8,
        )
        ax.set_xlabel(display_name, fontsize=9)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlim(-0.5, num_metrics - 0.5)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    axes[0].set_ylabel("Score", fontsize=9)

    if not any_bar_drawn:
        _warn(f"no data found for alias '{model_alias}'; skipping figure")
        plt.close(fig)
        return

    handles: list = []
    labels: list = []
    seen: set[str] = set()
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                seen.add(l)
                handles.append(h)
                labels.append(l)
        if len(handles) == len(CONFIGS):
            break
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=len(CONFIGS),
        frameon=False,
        fontsize=9,
        columnspacing=2.0,
        handlelength=1.5,
    )

    out_dir = Path(output_dir) / model_alias
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "panel.pdf"
    png_path = out_dir / "panel.png"
    fig.savefig(pdf_path, bbox_inches="tight", format="pdf")
    fig.savefig(png_path, bbox_inches="tight", format="png", dpi=300)
    plt.close(fig)
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


def all_(
    model_aliases: tuple[str, ...] = DEFAULT_ALIASES,
    results_dir: str = "results",
    output_dir: str = "outputs/analysis/models",
    n_bootstrap: int = 1000,
) -> None:
    for alias in model_aliases:
        per_model(
            model_alias=alias,
            results_dir=results_dir,
            output_dir=output_dir,
            n_bootstrap=n_bootstrap,
        )


if __name__ == "__main__":
    fire.Fire({"per_model": per_model, "all": all_})
