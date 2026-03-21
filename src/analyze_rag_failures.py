import json
import os
from collections import defaultdict

import fire
import matplotlib.pyplot as plt
import numpy as np


DOMAIN_CONFIGS = {
    "smcalflow": {
        "metric_key": "match",
        "failure_cutoff": None,
        "align_by": "index",
        "baseline_path": "results/baseline/baseline.json",
        "rag_path": "results/rag_cot/test_k64.json",
    },
    "smiles": {
        "metric_key": "fingerprint_similarity",
        "failure_cutoff": 0.5,
        "align_by": "gold",
        "baseline_path": "results/smiles/baseline/test.json",
        "rag_path": "results/rag_cot/standard/smiles/test.json",
    },
    "spice": {
        "metric_key": "ged_similarity",
        "failure_cutoff": 0.5,
        "align_by": "index",
        "baseline_path": "results/spice/baseline/test.json",
        "rag_path": "results/rag_cot/standard/spice/test.json",
    },
    "openscad": {
        "metric_key": "iou",
        "failure_cutoff": 0.1,
        "align_by": "index",
        "baseline_path": "results/openscad/baseline/test.json",
        "rag_path": "results/rag_cot/standard/openscad/test.json",
    },
    "verilog": {
        "metric_key": "passed",
        "failure_cutoff": None,
        "align_by": "task_id",
        "baseline_path": "results/verilog/baseline_samples.jsonl_results.jsonl",
        "rag_path": "results/rag_cot/standard/verilog/test_samples.jsonl_results.jsonl",
    },
}


def _load_results(path: str) -> list[dict]:
    """Load results from JSON (with 'results' key) or JSONL."""
    if path.endswith(".jsonl"):
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(path) as f:
        return json.load(f)["results"]


def _get_metric(sample: dict, metric_key: str) -> float:
    """Extract metric value, treating None as 0.0."""
    val = sample.get(metric_key)
    return 0.0 if val is None else float(val)


def _align_by_index(baseline: list[dict], rag: list[dict]) -> list[tuple[dict, dict]]:
    assert len(baseline) == len(rag), (
        f"Length mismatch: {len(baseline)} baseline vs {len(rag)} RAG"
    )
    return list(zip(baseline, rag))


def _align_by_gold(baseline: list[dict], rag: list[dict]) -> list[tuple[dict, dict]]:
    rag_by_gold = {r["gold"]: r for r in rag}
    pairs = []
    for b in baseline:
        r = rag_by_gold.get(b["gold"])
        if r is not None:
            pairs.append((b, r))
    return pairs


def _align_by_task_id(
    baseline: list[dict], rag: list[dict]
) -> list[tuple[dict, dict]]:
    """Group samples by task_id, aggregate with passed_any."""

    def _aggregate(samples: list[dict]) -> dict[str, dict]:
        groups: dict[str, list[dict]] = defaultdict(list)
        for s in samples:
            groups[s["task_id"]].append(s)
        return {
            tid: {"task_id": tid, "passed": any(s["passed"] for s in group)}
            for tid, group in groups.items()
        }

    base_agg = _aggregate(baseline)
    rag_agg = _aggregate(rag)
    shared = sorted(set(base_agg) & set(rag_agg))
    return [(base_agg[tid], rag_agg[tid]) for tid in shared]


def _is_boolean_metric(metric_key: str) -> bool:
    return metric_key in ("passed", "match")


def _plot_summary(domain_results: dict, output_path: str) -> None:
    """Stacked bar: baseline metric + RAG rescue improvement on top."""
    domains = list(domain_results.keys())
    baseline_means = [domain_results[d]["baseline_mean"] for d in domains]
    rescue_gains = [
        domain_results[d]["blended_mean"] - domain_results[d]["baseline_mean"]
        for d in domains
    ]

    x = np.arange(len(domains))
    width = 0.5

    fig, ax = plt.subplots(figsize=(max(6, len(domains) * 1.8), 5))
    ax.bar(x, baseline_means, width, label="Baseline", color="#1f77b4")
    ax.bar(x, rescue_gains, width, bottom=baseline_means,
           label="+ RAG rescue", color="#ff7f0e", hatch="//",
           edgecolor="white", linewidth=0.5)

    for i, (b, r) in enumerate(zip(baseline_means, rescue_gains)):
        combined = b + r
        ax.text(x[i], combined + 0.01, f"{combined:.1%}",
                ha="center", va="bottom", fontsize=9)
        if r > 0.005:
            ax.text(x[i], b + r / 2, f"+{r:.1%}",
                    ha="center", va="center", fontsize=8, color="black",
                    fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([d.upper() for d in domains])
    ax.set_ylabel("Mean Metric")
    ax.set_ylim(0, max(b + r for b, r in zip(baseline_means, rescue_gains)) * 1.15 + 0.02)
    ax.set_title("Baseline + RAG Failure Recovery")
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved {output_path}")


def analyze(
    domain: str,
    output_dir: str = "results/rag_failure_analysis",
    baseline_path: str | None = None,
    rag_path: str | None = None,
    metric_key: str | None = None,
    failure_cutoff: float | None = None,
    align_by: str | None = None,
) -> dict:
    """Analyze RAG failure recovery for a single domain."""
    cfg = DOMAIN_CONFIGS.get(domain, {})
    baseline_path = baseline_path or cfg["baseline_path"]
    rag_path = rag_path or cfg["rag_path"]
    metric_key = metric_key or cfg["metric_key"]
    failure_cutoff = failure_cutoff if failure_cutoff is not None else cfg["failure_cutoff"]
    align_by = align_by or cfg["align_by"]

    baseline = _load_results(baseline_path)
    rag = _load_results(rag_path)

    if align_by == "index":
        pairs = _align_by_index(baseline, rag)
    elif align_by == "gold":
        pairs = _align_by_gold(baseline, rag)
    elif align_by == "task_id":
        pairs = _align_by_task_id(baseline, rag)
    else:
        raise ValueError(f"Unknown align_by: {align_by}")

    boolean = _is_boolean_metric(metric_key)
    baseline_vals = []
    blended_vals = []

    for b, r in pairs:
        if boolean:
            bv = float(b[metric_key])
            rv = float(r[metric_key])
            baseline_vals.append(bv)
            blended_vals.append(rv if not bv else bv)
        else:
            bv = _get_metric(b, metric_key)
            rv = _get_metric(r, metric_key)
            baseline_vals.append(bv)
            blended_vals.append(rv if bv < failure_cutoff else bv)

    baseline_arr = np.array(baseline_vals)
    blended_arr = np.array(blended_vals)

    result = {
        "domain": domain,
        "total_paired": len(pairs),
        "metric_key": metric_key,
        "failure_cutoff": failure_cutoff,
        "baseline_mean": round(float(baseline_arr.mean()), 4),
        "blended_mean": round(float(blended_arr.mean()), 4),
        "rescue_gain": round(float(blended_arr.mean() - baseline_arr.mean()), 4),
        "num_rescued": int((blended_arr != baseline_arr).sum()),
    }

    domain_dir = os.path.join(output_dir, domain)
    os.makedirs(domain_dir, exist_ok=True)

    with open(os.path.join(domain_dir, "analysis.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved {domain_dir}/analysis.json")

    return result


def analyze_all(output_dir: str = "results/rag_failure_analysis") -> None:
    """Run failure recovery analysis for all 5 domains."""
    domain_results = {}
    for domain in DOMAIN_CONFIGS:
        print(f"\n{'='*60}")
        print(f"Analyzing {domain.upper()}")
        print(f"{'='*60}")
        domain_results[domain] = analyze(domain, output_dir=output_dir)

    summary = {domain: domain_results[domain] for domain in DOMAIN_CONFIGS}
    summary_path = os.path.join(output_dir, "summary.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {summary_path}")

    _plot_summary(domain_results, os.path.join(output_dir, "summary.png"))


if __name__ == "__main__":
    fire.Fire({
        "analyze": analyze,
        "analyze_all": analyze_all,
    })
