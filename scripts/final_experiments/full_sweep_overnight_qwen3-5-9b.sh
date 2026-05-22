#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="Qwen/Qwen3.5-9B"
MODEL_ALIAS="qwen3-5-9b"
DOMAIN="overnight"

TRAIN_PATH="data/${DOMAIN}/train.json"
VALID_PATH="data/${DOMAIN}/valid.json"
TEST_PATH="data/${DOMAIN}/test.json"
RAG_FILE="outputs/predicted_grammars/rag_cot/${DOMAIN}_test_k64.json"

RESULT_DIR="results/full_sweep/${DOMAIN}/${MODEL_ALIAS}"
BASELINE_RESULT="results/${DOMAIN}/${MODEL_ALIAS}/baseline.json"
RATIOS=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

if [[ -f ".env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source ".env"
    set +a
fi

: "${HF_NAMESPACE:?Set HF_NAMESPACE in your environment or .env}"

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "WARNING: HF_TOKEN is not set; HuggingFace checks will rely on cached auth."
fi

for path in "$TRAIN_PATH" "$VALID_PATH" "$TEST_PATH" "$RAG_FILE"; do
    if [[ ! -f "$path" ]]; then
        echo "Missing required file: $path" >&2
        exit 1
    fi
done

mkdir -p "$RESULT_DIR"

hf_model_exists() {
    local repo_id=$1
    uv run python - "$repo_id" <<'PY'
import os
import sys

from huggingface_hub import HfApi

repo_id = sys.argv[1]
api = HfApi(token=os.getenv("HF_TOKEN") or None)

try:
    exists = api.repo_exists(repo_id=repo_id, repo_type="model")
except Exception as exc:
    print(f"ERROR: failed to check HuggingFace repo {repo_id}: {type(exc).__name__}: {exc}", file=sys.stderr)
    sys.exit(2)

sys.exit(0 if exists else 1)
PY
}

ensure_model() {
    local ratio=$1
    local hub_id="${HF_NAMESPACE}/${MODEL_ALIAS}_${DOMAIN}-mixed-r${ratio}"
    local output_dir="outputs/${MODEL_ALIAS}-lora-${DOMAIN}-mixed-r${ratio}"

    echo "=== Model ratio=${ratio}: ${hub_id} ==="

    set +e
    hf_model_exists "$hub_id"
    local exists_status=$?
    set -e

    case "$exists_status" in
        0)
            echo "Adapter exists on HuggingFace; skipping training."
            ;;
        1)
            echo "Adapter missing; training ratio=${ratio}."
            uv run python src/train.py \
                --model_name "$MODEL_NAME" \
                --mixed_ratio "$ratio" \
                --num_train_epochs 1 \
                --train_path "$TRAIN_PATH" \
                --valid_path "$VALID_PATH" \
                --output_dir "$output_dir" \
                --hub_model_id "$hub_id"
            ;;
        *)
            exit "$exists_status"
            ;;
    esac
}

run_eval_if_missing() {
    local label=$1
    local output_path=$2
    shift 2

    if [[ -f "$output_path" ]]; then
        echo "Result exists; skipping ${label}: ${output_path}"
        return
    fi

    echo "=== Eval ${label}: ${output_path} ==="
    uv run python src/eval_overnight.py \
        --adapter "$HUB_ID" \
        --test_path "$TEST_PATH" \
        --skip_execution \
        "$@" \
        --output_path "$output_path"
}

for ratio in "${RATIOS[@]}"; do
    ensure_model "$ratio"
done

for ratio in "${RATIOS[@]}"; do
    HUB_ID="${HF_NAMESPACE}/${MODEL_ALIAS}_${DOMAIN}-mixed-r${ratio}"

    run_eval_if_missing \
        "gold grammar ratio=${ratio}" \
        "${RESULT_DIR}/gold_r${ratio}.json" \
        --include_grammar

    run_eval_if_missing \
        "RAG grammar ratio=${ratio}" \
        "${RESULT_DIR}/rag_r${ratio}.json" \
        --include_grammar \
        --grammar_file "$RAG_FILE"

    run_eval_if_missing \
        "no grammar ratio=${ratio}" \
        "${RESULT_DIR}/no_grammar_r${ratio}.json" \
        --noinclude_grammar
done

echo "=== Plotting full sweep ==="
uv run python - "$RESULT_DIR" "$BASELINE_RESULT" "${RATIOS[@]}" <<'PY'
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

result_dir = Path(sys.argv[1])
baseline_path = Path(sys.argv[2])
ratios = sys.argv[3:]

series = [
    ("gold", "Gold grammars", "#55A868", "o"),
    ("rag", "RAG grammars", "#4C72B0", "s"),
    ("no_grammar", "No grammar", "#DD8452", "^"),
]

metrics = [
    {
        "key": "accuracy",
        "suffix": "accuracy",
        "ylabel": "Exact-match accuracy",
        "summary": "accuracy_summary.csv",
        "plot": "accuracy_vs_proportion.png",
    },
]


def load_metric(key: str, ratio: str, metric_key: str) -> float:
    path = result_dir / f"{key}_r{ratio}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing result file: {path}")
    with path.open() as f:
        data = json.load(f)
    if metric_key not in data:
        raise KeyError(f"Missing {metric_key} in {path}")
    return float(data[metric_key])


def load_baseline(metric_key: str) -> tuple[float, Path]:
    path = baseline_path
    if not path.exists():
        path = result_dir / "no_grammar_r1.0.json"
    with path.open() as f:
        data = json.load(f)
    if metric_key not in data:
        raise KeyError(f"Missing {metric_key} in {path}")
    return float(data[metric_key]), path


def write_summary(metric: dict[str, str]) -> Path:
    summary_path = result_dir / metric["summary"]
    fieldnames = ["ratio"] + [f"{key}_{metric['suffix']}" for key, *_ in series]
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ratio in ratios:
            row = {"ratio": ratio}
            for key, *_ in series:
                row[f"{key}_{metric['suffix']}"] = load_metric(key, ratio, metric["key"])
            writer.writerow(row)
    return summary_path


def write_plot(metric: dict[str, str]) -> tuple[Path, Path]:
    baseline, source_path = load_baseline(metric["key"])
    xs = [float(ratio) for ratio in ratios]

    fig, ax = plt.subplots(figsize=(8, 5))
    for key, label, color, marker in series:
        ys = [load_metric(key, ratio, metric["key"]) for ratio in ratios]
        ax.plot(
            xs,
            ys,
            color=color,
            marker=marker,
            markersize=6,
            linewidth=1.8,
            label=label,
        )

    ax.axhline(
        baseline,
        color="#555555",
        linestyle=":",
        linewidth=1.8,
        label=f"Baseline ({baseline:.1%})",
    )

    ax.set_xlim(0, 1)
    ax.set_xticks([i / 10 for i in range(11)])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Proportion without grammars in training")
    ax.set_ylabel(metric["ylabel"])
    ax.grid(True, alpha=0.3)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=True,
        fontsize=10,
        columnspacing=1.2,
        handlelength=2.2,
    )

    plot_path = result_dir / metric["plot"]
    fig.tight_layout()
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return plot_path, source_path


for metric in metrics:
    summary_path = write_summary(metric)
    plot_path, baseline_source = write_plot(metric)
    print(f"Saved {summary_path}")
    print(f"Saved {plot_path} (baseline from {baseline_source})")
PY
