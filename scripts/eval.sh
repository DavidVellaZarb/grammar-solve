#!/usr/bin/env bash
set -euo pipefail

uv run python src/eval.py \
    --adapter "${HF_NAMESPACE:-}/qwen2.5-7b_smcalflow-baseline" \
    --noinclude_grammar \
    --output_path results/baseline/baseline.json \
    "$@"

uv run python src/eval.py \
    --adapter "${HF_NAMESPACE:-}/qwen2.5-7b_smcalflow" \
    --output_path results/baseline/test.json \
    "$@"

uv run python src/plot.py \
    --results_dir results \
    --models '["baseline"]' \
    --model_labels '{"baseline": "Baseline vs Grammar"}' \
    --test_labels '{"baseline": "Without Grammar", "test": "With Grammar (Ours)"}' \
    --output_path results/baseline/baseline_vs_grammar.png
