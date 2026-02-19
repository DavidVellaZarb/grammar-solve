#!/usr/bin/env bash
set -euo pipefail

ADAPTER="dv347/qwen2.5-7b_smcalflow"
RESULTS_DIR="results/baseline"

uv run python src/eval.py \
    --adapter "$ADAPTER" \
    --test_path data/smcalflow/test.json \
    --output_path "$RESULTS_DIR/test.json" \
    "$@"

uv run python src/eval.py \
    --adapter "$ADAPTER" \
    --test_path data/smcalflow/test_add_rule.json \
    --output_path "$RESULTS_DIR/test_add_rule.json" \
    "$@"

uv run python src/eval.py \
    --adapter "$ADAPTER" \
    --test_path data/smcalflow/test_remove_rule.json \
    --output_path "$RESULTS_DIR/test_remove_rule.json" \
    "$@"

uv run python src/plot.py \
    --results_dir results \
    --models '["baseline"]' \
    --output_path figures/ablation_baseline.png \
    --title "Baseline Model" \
    --model_labels '{"baseline": "Baseline"}' \
    --test_labels '{"test": "Original", "test_add_rule": "Added Rules", "test_remove_rule": "Removed Rules"}'
