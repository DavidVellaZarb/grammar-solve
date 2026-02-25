#!/usr/bin/env bash
set -euo pipefail

ADAPTER="${HF_NAMESPACE}/qwen2.5-7b_smcalflow"
RESULTS_DIR="results/ablations_baseline"

uv run python src/eval.py \
    --adapter "$ADAPTER" \
    --test_path data/smcalflow/test.json \
    --output_path "$RESULTS_DIR/test.json" \
    "$@"

uv run python src/eval.py \
    --adapter "$ADAPTER" \
    --test_path data/smcalflow/ablations/test_add_rule.json \
    --output_path "$RESULTS_DIR/test_add_rule.json" \
    "$@"

uv run python src/eval.py \
    --adapter "$ADAPTER" \
    --test_path data/smcalflow/ablations/test_remove_rule.json \
    --output_path "$RESULTS_DIR/test_remove_rule.json" \
    "$@"

uv run python src/eval.py \
    --adapter "$ADAPTER" \
    --test_path data/smcalflow/ablations/test_add_remove_rule.json \
    --output_path "$RESULTS_DIR/test_add_remove_rule.json" \
    "$@"

uv run python src/plot.py \
    --results_dir results \
    --models '["ablations_baseline"]' \
    --output_path results/ablations_baseline/ablation_baseline.png \
    --title "Baseline Model" \
    --model_labels '{"ablations_baseline": "Baseline"}' \
    --test_labels '{"test": "Original", "test_add_rule": "Added Rules", "test_remove_rule": "Removed Rules", "test_add_remove_rule": "Added+Removed Rules"}'
