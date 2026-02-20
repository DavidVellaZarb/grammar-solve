#!/usr/bin/env bash
set -euo pipefail

uv run python src/eval.py \
    --adapter "${HF_NAMESPACE:-}/qwen2.5-7b_smcalflow-baseline" \
    --noinclude_grammar \
    --output_path results/baseline/test.json \
    "$@"

uv run python src/eval.py \
    --adapter "${HF_NAMESPACE:-}/qwen2.5-7b_smcalflow" \
    --output_path results/grammar/test.json \
    "$@"

uv run python src/plot.py \
    --results_dir results \
    --models '["baseline", "grammar"]' \
    --model_labels '{"baseline": "Baseline", "grammar": "With Grammar (Ours)"}' \
    --test_labels '{"test": "Test Set"}' \
    --output_path results/baseline_vs_grammar.png
