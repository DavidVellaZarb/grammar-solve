#!/usr/bin/env bash
set -euo pipefail

uv run python src/eval.py \
    --adapter "${HF_NAMESPACE}/qwen2.5-7b_smcalflow_balanced-baseline" \
    --noinclude_grammar \
    --test_path "data/smcalflow/test_balanced.json" \
    --output_path results/balanced/baseline.json

uv run python src/eval.py \
    --adapter "${HF_NAMESPACE}/qwen2.5-7b_smcalflow_balanced" \
    --test_path "data/smcalflow/test_balanced.json" \
    --output_path results/balanced/test.json

uv run python src/plot.py \
    --results_dir results \
    --models '["balanced"]' \
    --model_labels '{"balanced": "Balanced: Baseline vs Grammar"}' \
    --test_labels '{"baseline": "Without Grammar", "test": "With Grammar (Ours)"}' \
    --output_path results/balanced/balanced_vs_grammar.png
