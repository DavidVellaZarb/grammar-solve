#!/usr/bin/env bash
set -euo pipefail

uv run python src/eval_spice.py \
    --adapter "${HF_NAMESPACE}/qwen2.5-7b_spice-baseline" \
    --test_path data/spice/test.json \
    --noinclude_grammar \
    --output_path results/spice/baseline/test.json

uv run python src/eval_spice.py \
    --adapter "${HF_NAMESPACE}/qwen2.5-7b_spice" \
    --test_path data/spice/test.json \
    --include_grammar \
    --output_path results/spice/grammar/test.json

uv run python src/plot.py plot_accuracies \
    --results_dir results/spice \
    --models '["baseline", "grammar"]' \
    --model_labels '{"baseline": "Baseline", "grammar": "Grammar-Guided (Ours)"}' \
    --output_path results/spice/comparison.png \
    --title "SPICE Netlist Generation Accuracy"
