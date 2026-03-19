#!/usr/bin/env bash
set -euo pipefail

uv run python src/eval.py \
    --adapter "${HF_NAMESPACE}/qwen2.5-7b_smcalflow-baseline" \
    --noinclude_grammar --constrained --batch_size 8 \
    --output_path results/constrained/baseline_constrained.json

uv run python src/eval.py \
    --adapter "${HF_NAMESPACE}/qwen2.5-7b_smcalflow" \
    --constrained --batch_size 8 \
    --output_path results/constrained/test_constrained.json

uv run python src/plot.py plot_bar_chart \
    --result_files '["results/baseline/baseline.json", "results/baseline/test.json", "results/constrained/baseline_constrained.json", "results/constrained/test_constrained.json"]' \
    --labels '["Baseline", "Grammar (Ours)", "Baseline + Constrained", "Grammar + Constrained"]' \
    --output_path results/constrained/constrained_decoding.png \
    --title "Effect of Grammar-Constrained Decoding"
