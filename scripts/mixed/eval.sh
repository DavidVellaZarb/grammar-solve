#!/usr/bin/env bash
set -euo pipefail

uv run python src/eval.py \
    --adapter "${HF_NAMESPACE}/qwen2.5-7b_smcalflow-baseline" \
    --noinclude_grammar \
    --output_path results/mixed/baseline.json

uv run python src/eval.py \
    --adapter "${HF_NAMESPACE}/qwen2.5-7b_smcalflow" \
    --noinclude_grammar \
    --output_path results/mixed/grammar_guided_no_grammar.json

uv run python src/eval.py \
    --adapter "${HF_NAMESPACE}/qwen2.5-7b_smcalflow-mixed" \
    --noinclude_grammar \
    --output_path results/mixed/mixed_no_grammar.json

uv run python src/eval.py \
    --adapter "${HF_NAMESPACE}/qwen2.5-7b_smcalflow" \
    --output_path results/mixed/grammar_guided_with_grammar.json

uv run python src/eval.py \
    --adapter "${HF_NAMESPACE}/qwen2.5-7b_smcalflow-mixed" \
    --output_path results/mixed/mixed_with_grammar.json

uv run python src/plot.py plot_bar_chart \
    --result_files '["results/mixed/baseline.json", "results/mixed/grammar_guided_no_grammar.json", "results/mixed/mixed_no_grammar.json", "results/mixed/grammar_guided_with_grammar.json", "results/mixed/mixed_with_grammar.json"]' \
    --labels '["Baseline", "Grammar-Guided\nw/o Grammar", "Mixed\nw/o Grammar", "Grammar-Guided\nw/ Grammar", "Mixed\nw/ Grammar"]' \
    --output_path results/mixed/mixed_comparison.png
