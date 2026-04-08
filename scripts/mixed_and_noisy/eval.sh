#!/usr/bin/env bash
set -euo pipefail

ADAPTER="${HF_NAMESPACE}/qwen2.5-7b_smcalflow-mixed-noisy"

uv run python src/eval.py \
    --adapter "$ADAPTER" \
    --grammar_file outputs/predicted_grammars/rag/test_k64.json \
    --output_path results/mixed_and_noisy/mixed_noisy_with_rag_grammar.json

uv run python src/eval.py \
    --adapter "$ADAPTER" \
    --output_path results/mixed_and_noisy/mixed_noisy_with_gold_grammar.json

uv run python src/plot.py plot_bar_chart \
    --result_files '["results/mixed/mixed_with_grammar.json", "results/mixed_and_noisy/mixed_noisy_with_rag_grammar.json", "results/mixed_and_noisy/mixed_noisy_with_gold_grammar.json"]' \
    --labels '["Mixed\nw/ Gold Grammar", "Mixed+Noisy\nw/ RAG Grammar", "Mixed+Noisy\nw/ Gold Grammar"]' \
    --output_path results/mixed_and_noisy/mixed_noisy_comparison.png
