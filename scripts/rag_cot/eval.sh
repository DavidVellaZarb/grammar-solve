#!/usr/bin/env bash
set -euo pipefail

ADAPTER="${HF_NAMESPACE}/qwen2.5-7b_smcalflow-add-remove-rule-p20"

uv run python src/eval.py \
    --adapter "$ADAPTER" \
    --grammar_file outputs/predicted_grammars/rag_cot/test_k64.json \
    --test_path data/smcalflow/test.json \
    --output_path results/rag_cot/test_k64.json

uv run python src/plot.py plot_bar_chart \
    --result_files '["results/rag_ablated/test_k64.json", "results/rag_cot/test_k64.json"]' \
    --labels '["RAG k=64", "RAG CoT k=64"]' \
    --output_path results/rag_cot/rag_cot_comparison.png \
    --title "RAG CoT vs Standard — Program Accuracy" \
    --ylabel Accuracy
