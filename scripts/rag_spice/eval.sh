#!/usr/bin/env bash
set -euo pipefail

uv run python src/eval_spice.py \
    --adapter "${HF_NAMESPACE}/qwen2.5-7b_spice" \
    --grammar_file outputs/predicted_grammars/rag/spice_test_k64.json \
    --test_path data/spice/test.json \
    --output_path results/rag_spice/test.json

uv run python src/plot.py plot_multi_metrics \
    --result_files '["results/spice/baseline/test.json", "results/rag_spice/test.json", "results/spice/grammar/test.json"]' \
    --metrics '["ged_similarity", "syntax_validity", "exact_match", "bleu", "component_f1"]' \
    --labels '["Baseline", "RAG", "Gold Grammar"]' \
    --metric_labels '{"ged_similarity": "GED Similarity", "syntax_validity": "Syntax Validity", "exact_match": "Exact Match", "bleu": "BLEU", "component_f1": "Component F1"}' \
    --output_path results/rag_spice/comparison.png \
    --title "SPICE — RAG Grammar Prediction"
