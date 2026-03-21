#!/usr/bin/env bash
set -euo pipefail

ADAPTER_A="${HF_NAMESPACE}/qwen2.5-7b_spice-add-specific-remove-a"
ADAPTER_B="${HF_NAMESPACE}/qwen2.5-7b_spice-add-specific-remove-b"
GRAMMAR_FILE="outputs/predicted_grammars/spice_with_modified_prompt/spice_test_k64.json"
RESULTS_DIR="results/rag_spice_ablations"

uv run python src/eval_spice.py \
    --adapter "$ADAPTER_A" \
    --grammar_file "$GRAMMAR_FILE" \
    --test_path data/spice/test.json \
    --output_path "$RESULTS_DIR/config_a/test.json"

uv run python src/eval_spice.py \
    --adapter "$ADAPTER_B" \
    --grammar_file "$GRAMMAR_FILE" \
    --test_path data/spice/test.json \
    --output_path "$RESULTS_DIR/config_b/test.json"

uv run python src/plot.py plot_multi_metrics \
    --result_files '["results/spice_with_modified_prompt/k64/test.json",
                     "'"$RESULTS_DIR"'/config_a/test.json",
                     "'"$RESULTS_DIR"'/config_b/test.json"]' \
    --metrics '["ged_similarity", "syntax_validity", "exact_match", "bleu", "component_f1"]' \
    --labels '["Gold-Trained + RAG", "Ablation A + RAG", "Ablation B + RAG"]' \
    --metric_labels '{"ged_similarity": "GED Similarity", "syntax_validity": "Syntax Validity", "exact_match": "Exact Match", "bleu": "BLEU", "component_f1": "Component F1"}' \
    --output_path "$RESULTS_DIR/comparison.png" \
    --title "SPICE — RAG-Noise Training Ablation"
