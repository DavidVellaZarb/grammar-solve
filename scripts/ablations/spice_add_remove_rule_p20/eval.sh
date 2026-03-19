#!/usr/bin/env bash
set -euo pipefail

ADAPTER="${HF_NAMESPACE}/qwen2.5-7b_spice-add-remove-rule-p20"
RESULTS_DIR="results/ablations/spice_add_remove_rule_p20"

uv run python src/eval_spice.py \
    --adapter "$ADAPTER" \
    --test_path data/spice/test.json \
    --include_grammar \
    --output_path "$RESULTS_DIR/gold_grammar.json"

uv run python src/eval_spice.py \
    --adapter "$ADAPTER" \
    --test_path data/spice/test.json \
    --grammar_file outputs/predicted_grammars/rag/spice_test_k64.json \
    --output_path "$RESULTS_DIR/rag_grammar.json"

uv run python src/plot.py plot_multi_metrics \
    --result_files '["results/spice/baseline/test.json", "'"$RESULTS_DIR"'/rag_grammar.json", "'"$RESULTS_DIR"'/gold_grammar.json"]' \
    --metrics '["ged_similarity", "syntax_validity", "exact_match", "bleu", "component_f1"]' \
    --labels '["Baseline", "RAG", "Gold Grammar"]' \
    --metric_labels '{"ged_similarity": "GED Similarity", "syntax_validity": "Syntax Validity", "exact_match": "Exact Match", "bleu": "BLEU", "component_f1": "Component F1"}' \
    --output_path "$RESULTS_DIR/comparison.png" \
    --title "SPICE — Noisy Training Grammars (add_remove=20%)"
