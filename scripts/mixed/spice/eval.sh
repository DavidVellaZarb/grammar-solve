#!/usr/bin/env bash
set -euo pipefail

ADAPTER="${HF_NAMESPACE}/qwen2.5-7b_spice-mixed"
RESULT_DIR=results/mixed/spice
PRED_DIR=outputs/predicted_grammars/rag_cot

echo "=== Mixed w/o Grammar ==="

uv run python src/eval_spice.py \
    --adapter "$ADAPTER" \
    --test_path data/spice/test.json \
    --noinclude_grammar \
    --output_path "${RESULT_DIR}/no_grammar/test.json"

echo "=== Mixed w/ Gold Grammar ==="

uv run python src/eval_spice.py \
    --adapter "$ADAPTER" \
    --test_path data/spice/test.json \
    --include_grammar \
    --output_path "${RESULT_DIR}/gold_grammar/test.json"

echo "=== Mixed w/ RAG CoT Grammar ==="

uv run python src/eval_spice.py \
    --adapter "$ADAPTER" \
    --grammar_file "${PRED_DIR}/spice_test_k64.json" \
    --test_path data/spice/test.json \
    --output_path "${RESULT_DIR}/rag_cot/test.json"

echo "=== Plotting comparison ==="

uv run python src/plot.py plot_multi_metrics \
    --result_files '["results/spice/baseline/test.json", "results/spice/grammar/test.json", "results/rag_cot/standard/spice/test.json", "'"${RESULT_DIR}"'/no_grammar/test.json", "'"${RESULT_DIR}"'/gold_grammar/test.json", "'"${RESULT_DIR}"'/rag_cot/test.json"]' \
    --metrics '["ged_similarity", "syntax_validity", "exact_match", "bleu", "component_f1"]' \
    --labels '["Baseline", "Grammar-Guided\nw/ Gold", "Grammar-Guided\nw/ RAG CoT", "Mixed\nw/o Grammar", "Mixed\nw/ Gold", "Mixed\nw/ RAG CoT"]' \
    --metric_labels '{"ged_similarity": "GED Similarity", "syntax_validity": "Syntax Validity", "exact_match": "Exact Match", "bleu": "BLEU", "component_f1": "Component F1"}' \
    --output_path "${RESULT_DIR}/comparison.png" \
    --title "SPICE — Mixed Training Comparison"
