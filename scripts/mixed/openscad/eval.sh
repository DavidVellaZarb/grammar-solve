#!/usr/bin/env bash
set -euo pipefail

ADAPTER="${HF_NAMESPACE}/qwen2.5-7b_openscad-mixed"
RESULT_DIR=results/mixed/openscad
PRED_DIR=outputs/predicted_grammars/rag_cot

echo "=== Mixed w/o Grammar ==="

uv run python src/eval_openscad.py \
    --adapter "$ADAPTER" \
    --test_path data/openscad/test.json \
    --noinclude_grammar \
    --output_path "${RESULT_DIR}/no_grammar/test.json"

echo "=== Mixed w/ Gold Grammar ==="

uv run python src/eval_openscad.py \
    --adapter "$ADAPTER" \
    --test_path data/openscad/test.json \
    --include_grammar \
    --output_path "${RESULT_DIR}/gold_grammar/test.json"

echo "=== Mixed w/ RAG CoT Grammar ==="

uv run python src/eval_openscad.py \
    --adapter "$ADAPTER" \
    --grammar_file "${PRED_DIR}/openscad_test_k64.json" \
    --test_path data/openscad/test.json \
    --output_path "${RESULT_DIR}/rag_cot/test.json"

echo "=== Plotting comparison ==="

uv run python src/plot.py plot_multi_metrics \
    --result_files '["results/openscad/baseline/test.json", "results/openscad/grammar/test.json", "results/rag_cot/standard/openscad/test.json", "'"${RESULT_DIR}"'/no_grammar/test.json", "'"${RESULT_DIR}"'/gold_grammar/test.json", "'"${RESULT_DIR}"'/rag_cot/test.json"]' \
    --metrics '["iou", "syntax_validity", "bleu"]' \
    --labels '["Baseline", "Grammar-Guided\nw/ Gold", "Grammar-Guided\nw/ RAG CoT", "Mixed\nw/o Grammar", "Mixed\nw/ Gold", "Mixed\nw/ RAG CoT"]' \
    --metric_labels '{"iou": "Volumetric IoU", "syntax_validity": "Syntax Validity", "bleu": "BLEU"}' \
    --output_path "${RESULT_DIR}/comparison.png" \
    --title "OpenSCAD — Mixed Training Comparison"
