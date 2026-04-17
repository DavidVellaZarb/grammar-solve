#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME=$1
MODEL_ALIAS=$2

RESULT_DIR="results/openscad/${MODEL_ALIAS}"
PRED_DIR=outputs/predicted_grammars/rag_cot

BASELINE_ADAPTER="${HF_NAMESPACE}/${MODEL_ALIAS}_openscad-baseline"
OURS_ADAPTER="${HF_NAMESPACE}/${MODEL_ALIAS}_openscad-mixed-r0.1"

echo "=== Baseline (no grammar) ==="
uv run python src/eval_openscad.py \
    --adapter "$BASELINE_ADAPTER" \
    --test_path data/openscad/test.json \
    --noinclude_grammar \
    --output_path "${RESULT_DIR}/baseline.json"

echo "=== Ours (RAG grammar) ==="
uv run python src/eval_openscad.py \
    --adapter "$OURS_ADAPTER" \
    --test_path data/openscad/test.json \
    --include_grammar \
    --grammar_file "${PRED_DIR}/openscad_test_k64.json" \
    --output_path "${RESULT_DIR}/rag.json"

echo "=== Ours (gold grammar) ==="
uv run python src/eval_openscad.py \
    --adapter "$OURS_ADAPTER" \
    --test_path data/openscad/test.json \
    --include_grammar \
    --output_path "${RESULT_DIR}/gold.json"

echo "=== Plotting ==="
uv run python src/plot.py plot_paper_results \
    --result_files "[\"${RESULT_DIR}/baseline.json\", \"${RESULT_DIR}/rag.json\", \"${RESULT_DIR}/gold.json\"]" \
    --labels '["Baseline", "Ours (RAG)", "Gold Grammar"]' \
    --metrics '["iou", "syntax_validity"]' \
    --metric_labels '{"iou": "Volumetric IoU", "syntax_validity": "Syntax Validity"}' \
    --per_example_fields '{"iou": "iou", "syntax_validity": "valid"}' \
    --output_path "${RESULT_DIR}/comparison.png" \
    --title "OpenSCAD (${MODEL_ALIAS})"
