#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME=$1
MODEL_ALIAS=$2

RESULT_DIR="results/openscad/${MODEL_ALIAS}"
PRED_DIR=outputs/predicted_grammars/rag_cot

echo "=== Baseline (2-epoch, no grammar) ==="
uv run python src/eval_openscad.py \
    --adapter "${HF_NAMESPACE}/${MODEL_ALIAS}_openscad-baseline-2epoch" \
    --test_path data/openscad/test.json \
    --noinclude_grammar \
    --output_path "${RESULT_DIR}/baseline.json"

echo "=== Ours (no grammar) ==="
uv run python src/eval_openscad.py \
    --adapter "${HF_NAMESPACE}/${MODEL_ALIAS}_openscad-mixed" \
    --test_path data/openscad/test.json \
    --noinclude_grammar \
    --output_path "${RESULT_DIR}/no_grammar.json"

echo "=== Ours (mixed + RAG grammar) ==="
uv run python src/eval_openscad.py \
    --adapter "${HF_NAMESPACE}/${MODEL_ALIAS}_openscad-mixed" \
    --test_path data/openscad/test.json \
    --include_grammar \
    --grammar_file "${PRED_DIR}/openscad_test_k64.json" \
    --output_path "${RESULT_DIR}/rag.json"

echo "=== Gold grammar ==="
uv run python src/eval_openscad.py \
    --adapter "${HF_NAMESPACE}/${MODEL_ALIAS}_openscad-mixed" \
    --test_path data/openscad/test.json \
    --include_grammar \
    --output_path "${RESULT_DIR}/gold.json"

echo "=== Plotting ==="
uv run python src/plot.py plot_paper_results \
    --result_files "[\"${RESULT_DIR}/baseline.json\", \"${RESULT_DIR}/no_grammar.json\", \"${RESULT_DIR}/rag.json\", \"${RESULT_DIR}/gold.json\"]" \
    --labels '["Baseline", "Ours (No Grammar)", "Ours (RAG)", "Gold Grammar"]' \
    --metrics '["iou", "syntax_validity"]' \
    --metric_labels '{"iou": "Volumetric IoU", "syntax_validity": "Syntax Validity"}' \
    --per_example_fields '{"iou": "iou", "syntax_validity": "valid"}' \
    --output_path "${RESULT_DIR}/comparison.png" \
    --title "OpenSCAD (${MODEL_ALIAS})"
