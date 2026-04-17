#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME=$1
MODEL_ALIAS=$2

RESULT_DIR="results/spice/${MODEL_ALIAS}"
PRED_DIR=outputs/predicted_grammars/rag_cot

BASELINE_ADAPTER="${HF_NAMESPACE}/${MODEL_ALIAS}_spice-baseline"
OURS_ADAPTER="${HF_NAMESPACE}/${MODEL_ALIAS}_spice-mixed-r0.1"

echo "=== Baseline (no grammar) ==="
uv run python src/eval_spice.py \
    --adapter "$BASELINE_ADAPTER" \
    --test_path data/spice/test.json \
    --noinclude_grammar \
    --output_path "${RESULT_DIR}/baseline.json"

echo "=== Ours (RAG grammar) ==="
uv run python src/eval_spice.py \
    --adapter "$OURS_ADAPTER" \
    --test_path data/spice/test.json \
    --include_grammar \
    --grammar_file "${PRED_DIR}/spice_test_k64.json" \
    --output_path "${RESULT_DIR}/rag.json"

echo "=== Ours (gold grammar) ==="
uv run python src/eval_spice.py \
    --adapter "$OURS_ADAPTER" \
    --test_path data/spice/test.json \
    --include_grammar \
    --output_path "${RESULT_DIR}/gold.json"

echo "=== Plotting ==="
uv run python src/plot.py plot_paper_results \
    --result_files "[\"${RESULT_DIR}/baseline.json\", \"${RESULT_DIR}/rag.json\", \"${RESULT_DIR}/gold.json\"]" \
    --labels '["Baseline", "Ours (RAG)", "Gold Grammar"]' \
    --metrics '["ged_similarity", "component_f1"]' \
    --metric_labels '{"ged_similarity": "GED Similarity", "component_f1": "Component F1"}' \
    --per_example_fields '{"ged_similarity": "ged_similarity", "component_f1": "component_f1"}' \
    --output_path "${RESULT_DIR}/comparison.png" \
    --title "SPICE (${MODEL_ALIAS})"
