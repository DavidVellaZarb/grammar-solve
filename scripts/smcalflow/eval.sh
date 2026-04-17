#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME=$1
MODEL_ALIAS=$2

RESULT_DIR="results/smcalflow/${MODEL_ALIAS}"
PRED_DIR=outputs/predicted_grammars/rag_cot

BASELINE_ADAPTER="${HF_NAMESPACE}/${MODEL_ALIAS}_smcalflow-baseline"
OURS_ADAPTER="${HF_NAMESPACE}/${MODEL_ALIAS}_smcalflow-mixed-r0.1"

echo "=== Baseline (no grammar) ==="
uv run python src/eval.py \
    --adapter "$BASELINE_ADAPTER" \
    --test_path data/smcalflow/test.json \
    --noinclude_grammar \
    --output_path "${RESULT_DIR}/baseline.json"

echo "=== Ours (RAG grammar) ==="
uv run python src/eval.py \
    --adapter "$OURS_ADAPTER" \
    --test_path data/smcalflow/test.json \
    --include_grammar \
    --grammar_file "${PRED_DIR}/smcalflow_test_k64.json" \
    --output_path "${RESULT_DIR}/rag.json"

echo "=== Ours (gold grammar) ==="
uv run python src/eval.py \
    --adapter "$OURS_ADAPTER" \
    --test_path data/smcalflow/test.json \
    --include_grammar \
    --output_path "${RESULT_DIR}/gold.json"

echo "=== Plotting ==="
uv run python src/plot.py plot_paper_results \
    --result_files "[\"${RESULT_DIR}/baseline.json\", \"${RESULT_DIR}/rag.json\", \"${RESULT_DIR}/gold.json\"]" \
    --labels '["Baseline", "Ours (RAG)", "Gold Grammar"]' \
    --metrics '["accuracy"]' \
    --metric_labels '{"accuracy": "Exact Match"}' \
    --per_example_fields '{"accuracy": "match"}' \
    --output_path "${RESULT_DIR}/comparison.png" \
    --title "SMCalFlow (${MODEL_ALIAS})"
