#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME=$1
MODEL_ALIAS=$2

RESULT_DIR="results/overnight/${MODEL_ALIAS}"
PRED_DIR=outputs/predicted_grammars/rag_cot

echo "=== Baseline (2-epoch, no grammar) ==="
uv run python src/eval_overnight.py \
    --adapter "${HF_NAMESPACE}/${MODEL_ALIAS}_overnight-baseline-2epoch" \
    --test_path data/overnight/test.json \
    --noinclude_grammar \
    --output_path "${RESULT_DIR}/baseline.json"

echo "=== Ours (no grammar) ==="
uv run python src/eval_overnight.py \
    --adapter "${HF_NAMESPACE}/${MODEL_ALIAS}_overnight-mixed" \
    --test_path data/overnight/test.json \
    --noinclude_grammar \
    --output_path "${RESULT_DIR}/no_grammar.json"

echo "=== Ours (mixed + RAG grammar) ==="
uv run python src/eval_overnight.py \
    --adapter "${HF_NAMESPACE}/${MODEL_ALIAS}_overnight-mixed" \
    --test_path data/overnight/test.json \
    --include_grammar \
    --grammar_file "${PRED_DIR}/overnight_test_k64.json" \
    --output_path "${RESULT_DIR}/rag.json"

echo "=== Gold grammar ==="
uv run python src/eval_overnight.py \
    --adapter "${HF_NAMESPACE}/${MODEL_ALIAS}_overnight-mixed" \
    --test_path data/overnight/test.json \
    --include_grammar \
    --output_path "${RESULT_DIR}/gold.json"

echo "=== Plotting ==="
uv run python src/plot.py plot_paper_results \
    --result_files "[\"${RESULT_DIR}/baseline.json\", \"${RESULT_DIR}/no_grammar.json\", \"${RESULT_DIR}/rag.json\", \"${RESULT_DIR}/gold.json\"]" \
    --labels '["Baseline", "Ours (No Grammar)", "Ours (RAG)", "Gold Grammar"]' \
    --metrics '["accuracy", "execution_accuracy"]' \
    --metric_labels '{"accuracy": "Exact Match", "execution_accuracy": "Execution Accuracy"}' \
    --per_example_fields '{"accuracy": "match", "execution_accuracy": "execution_match"}' \
    --output_path "${RESULT_DIR}/comparison.png" \
    --title "Overnight-Blocks (${MODEL_ALIAS})"
