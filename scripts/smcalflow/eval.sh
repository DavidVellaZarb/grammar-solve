#!/usr/bin/env bash
set -euo pipefail

RESULT_DIR=results/smcalflow
PRED_DIR=outputs/predicted_grammars/rag_cot

echo "=== Baseline (2-epoch, no grammar) ==="
uv run python src/eval.py \
    --adapter "${HF_NAMESPACE}/qwen2.5-7b_smcalflow-baseline-2epoch" \
    --test_path data/smcalflow/test.json \
    --noinclude_grammar \
    --output_path "${RESULT_DIR}/baseline.json"

echo "=== Ours (mixed + RAG grammar) ==="
uv run python src/eval.py \
    --adapter "${HF_NAMESPACE}/qwen2.5-7b_smcalflow-mixed" \
    --test_path data/smcalflow/test.json \
    --include_grammar \
    --grammar_file "${PRED_DIR}/smcalflow_test_k64.json" \
    --output_path "${RESULT_DIR}/rag.json"

echo "=== Gold grammar ==="
uv run python src/eval.py \
    --adapter "${HF_NAMESPACE}/qwen2.5-7b_smcalflow-mixed" \
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
    --title "SMCalFlow"
