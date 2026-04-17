#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME=$1
MODEL_ALIAS=$2

if ! command -v iverilog &> /dev/null; then
    echo "Error: iverilog not found. Install: apt-get install -y iverilog"
    exit 1
fi

RESULT_DIR="results/verilog/${MODEL_ALIAS}"
PRED_DIR=outputs/predicted_grammars/rag_cot

BASELINE_ADAPTER="${HF_NAMESPACE}/${MODEL_ALIAS}_mg-verilog-baseline"
OURS_ADAPTER="${HF_NAMESPACE}/${MODEL_ALIAS}_mg-verilog-mixed-r0.1"

echo "=== Baseline (no grammar) ==="
uv run python src/eval_verilog.py \
    --adapter "$BASELINE_ADAPTER" \
    --problem_file data/verilog_eval/VerilogEval_Human.jsonl \
    --noinclude_grammar \
    --n_samples 5 \
    --temperature 0.8 \
    --output_path "${RESULT_DIR}/baseline.json"

echo "=== Ours (RAG grammar) ==="
uv run python src/eval_verilog.py \
    --adapter "$OURS_ADAPTER" \
    --problem_file data/verilog_eval/VerilogEval_Human.jsonl \
    --include_grammar \
    --grammar_file "${PRED_DIR}/verilog_test_k64.json" \
    --n_samples 5 \
    --temperature 0.8 \
    --output_path "${RESULT_DIR}/rag.json"

echo "=== Ours (gold grammar) ==="
uv run python src/eval_verilog.py \
    --adapter "$OURS_ADAPTER" \
    --problem_file data/verilog_eval/VerilogEval_Human.jsonl \
    --include_grammar \
    --n_samples 5 \
    --temperature 0.8 \
    --output_path "${RESULT_DIR}/gold.json"

echo "=== Plotting ==="
uv run python src/plot.py plot_paper_pass_at_k \
    --result_files "[\"${RESULT_DIR}/baseline.json\", \"${RESULT_DIR}/rag.json\", \"${RESULT_DIR}/gold.json\"]" \
    --labels '["Baseline", "Ours (RAG)", "Gold Grammar"]' \
    --output_path "${RESULT_DIR}/comparison.png" \
    --title "VerilogEval (${MODEL_ALIAS})"
