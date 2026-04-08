#!/usr/bin/env bash
set -euo pipefail

if ! command -v iverilog &> /dev/null; then
    echo "Error: iverilog not found. Install: apt-get install -y iverilog"
    exit 1
fi

ADAPTER="${HF_NAMESPACE}/qwen2.5-7b_mg-verilog-mixed"
RESULT_DIR=results/mixed/verilog
PRED_DIR=outputs/predicted_grammars/rag_cot

echo "=== Mixed w/o Grammar ==="

uv run python src/eval_verilog.py \
    --adapter "$ADAPTER" \
    --problem_file data/verilog_eval/VerilogEval_Human.jsonl \
    --noinclude_grammar \
    --n_samples 5 \
    --temperature 0.8 \
    --output_path "${RESULT_DIR}/no_grammar/test.json"

echo "=== Mixed w/ Gold Grammar ==="

uv run python src/eval_verilog.py \
    --adapter "$ADAPTER" \
    --problem_file data/verilog_eval/VerilogEval_Human.jsonl \
    --include_grammar \
    --n_samples 5 \
    --temperature 0.8 \
    --output_path "${RESULT_DIR}/gold_grammar/test.json"

echo "=== Mixed w/ RAG CoT Grammar ==="

uv run python src/eval_verilog.py \
    --adapter "$ADAPTER" \
    --grammar_file "${PRED_DIR}/verilog_test_k64.json" \
    --problem_file data/verilog_eval/VerilogEval_Human.jsonl \
    --include_grammar \
    --n_samples 5 \
    --temperature 0.8 \
    --output_path "${RESULT_DIR}/rag_cot/test.json"

echo "=== Plotting comparison ==="

uv run python src/plot.py plot_pass_at_k \
    --result_files '["results/verilog/baseline.json", "results/verilog/grammar.json", "results/rag_cot/standard/verilog/test.json", "'"${RESULT_DIR}"'/no_grammar/test.json", "'"${RESULT_DIR}"'/gold_grammar/test.json", "'"${RESULT_DIR}"'/rag_cot/test.json"]' \
    --labels '["Baseline", "Grammar-Guided\nw/ Gold", "Grammar-Guided\nw/ RAG CoT", "Mixed\nw/o Grammar", "Mixed\nw/ Gold", "Mixed\nw/ RAG CoT"]' \
    --output_path "${RESULT_DIR}/pass_at_k.png" \
    --title "Verilog — Mixed Training Comparison"
