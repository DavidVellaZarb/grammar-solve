#!/usr/bin/env bash
set -euo pipefail

if ! command -v iverilog &> /dev/null; then
    echo "Error: iverilog not found. Install: apt-get install -y iverilog"
    exit 1
fi

ADAPTER="${HF_NAMESPACE}/qwen2.5-7b_mg-verilog-add-remove-rule-p20"
RESULTS_DIR="results/ablations/verilog_add_remove_rule_p20"

uv run python src/eval_verilog.py \
    --adapter "$ADAPTER" \
    --problem_file data/verilog_eval/VerilogEval_Human.jsonl \
    --include_grammar \
    --n_samples 5 \
    --temperature 0.8 \
    --output_path "$RESULTS_DIR/gold_grammar.json"

uv run python src/eval_verilog.py \
    --adapter "$ADAPTER" \
    --problem_file data/verilog_eval/VerilogEval_Human.jsonl \
    --include_grammar \
    --grammar_file outputs/predicted_grammars/rag/verilog_test_k64.json \
    --n_samples 5 \
    --temperature 0.8 \
    --output_path "$RESULTS_DIR/rag_grammar.json"

uv run python src/plot.py plot_pass_at_k \
    --result_files '["results/verilog/baseline.json", "'"$RESULTS_DIR"'/rag_grammar.json", "'"$RESULTS_DIR"'/gold_grammar.json"]' \
    --labels '["Baseline", "RAG", "Gold Grammar"]' \
    --output_path "$RESULTS_DIR/pass_at_k.png" \
    --title "Verilog — Ablated Model (add_remove p=20%)"
