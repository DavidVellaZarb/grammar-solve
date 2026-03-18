#!/usr/bin/env bash
set -euo pipefail

if ! command -v iverilog &> /dev/null; then
    echo "Error: iverilog not found. Install: apt-get install -y iverilog"
    exit 1
fi

uv run python src/eval_verilog.py \
    --adapter "${HF_NAMESPACE}/qwen2.5-7b_mg-verilog" \
    --grammar_file outputs/predicted_grammars/rag/verilog_test_k64.json \
    --problem_file data/verilog_eval/VerilogEval_Human.jsonl \
    --include_grammar \
    --n_samples 5 \
    --temperature 0.8 \
    --output_path results/rag_verilog/test.json

uv run python src/plot.py plot_pass_at_k \
    --result_files '["results/verilog/baseline.json", "results/rag_verilog/test.json", "results/verilog/grammar.json"]' \
    --labels '["Baseline", "RAG", "Gold Grammar"]' \
    --output_path results/rag_verilog/pass_at_k.png \
    --title "Verilog — RAG Grammar Prediction"
