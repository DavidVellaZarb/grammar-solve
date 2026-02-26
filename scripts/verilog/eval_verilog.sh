#!/usr/bin/env bash
set -euo pipefail

if ! command -v iverilog &> /dev/null; then
    echo "Error: iverilog not found. Install: apt-get install -y iverilog"
    exit 1
fi

uv run python src/eval_verilog.py \
    --adapter "${HF_NAMESPACE}/qwen2.5-7b_mg-verilog-baseline" \
    --problem_file data/verilog_eval/VerilogEval_Human.jsonl \
    --noinclude_grammar \
    --output_path results/verilog_eval/baseline.json \

uv run python src/eval_verilog.py \
    --adapter "${HF_NAMESPACE}/qwen2.5-7b_mg-verilog" \
    --problem_file data/verilog_eval/VerilogEval_Human.jsonl \
    --include_grammar \
    --output_path results/verilog_eval/grammar.json \
