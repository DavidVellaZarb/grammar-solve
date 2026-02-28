#!/usr/bin/env bash
set -euo pipefail

uv run python src/generate_grammar.py \
    --adapter "${HF_NAMESPACE}/qwen2.5-7b_mg-verilog-grammar" \
    --test_path data/verilog_eval/VerilogEval_Human.jsonl \
    --output_path "outputs/predicted_grammars/verilog_generative.json"
