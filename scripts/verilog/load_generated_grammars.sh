#!/usr/bin/env bash
set -euo pipefail

uv run python src/generate_grammar.py \
    --adapter "${HF_NAMESPACE}/qwen2.5-7b_mg-verilog-grammar" \
    --test_path data/mg_verilog/test_detailed.json \
    --output_path "outputs/predicted_grammars/verilog_generative.json"
