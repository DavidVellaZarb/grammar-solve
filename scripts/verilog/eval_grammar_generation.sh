#!/usr/bin/env bash
set -euo pipefail

if ! command -v iverilog &> /dev/null; then
    echo "Error: iverilog not found. Install: apt-get install -y iverilog"
    exit 1
fi

uv run python src/eval_verilog.py \
    --adapter "${HF_NAMESPACE}/qwen2.5-7b_mg-verilog" \
    --problem_file data/verilog/VerilogEval_Human.jsonl \
    --include_grammar \
    --grammar_file "outputs/predicted_grammars/verilog_generative.json" \
    --n_samples 5 \
    --temperature 0.8 \
    --output_path results/verilog/grammar_generation.json
