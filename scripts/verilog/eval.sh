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
    --n_samples 5 \
    --temperature 0.8 \
    --output_path results/verilog/baseline.json

uv run python src/eval_verilog.py \
    --adapter "${HF_NAMESPACE}/qwen2.5-7b_mg-verilog" \
    --problem_file data/verilog_eval/VerilogEval_Human.jsonl \
    --include_grammar \
    --n_samples 5 \
    --temperature 0.8 \
    --output_path results/verilog/grammar.json

uv run python src/plot.py plot_pass_at_k \
    --result_files '["results/verilog/baseline.json", "results/verilog/grammar.json"]' \
    --labels '["Baseline", "Grammar-Guided (Ours)"]' \
    --output_path results/verilog/pass_at_k.png \
    --title "VerilogEval Functional Correctness"
