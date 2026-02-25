#!/usr/bin/env bash
set -euo pipefail

uv run python src/eval.py \
    --adapter "${HF_NAMESPACE}/qwen2.5-7b_mg-verilog-baseline" \
    --test_path data/mg_verilog/test_high_level.json \
    --noinclude_grammar \
    --max_new_tokens 1024 \
    --output_path results/verilog/baseline.json

uv run python src/eval.py \
    --adapter "${HF_NAMESPACE}/qwen2.5-7b_mg-verilog" \
    --test_path data/mg_verilog/test_high_level.json \
    --max_new_tokens 1024 \
    --output_path results/verilog/test.json

uv run python src/plot.py \
    --results_dir results \
    --models '["verilog"]' \
    --model_labels '{"verilog": "Baseline vs Grammar"}' \
    --test_labels '{"baseline": "Without Grammar", "test": "With Grammar (Ours)"}' \
    --output_path results/verilog/baseline_vs_grammar.png
