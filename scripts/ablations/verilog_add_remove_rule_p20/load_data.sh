#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/mg_verilog/ablations

uv run python src/modify_grammar.py \
    --input_path data/mg_verilog/train_detailed.json \
    --output_path data/mg_verilog/ablations/train_add_remove_rule_p20.json \
    --grammar_file grammars/verilog.lark \
    --operations '["add_remove"]' \
    --seed 42 \
    --proportion 0.2
