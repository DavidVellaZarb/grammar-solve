#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/spice/ablations

uv run python src/modify_grammar.py \
    --input_path data/spice/train.json \
    --output_path data/spice/ablations/train_add_remove_rule_p20.json \
    --grammar_file grammars/spice.lark \
    --operations '["add", "remove"]' \
    --seed 42 \
    --proportion 0.2 \
    --balanced
