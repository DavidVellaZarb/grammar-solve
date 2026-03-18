#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/openscad/ablations

uv run python src/modify_grammar.py \
    --input_path data/openscad/train.json \
    --output_path data/openscad/ablations/train_add_remove_rule_p20.json \
    --grammar_file grammars/openscad.lark \
    --operations '["add", "remove"]' \
    --seed 42 \
    --proportion 0.2 \
    --balanced
