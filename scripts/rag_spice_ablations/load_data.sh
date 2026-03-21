#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/spice/ablations

uv run python src/modify_grammar.py \
    --input_path data/spice/train.json \
    --output_path data/spice/ablations/train_add_specific_remove_a.json \
    --grammar_file grammars/spice.lark \
    --operations '["add_specific", "remove"]' \
    --n_ops '[[4, 6], [2, 4]]' \
    --proportion 0.2 \
    --seed 42

uv run python src/modify_grammar.py \
    --input_path data/spice/train.json \
    --output_path data/spice/ablations/train_add_specific_remove_b.json \
    --grammar_file grammars/spice.lark \
    --operations '["add_specific", "remove"]' \
    --n_ops '[[6, 8], [4, 6]]' \
    --proportion 0.2 \
    --seed 42
