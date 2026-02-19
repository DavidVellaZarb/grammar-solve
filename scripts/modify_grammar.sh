#!/usr/bin/env bash
set -euo pipefail

for split in train test; do
    uv run python src/modify_grammar.py \
        --input_path "data/smcalflow/${split}.json" \
        --output_path "data/smcalflow/${split}_add_rule.json" \
        --operations '["add"]' --seed 42 "$@"

    uv run python src/modify_grammar.py \
        --input_path "data/smcalflow/${split}.json" \
        --output_path "data/smcalflow/${split}_remove_rule.json" \
        --operations '["remove"]' --seed 42 "$@"
done
