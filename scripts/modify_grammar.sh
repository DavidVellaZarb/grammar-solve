#!/usr/bin/env bash
set -euo pipefail

uv run python src/modify_grammar.py \
    --input_path "data/smcalflow/train.json" \
    --output_path "data/smcalflow/train_add_rule.json" \
    --operations '["add"]' --seed 42 "$@"

uv run python src/modify_grammar.py \
    --input_path "data/smcalflow/train.json" \
    --output_path "data/smcalflow/train_remove_rule.json" \
    --operations '["remove"]' --seed 42 "$@"

uv run python src/modify_grammar.py \
    --input_path "data/smcalflow/test.json" \
    --output_path "data/smcalflow/test_add_rule.json" \
    --operations '["add"]' --seed 42 --proportion 1 "$@"

uv run python src/modify_grammar.py \
    --input_path "data/smcalflow/test.json" \
    --output_path "data/smcalflow/test_remove_rule.json" \
    --operations '["remove"]' --seed 42 --proportion 1 "$@"
