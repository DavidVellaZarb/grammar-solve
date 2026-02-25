#!/usr/bin/env bash
set -euo pipefail

uv run python src/modify_grammar.py \
    --input_path "data/smcalflow/train.json" \
    --output_path "data/smcalflow/ablations/train_add_rule_p10.json" \
    --operations '["add"]' --seed 42 "$@"

uv run python src/modify_grammar.py \
    --input_path "data/smcalflow/train.json" \
    --output_path "data/smcalflow/ablations/train_remove_rule_p10.json" \
    --operations '["remove"]' --seed 42 "$@"

uv run python src/modify_grammar.py \
    --input_path "data/smcalflow/test.json" \
    --output_path "data/smcalflow/ablations/test_add_rule.json" \
    --operations '["add"]' --seed 42 --proportion 1 "$@"

uv run python src/modify_grammar.py \
    --input_path "data/smcalflow/test.json" \
    --output_path "data/smcalflow/ablations/test_remove_rule.json" \
    --operations '["remove"]' --seed 42 --proportion 1 "$@"

uv run python src/modify_grammar.py \
    --input_path "data/smcalflow/train.json" \
    --output_path "data/smcalflow/ablations/train_add_remove_rule_p10.json" \
    --operations '["add", "remove"]' --seed 42 --balanced "$@"

uv run python src/modify_grammar.py \
    --input_path "data/smcalflow/test.json" \
    --output_path "data/smcalflow/ablations/test_add_remove_rule.json" \
    --operations '["add", "remove"]' --seed 42 --proportion 1 --balanced "$@"

uv run python src/modify_grammar.py \
    --input_path "data/smcalflow/train.json" \
    --output_path "data/smcalflow/ablations/train_add_rule_p20.json" \
    --operations '["add"]' --seed 42 --proportion 0.2 "$@"

uv run python src/modify_grammar.py \
    --input_path "data/smcalflow/train.json" \
    --output_path "data/smcalflow/ablations/train_remove_rule_p20.json" \
    --operations '["remove"]' --seed 42 --proportion 0.2 "$@"

uv run python src/modify_grammar.py \
    --input_path "data/smcalflow/train.json" \
    --output_path "data/smcalflow/ablations/train_add_remove_rule_p20.json" \
    --operations '["add", "remove"]' --seed 42 --proportion 0.2 --balanced "$@"
