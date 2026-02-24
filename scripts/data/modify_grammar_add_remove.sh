#!/usr/bin/env bash
set -euo pipefail

# --- Fixed n_ops ---

for N in 1 2 3 4; do
    uv run python src/modify_grammar.py \
        --input_path "data/smcalflow/train.json" \
        --output_path "data/smcalflow/train_add_remove_n${N}_p20.json" \
        --operations '["add_remove"]' --seed 42 --proportion 0.2 --n_ops "$N" "$@"

    uv run python src/modify_grammar.py \
        --input_path "data/smcalflow/test.json" \
        --output_path "data/smcalflow/test_add_remove_n${N}.json" \
        --operations '["add_remove"]' --seed 42 --proportion 1 --n_ops "$N" "$@"
done

# --- Interval n_ops ---

for RANGE in "2,4" "3,5" "4,6"; do
    LOW="${RANGE%,*}"
    HIGH="${RANGE#*,}"
    LABEL="${LOW}-${HIGH}"

    uv run python src/modify_grammar.py \
        --input_path "data/smcalflow/train.json" \
        --output_path "data/smcalflow/train_add_remove_n${LABEL}_p20.json" \
        --operations '["add_remove"]' --seed 42 --proportion 0.2 --n_ops "[$LOW,$HIGH]" "$@"

    uv run python src/modify_grammar.py \
        --input_path "data/smcalflow/test.json" \
        --output_path "data/smcalflow/test_add_remove_n${LABEL}.json" \
        --operations '["add_remove"]' --seed 42 --proportion 1 --n_ops "[$LOW,$HIGH]" "$@"
done
