#!/usr/bin/env bash
set -euo pipefail

GOLD_GENERIC="data/smcalflow/test_generic.json"
GOLD_SPECIALIZED="data/smcalflow/test.json"

for THRESHOLD in 0.3 0.5 0.7; do
    echo "=== Threshold: $THRESHOLD ==="

    GENERIC_OUT="outputs/predicted_grammars/classifier_t${THRESHOLD}_generic.json"
    SPECIALIZED_OUT="outputs/predicted_grammars/classifier_t${THRESHOLD}_specialized.json"

    uv run python src/classifier.py predict \
        --threshold "$THRESHOLD" \
        --output_path "$GENERIC_OUT"

    uv run python src/specialize_grammar.py \
        --test_path "$GENERIC_OUT" \
        --output_path "$SPECIALIZED_OUT"

    uv run python src/eval_grammar.py \
        --predicted_path "$GENERIC_OUT" \
        --gold_path "$GOLD_GENERIC" \
        --write

    uv run python src/eval_grammar.py \
        --predicted_path "$SPECIALIZED_OUT" \
        --gold_path "$GOLD_SPECIALIZED" \
        --write
done
