#!/usr/bin/env bash
set -euo pipefail

for k in 8 16 32; do
    uv run python src/rag_grammar.py predict \
        --test_path data/smcalflow/test.json \
        --train_path data/smcalflow/train.json \
        --k $k \
        --output_path outputs/predicted_grammars/rag/test_k${k}.json

    uv run python src/rag_grammar.py predict \
        --test_path data/smcalflow/test_balanced.json \
        --train_path data/smcalflow/train_balanced.json \
        --k $k \
        --output_path outputs/predicted_grammars/rag/test_balanced_k${k}.json

done
