#!/usr/bin/env bash
set -euo pipefail

for k in 8 16 32 64 128; do
    uv run python src/rag_grammar.py \
        --test_path data/smcalflow/test.json \
        --train_path data/smcalflow/train.json \
        --k $k \
        --output_path outputs/predicted_grammars/rag_ablated/test_k${k}.json \

    uv run python src/eval_grammar.py \
        --predicted_path outputs/predicted_grammars/rag_ablated/test_k${k}.json \
        --gold_path data/smcalflow/test.json \
        --write
done
