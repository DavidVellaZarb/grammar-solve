#!/usr/bin/env bash
set -euo pipefail

uv run python src/rag_grammar.py predict \
    --test_path data/smcalflow/test.json \
    --train_path data/smcalflow/train_balanced_cot.json \
    --k 64 \
    --output_path outputs/predicted_grammars/rag_cot/test_k64.json \
    --cache_path cache/rag_cot_cache.json \
    --max_tokens 4096 \
    --prompt_style cot \
    --mode batch \
    "$@"

uv run python src/eval_grammar.py \
    --predicted_path outputs/predicted_grammars/rag_cot/test_k64.json \
    --gold_path data/smcalflow/test.json \
    --write
