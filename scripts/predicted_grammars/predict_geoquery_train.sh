#!/usr/bin/env bash
set -euo pipefail

uv run python src/rag_grammar.py predict \
    --test_path data/geoquery/train.json \
    --train_path data/geoquery/train_cot.json \
    --grammar_path grammars/geoquery.lark \
    --k 64 \
    --output_path outputs/predicted_grammars/rag_cot/geoquery_train_k64.json \
    --cache_path cache/rag_geoquery_train_cache.json \
    --max_tokens 4096 \
    --prompt_style cot \
    --exclude_self \
    --mode batch
