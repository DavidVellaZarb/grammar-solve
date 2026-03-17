#!/usr/bin/env bash
set -euo pipefail

uv run python src/rag_grammar.py predict \
    --test_path data/spice/test.json \
    --train_path data/spice/train.json \
    --grammar_path grammars/spice.lark \
    --k 64 \
    --output_path outputs/predicted_grammars/rag/spice_test_k64.json \
    --cache_path cache/rag_spice_cache.json \
    --mode batch
