#!/usr/bin/env bash
set -euo pipefail

uv run python src/rag_grammar.py \
    --test_path data/openscad/test.json \
    --train_path data/openscad/train.json \
    --grammar_path grammars/openscad.lark \
    --k 64 \
    --output_path outputs/predicted_grammars/rag/openscad_test_k64.json \
    --cache_path cache/rag_openscad_cache.json
