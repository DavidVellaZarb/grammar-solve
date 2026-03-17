#!/usr/bin/env bash
set -euo pipefail

uv run python src/rag_grammar.py predict \
    --test_path data/smiles/test.json \
    --train_path data/smiles/train.json \
    --grammar_path grammars/smiles.lark \
    --k 64 \
    --output_path outputs/predicted_grammars/rag/smiles_test_k64.json \
    --cache_path cache/rag_smiles_cache.json \
    --mode batch
