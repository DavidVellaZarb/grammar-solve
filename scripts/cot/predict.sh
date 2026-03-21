#!/usr/bin/env bash
set -euo pipefail

uv run python src/generate_grammar.py \
    --adapter "${HF_NAMESPACE}/qwen2.5-7b_smcalflow_balanced-grammar-cot" \
    --test_path "data/smcalflow/test_balanced.json" \
    --output_path "outputs/predicted_grammars/balanced_cot.json" \
    --task grammar_cot \
    --max_new_tokens 1024

