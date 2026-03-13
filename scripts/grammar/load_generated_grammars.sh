#!/usr/bin/env bash
set -euo pipefail

uv run python src/generate_grammar.py \
    --adapter "${HF_NAMESPACE}/qwen2.5-7b_smcalflow-grammar" \
    --output_path "outputs/predicted_grammars/generative.json"
