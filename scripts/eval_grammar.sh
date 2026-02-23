#!/usr/bin/env bash
set -euo pipefail

uv run python src/eval.py \
    --adapter "${HF_NAMESPACE}/qwen2.5-7b_smcalflow" \
    --grammar_file "outputs/predicted_grammars/generative.json" \
    --output_path "results/grammar_model/test.json" \
    "$@"
