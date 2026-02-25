#!/usr/bin/env bash
set -euo pipefail

uv run python src/train.py \
    --noinclude_grammar \
    --output_dir "outputs/qwen2.5-7b-lora-baseline" \
    --hub_model_id "${HF_NAMESPACE}/qwen2.5-7b_smcalflow-baseline" \
    "$@"

uv run python src/train.py \
    --output_dir "outputs/qwen2.5-7b-lora" \
    "$@"

uv run python src/train.py \
    --task grammar_program \
    --output_dir "outputs/qwen2.5-7b-lora-grammar-program" \
    --hub_model_id "${HF_NAMESPACE}/qwen2.5-7b_smcalflow-grammar-program" \
    --max_seq_length 2048 \
    "$@"
