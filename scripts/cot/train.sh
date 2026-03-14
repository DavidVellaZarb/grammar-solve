#!/usr/bin/env bash
set -euo pipefail

uv run python src/train.py \
    --task grammar_cot \
    --train_path "data/smcalflow/train_balanced_cot.json" \
    --valid_path "data/smcalflow/valid_balanced_cot.json" \
    --output_dir "outputs/qwen2.5-7b-lora-grammar-cot" \
    --hub_model_id "${HF_NAMESPACE}/qwen2.5-7b_smcalflow_balanced-grammar-cot" \
    --max_seq_length 2048 \
    "$@"
