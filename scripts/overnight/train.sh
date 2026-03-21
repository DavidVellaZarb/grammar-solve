#!/usr/bin/env bash
set -euo pipefail

uv run python src/train.py \
    --train_path data/overnight/train.json \
    --valid_path data/overnight/valid.json \
    --noinclude_grammar \
    --output_dir outputs/qwen2.5-7b-lora-overnight-baseline \
    --hub_model_id "${HF_NAMESPACE}/qwen2.5-7b_overnight-baseline" \
    --max_seq_length 1024 \
    "$@"

uv run python src/train.py \
    --train_path data/overnight/train.json \
    --valid_path data/overnight/valid.json \
    --output_dir outputs/qwen2.5-7b-lora-overnight \
    --hub_model_id "${HF_NAMESPACE}/qwen2.5-7b_overnight" \
    --max_seq_length 1024 \
    "$@"
