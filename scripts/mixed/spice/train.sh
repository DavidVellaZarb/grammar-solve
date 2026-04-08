#!/usr/bin/env bash
set -euo pipefail

uv run python src/train.py \
    --mixed \
    --train_path data/spice/train.json \
    --valid_path data/spice/valid.json \
    --output_dir "outputs/qwen2.5-7b-lora-spice-mixed" \
    --hub_model_id "${HF_NAMESPACE}/qwen2.5-7b_spice-mixed" \
    --max_seq_length 2048 \
    "$@"
