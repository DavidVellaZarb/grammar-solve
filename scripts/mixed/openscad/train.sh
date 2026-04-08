#!/usr/bin/env bash
set -euo pipefail

uv run python src/train.py \
    --mixed \
    --train_path data/openscad/train.json \
    --valid_path data/openscad/valid.json \
    --output_dir "outputs/qwen2.5-7b-lora-openscad-mixed" \
    --hub_model_id "${HF_NAMESPACE}/qwen2.5-7b_openscad-mixed" \
    --max_seq_length 2048 \
    "$@"
