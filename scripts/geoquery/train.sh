#!/usr/bin/env bash
set -euo pipefail

uv run python src/train.py \
    --train_path data/geoquery/train.json \
    --valid_path data/geoquery/valid.json \
    --noinclude_grammar \
    --output_dir outputs/qwen2.5-7b-lora-geoquery-baseline \
    --hub_model_id "${HF_NAMESPACE}/qwen2.5-7b_geoquery-baseline" \
    --max_seq_length 1024

uv run python src/train.py \
    --train_path data/geoquery/train.json \
    --valid_path data/geoquery/valid.json \
    --output_dir outputs/qwen2.5-7b-lora-geoquery \
    --hub_model_id "${HF_NAMESPACE}/qwen2.5-7b_geoquery" \
    --max_seq_length 1024

