#!/usr/bin/env bash
set -euo pipefail

uv run python src/train.py \
    --mixed \
    --output_dir "outputs/qwen2.5-7b-lora-mixed" \
    --hub_model_id "${HF_NAMESPACE}/qwen2.5-7b_smcalflow-mixed" \
    "$@"
