#!/usr/bin/env bash
set -euo pipefail

uv run python src/train.py \
    --task grammar \
    --output_dir "outputs/qwen2.5-7b-lora-grammar" \
    --hub_model_id "${HF_NAMESPACE}/qwen2.5-7b_smcalflow-grammar" \
    "$@"
