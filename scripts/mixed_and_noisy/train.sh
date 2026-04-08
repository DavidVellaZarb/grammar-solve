#!/usr/bin/env bash
set -euo pipefail

uv run python src/train.py \
    --mixed \
    --train_path data/smcalflow/ablations/train_add_remove_n3-5_p20.json \
    --valid_path data/smcalflow/valid.json \
    --output_dir "outputs/qwen2.5-7b-lora-mixed-noisy" \
    --hub_model_id "${HF_NAMESPACE}/qwen2.5-7b_smcalflow-mixed-noisy" \
    "$@"
