#!/usr/bin/env bash
set -euo pipefail

uv run python src/train.py \
    --mixed \
    --train_path data/smiles/train.json \
    --valid_path data/smiles/valid.json \
    --output_dir "outputs/qwen2.5-7b-lora-smiles-mixed" \
    --hub_model_id "${HF_NAMESPACE}/qwen2.5-7b_smiles-mixed" \
    --max_seq_length 1024 \
    "$@"
