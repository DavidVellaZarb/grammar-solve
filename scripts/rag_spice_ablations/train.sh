#!/usr/bin/env bash
set -euo pipefail

uv run python src/train.py \
    --train_path data/spice/ablations/train_add_specific_remove_a.json \
    --valid_path data/spice/valid.json \
    --hub_model_id "${HF_NAMESPACE}/qwen2.5-7b_spice-add-specific-remove-a" \
    --max_seq_length 2048

uv run python src/train.py \
    --train_path data/spice/ablations/train_add_specific_remove_b.json \
    --valid_path data/spice/valid.json \
    --hub_model_id "${HF_NAMESPACE}/qwen2.5-7b_spice-add-specific-remove-b" \
    --max_seq_length 2048
