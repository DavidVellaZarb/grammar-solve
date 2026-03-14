#!/usr/bin/env bash
set -euo pipefail

uv run python src/train.py \
    --train_path data/spice/train.json \
    --valid_path data/spice/valid.json \
    --noinclude_grammar \
    --output_dir outputs/qwen2.5-7b-lora-spice-baseline \
    --hub_model_id "${HF_NAMESPACE}/qwen2.5-7b_spice-baseline" \
    --max_seq_length 2048

uv run python src/train.py \
    --train_path data/spice/train.json \
    --valid_path data/spice/valid.json \
    --output_dir outputs/qwen2.5-7b-lora-spice \
    --hub_model_id "${HF_NAMESPACE}/qwen2.5-7b_spice" \
    --max_seq_length 2048
