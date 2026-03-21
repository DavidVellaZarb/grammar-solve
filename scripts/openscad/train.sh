#!/usr/bin/env bash
set -euo pipefail

uv run python src/train.py \
    --train_path data/openscad/train.json \
    --valid_path data/openscad/valid.json \
    --noinclude_grammar \
    --output_dir outputs/qwen2.5-7b-lora-openscad-baseline \
    --hub_model_id "${HF_NAMESPACE}/qwen2.5-7b_openscad-baseline" \
    --max_seq_length 2048

uv run python src/train.py \
    --train_path data/openscad/train.json \
    --valid_path data/openscad/valid.json \
    --output_dir outputs/qwen2.5-7b-lora-openscad \
    --hub_model_id "${HF_NAMESPACE}/qwen2.5-7b_openscad" \
    --max_seq_length 2048

