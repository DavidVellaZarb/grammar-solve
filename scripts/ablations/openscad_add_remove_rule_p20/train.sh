#!/usr/bin/env bash
set -euo pipefail

uv run python src/train.py \
    --train_path data/openscad/ablations/train_add_remove_rule_p20.json \
    --valid_path data/openscad/valid.json \
    --output_dir outputs/qwen2.5-7b-lora-openscad-add-remove-rule-p20 \
    --hub_model_id "${HF_NAMESPACE}/qwen2.5-7b_openscad-add-remove-rule-p20" \
    --max_seq_length 2048
