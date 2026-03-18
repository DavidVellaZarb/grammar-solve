#!/usr/bin/env bash
set -euo pipefail

ADAPTER="${HF_NAMESPACE}/qwen2.5-7b_spice-add-remove-rule-p20"

uv run python src/train.py \
    --train_path data/spice/ablations/train_add_remove_rule_p20.json \
    --valid_path data/spice/valid.json \
    --hub_model_id "$ADAPTER" \
    --max_seq_length 2048
