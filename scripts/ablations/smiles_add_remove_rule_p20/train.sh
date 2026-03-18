#!/usr/bin/env bash
set -euo pipefail

uv run python src/train.py \
    --train_path data/smiles/ablations/train_add_remove_rule_p20.json \
    --valid_path data/smiles/valid.json \
    --output_dir outputs/qwen2.5-7b-lora-smiles-add-remove-rule-p20 \
    --hub_model_id "${HF_NAMESPACE}/qwen2.5-7b_smiles-add-remove-rule-p20" \
    --max_seq_length 1024
