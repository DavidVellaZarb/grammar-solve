#!/usr/bin/env bash
set -euo pipefail

uv run python src/train.py \
    --train_path data/mg_verilog/ablations/train_add_remove_rule_p20.json \
    --valid_path data/mg_verilog/valid_detailed.json \
    --output_dir outputs/qwen2.5-7b-lora-verilog-add-remove-rule-p20 \
    --hub_model_id "${HF_NAMESPACE}/qwen2.5-7b_mg-verilog-add-remove-rule-p20" \
    --max_seq_length 2048
