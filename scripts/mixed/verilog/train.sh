#!/usr/bin/env bash
set -euo pipefail

uv run python src/train.py \
    --mixed \
    --train_path data/mg_verilog/train_detailed.json \
    --valid_path data/mg_verilog/valid_detailed.json \
    --output_dir "outputs/qwen2.5-7b-lora-verilog-mixed" \
    --hub_model_id "${HF_NAMESPACE}/qwen2.5-7b_mg-verilog-mixed" \
    --max_seq_length 2048 \
    "$@"
