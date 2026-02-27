#!/usr/bin/env bash
set -euo pipefail

uv run python src/train.py \
    --train_path data/mg_verilog/train_detailed.json \
    --valid_path data/mg_verilog/valid_detailed.json \
    --noinclude_grammar \
    --output_dir outputs/qwen2.5-7b-lora-verilog-baseline \
    --hub_model_id "${HF_NAMESPACE}/qwen2.5-7b_mg-verilog-baseline" \
    --max_seq_length 2048

uv run python src/train.py \
    --train_path data/mg_verilog/train_detailed.json \
    --valid_path data/mg_verilog/valid_detailed.json \
    --output_dir outputs/qwen2.5-7b-lora-verilog \
    --hub_model_id "${HF_NAMESPACE}/qwen2.5-7b_mg-verilog" \
    --max_seq_length 2048
