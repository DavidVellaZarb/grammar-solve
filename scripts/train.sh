#!/usr/bin/env bash
set -euo pipefail
uv run python src/train.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --train_path "data/smcalflow/train.json" \
    --valid_path "data/smcalflow/valid.json" \
    --output_dir "outputs/qwen2.5-7b-lora" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --lora_r 64 \
    --lora_alpha 128 \
    --max_seq_length 1024 \
    "$@"
