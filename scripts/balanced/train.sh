#!/usr/bin/env bash
set -euo pipefail

uv run python src/train.py \
    --train_path "data/smcalflow/train_balanced.json" \
    --valid_path "data/smcalflow/valid_balanced.json" \
    --noinclude_grammar \
    --output_dir "outputs/qwen2.5-7b-lora-balanced-baseline" \
    --hub_model_id "${HF_NAMESPACE}/qwen2.5-7b_smcalflow_balanced-baseline"

uv run python src/train.py \
    --train_path "data/smcalflow/train_balanced.json" \
    --valid_path "data/smcalflow/valid_balanced.json" \
    --output_dir "outputs/qwen2.5-7b-lora-balanced" \
    --hub_model_id "${HF_NAMESPACE}/qwen2.5-7b_smcalflow_balanced"

uv run python src/train.py \
    --task grammar \
    --train_path "data/smcalflow/train_balanced.json" \
    --valid_path "data/smcalflow/valid_balanced.json" \
    --output_dir "outputs/qwen2.5-7b-lora-balanced-grammar" \
    --hub_model_id "${HF_NAMESPACE}/qwen2.5-7b_smcalflow_balanced-grammar"
