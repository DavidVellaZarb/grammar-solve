#!/usr/bin/env bash
set -euo pipefail

uv run python src/train.py \
    --train_path data/pddl_satellite/train.json \
    --valid_path data/pddl_satellite/valid.json \
    --noinclude_grammar \
    --output_dir outputs/qwen2.5-7b-lora-pddl-satellite-baseline \
    --hub_model_id "${HF_NAMESPACE}/qwen2.5-7b_pddl-satellite-baseline" \
    --max_seq_length 2048

uv run python src/train.py \
    --train_path data/pddl_satellite/train.json \
    --valid_path data/pddl_satellite/valid.json \
    --output_dir outputs/qwen2.5-7b-lora-pddl-satellite \
    --hub_model_id "${HF_NAMESPACE}/qwen2.5-7b_pddl-satellite" \
    --max_seq_length 2048
