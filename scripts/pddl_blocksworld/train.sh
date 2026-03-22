#!/usr/bin/env bash
set -euo pipefail

uv run python src/train.py \
    --train_path data/pddl_blocksworld/train.json \
    --valid_path data/pddl_blocksworld/valid.json \
    --noinclude_grammar \
    --output_dir outputs/qwen2.5-7b-lora-pddl-blocksworld-baseline \
    --hub_model_id "${HF_NAMESPACE}/qwen2.5-7b_pddl-blocksworld-baseline" \
    --max_seq_length 2048

uv run python src/train.py \
    --train_path data/pddl_blocksworld/train.json \
    --valid_path data/pddl_blocksworld/valid.json \
    --output_dir outputs/qwen2.5-7b-lora-pddl-blocksworld \
    --hub_model_id "${HF_NAMESPACE}/qwen2.5-7b_pddl-blocksworld" \
    --max_seq_length 2048
