#!/usr/bin/env bash
set -euo pipefail

uv run python src/train.py \
    --train_path data/pddl_depot/train.json \
    --valid_path data/pddl_depot/valid.json \
    --noinclude_grammar \
    --output_dir outputs/qwen2.5-7b-lora-pddl-depot-baseline \
    --hub_model_id "${HF_NAMESPACE}/qwen2.5-7b_pddl-depot-baseline" \
    --max_seq_length 2048

uv run python src/train.py \
    --train_path data/pddl_depot/train.json \
    --valid_path data/pddl_depot/valid.json \
    --output_dir outputs/qwen2.5-7b-lora-pddl-depot \
    --hub_model_id "${HF_NAMESPACE}/qwen2.5-7b_pddl-depot" \
    --max_seq_length 2048
