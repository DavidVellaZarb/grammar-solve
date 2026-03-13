#!/usr/bin/env bash
set -euo pipefail

uv run python src/train.py \
    --train_path data/smiles/train.json \
    --valid_path data/smiles/valid.json \
    --noinclude_grammar \
    --output_dir outputs/qwen2.5-7b-lora-smiles-baseline \
    --hub_model_id "${HF_NAMESPACE}/qwen2.5-7b_smiles-baseline" \
    --max_seq_length 1024
uv run python src/train.py \
    --train_path data/smiles/train.json \
    --valid_path data/smiles/valid.json \
    --output_dir outputs/qwen2.5-7b-lora-smiles \
    --hub_model_id "${HF_NAMESPACE}/qwen2.5-7b_smiles" \
    --max_seq_length 1024