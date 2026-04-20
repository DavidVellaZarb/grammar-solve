#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME=$1
MODEL_ALIAS=$2
shift 2

TRAIN_PATH=data/spice/train.json
VALID_PATH=data/spice/valid.json

HUB_ID="${HF_NAMESPACE}/${MODEL_ALIAS}_spice-baseline"
uv run python src/train.py \
    --model_name "$MODEL_NAME" \
    --noinclude_grammar \
    --num_train_epochs 1 \
    --train_path "$TRAIN_PATH" \
    --valid_path "$VALID_PATH" \
    --output_dir "outputs/${MODEL_ALIAS}-lora-spice-baseline" \
    --hub_model_id "$HUB_ID" \
    --max_seq_length 2048 \
    "$@"

HUB_ID="${HF_NAMESPACE}/${MODEL_ALIAS}_spice-mixed-r0.1"
uv run python src/train.py \
    --model_name "$MODEL_NAME" \
    --mixed_ratio 0.1 \
    --num_train_epochs 1 \
    --train_path "$TRAIN_PATH" \
    --valid_path "$VALID_PATH" \
    --output_dir "outputs/${MODEL_ALIAS}-lora-spice-mixed-r0.1" \
    --hub_model_id "$HUB_ID" \
    --max_seq_length 2048 \
    "$@"
