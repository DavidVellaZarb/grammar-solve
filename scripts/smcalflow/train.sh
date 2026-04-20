#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME=$1
MODEL_ALIAS=$2
shift 2

TRAIN_PATH=data/smcalflow/train.json
VALID_PATH=data/smcalflow/valid.json

HUB_ID="${HF_NAMESPACE}/${MODEL_ALIAS}_smcalflow-baseline"
uv run python src/train.py \
    --model_name "$MODEL_NAME" \
    --noinclude_grammar \
    --num_train_epochs 1 \
    --train_path "$TRAIN_PATH" \
    --valid_path "$VALID_PATH" \
    --output_dir "outputs/${MODEL_ALIAS}-lora-smcalflow-baseline" \
    --hub_model_id "$HUB_ID" \
    "$@"

HUB_ID="${HF_NAMESPACE}/${MODEL_ALIAS}_smcalflow-mixed-r0.1"
uv run python src/train.py \
    --model_name "$MODEL_NAME" \
    --mixed_ratio 0.1 \
    --num_train_epochs 1 \
    --train_path "$TRAIN_PATH" \
    --valid_path "$VALID_PATH" \
    --output_dir "outputs/${MODEL_ALIAS}-lora-smcalflow-mixed-r0.1" \
    --hub_model_id "$HUB_ID" \
    "$@"
