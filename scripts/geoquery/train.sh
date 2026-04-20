#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME=$1
MODEL_ALIAS=$2
shift 2

model_exists() {
    uv run python -c "from huggingface_hub import repo_exists; print(repo_exists('$1', repo_type='model'))" 2>/dev/null | grep -q "True"
}

TRAIN_PATH=data/geoquery/train.json
VALID_PATH=data/geoquery/valid.json

HUB_ID="${HF_NAMESPACE}/${MODEL_ALIAS}_geoquery-baseline"
if model_exists "$HUB_ID"; then
    echo "SKIP $HUB_ID (exists)"
else
    uv run python src/train.py \
        --model_name "$MODEL_NAME" \
        --noinclude_grammar \
        --num_train_epochs 1 \
        --train_path "$TRAIN_PATH" \
        --valid_path "$VALID_PATH" \
        --output_dir "outputs/${MODEL_ALIAS}-lora-geoquery-baseline" \
        --hub_model_id "$HUB_ID" \
        "$@"
fi

HUB_ID="${HF_NAMESPACE}/${MODEL_ALIAS}_geoquery-mixed-r0.1"
if model_exists "$HUB_ID"; then
    echo "SKIP $HUB_ID (exists)"
else
    uv run python src/train.py \
        --model_name "$MODEL_NAME" \
        --mixed_ratio 0.1 \
        --num_train_epochs 1 \
        --train_path "$TRAIN_PATH" \
        --valid_path "$VALID_PATH" \
        --output_dir "outputs/${MODEL_ALIAS}-lora-geoquery-mixed-r0.1" \
        --hub_model_id "$HUB_ID" \
        "$@"
fi
