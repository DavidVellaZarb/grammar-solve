#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME=$1
MODEL_ALIAS=$2

model_exists() {
    uv run python -c "from huggingface_hub import repo_exists; print(repo_exists('$1', repo_type='model'))" 2>/dev/null | grep -q "True"
}

TRAIN_PATH=data/smiles/train.json
VALID_PATH=data/smiles/valid.json

HUB_ID="${HF_NAMESPACE}/${MODEL_ALIAS}_smiles-baseline"
if model_exists "$HUB_ID"; then
    echo "SKIP $HUB_ID (exists)"
else
    uv run python src/train.py \
        --model_name "$MODEL_NAME" \
        --noinclude_grammar \
        --num_train_epochs 1 \
        --train_path "$TRAIN_PATH" \
        --valid_path "$VALID_PATH" \
        --output_dir "outputs/${MODEL_ALIAS}-lora-smiles-baseline" \
        --hub_model_id "$HUB_ID"
fi

HUB_ID="${HF_NAMESPACE}/${MODEL_ALIAS}_smiles-baseline-2epoch"
if model_exists "$HUB_ID"; then
    echo "SKIP $HUB_ID (exists)"
else
    uv run python src/train.py \
        --model_name "$MODEL_NAME" \
        --noinclude_grammar \
        --num_train_epochs 2 \
        --train_path "$TRAIN_PATH" \
        --valid_path "$VALID_PATH" \
        --output_dir "outputs/${MODEL_ALIAS}-lora-smiles-baseline-2epoch" \
        --hub_model_id "$HUB_ID"
fi

HUB_ID="${HF_NAMESPACE}/${MODEL_ALIAS}_smiles-mixed"
if model_exists "$HUB_ID"; then
    echo "SKIP $HUB_ID (exists)"
else
    uv run python src/train.py \
        --model_name "$MODEL_NAME" \
        --mixed \
        --train_path "$TRAIN_PATH" \
        --valid_path "$VALID_PATH" \
        --output_dir "outputs/${MODEL_ALIAS}-lora-smiles-mixed" \
        --hub_model_id "$HUB_ID"
fi
