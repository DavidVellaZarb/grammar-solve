#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME=$1
MODEL_ALIAS=$2

model_exists() {
    uv run python -c "from huggingface_hub import repo_exists; print(repo_exists('$1', repo_type='model'))" 2>/dev/null | grep -q "True"
}

TRAIN_PATH=data/mg_verilog/train_detailed.json
VALID_PATH=data/mg_verilog/valid_detailed.json

HUB_ID="${HF_NAMESPACE}/${MODEL_ALIAS}_mg-verilog-baseline"
if model_exists "$HUB_ID"; then
    echo "SKIP $HUB_ID (exists)"
else
    uv run python src/train.py \
        --model_name "$MODEL_NAME" \
        --noinclude_grammar \
        --num_train_epochs 1 \
        --train_path "$TRAIN_PATH" \
        --valid_path "$VALID_PATH" \
        --output_dir "outputs/${MODEL_ALIAS}-lora-verilog-baseline" \
        --hub_model_id "$HUB_ID" \
        --max_seq_length 2048
fi

HUB_ID="${HF_NAMESPACE}/${MODEL_ALIAS}_mg-verilog-baseline-2epoch"
if model_exists "$HUB_ID"; then
    echo "SKIP $HUB_ID (exists)"
else
    uv run python src/train.py \
        --model_name "$MODEL_NAME" \
        --noinclude_grammar \
        --num_train_epochs 2 \
        --train_path "$TRAIN_PATH" \
        --valid_path "$VALID_PATH" \
        --output_dir "outputs/${MODEL_ALIAS}-lora-verilog-baseline-2epoch" \
        --hub_model_id "$HUB_ID" \
        --max_seq_length 2048
fi

HUB_ID="${HF_NAMESPACE}/${MODEL_ALIAS}_mg-verilog-mixed"
if model_exists "$HUB_ID"; then
    echo "SKIP $HUB_ID (exists)"
else
    uv run python src/train.py \
        --model_name "$MODEL_NAME" \
        --mixed_duplicate \
        --train_path "$TRAIN_PATH" \
        --valid_path "$VALID_PATH" \
        --output_dir "outputs/${MODEL_ALIAS}-lora-verilog-mixed" \
        --hub_model_id "$HUB_ID" \
        --max_seq_length 2048
fi
