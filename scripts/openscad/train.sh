#!/usr/bin/env bash
set -euo pipefail

model_exists() {
    uv run python -c "from huggingface_hub import repo_exists; print(repo_exists('$1', repo_type='model'))" 2>/dev/null | grep -q "True"
}

TRAIN_PATH=data/openscad/train.json
VALID_PATH=data/openscad/valid.json

HUB_ID="${HF_NAMESPACE}/qwen2.5-7b_openscad-baseline"
if model_exists "$HUB_ID"; then
    echo "SKIP $HUB_ID (exists)"
else
    uv run python src/train.py \
        --noinclude_grammar \
        --num_train_epochs 1 \
        --train_path "$TRAIN_PATH" \
        --valid_path "$VALID_PATH" \
        --output_dir outputs/qwen2.5-7b-lora-openscad-baseline \
        --hub_model_id "$HUB_ID" \
        --max_seq_length 2048
fi

HUB_ID="${HF_NAMESPACE}/qwen2.5-7b_openscad-baseline-2epoch"
if model_exists "$HUB_ID"; then
    echo "SKIP $HUB_ID (exists)"
else
    uv run python src/train.py \
        --noinclude_grammar \
        --num_train_epochs 2 \
        --train_path "$TRAIN_PATH" \
        --valid_path "$VALID_PATH" \
        --output_dir outputs/qwen2.5-7b-lora-openscad-baseline-2epoch \
        --hub_model_id "$HUB_ID" \
        --max_seq_length 2048
fi

HUB_ID="${HF_NAMESPACE}/qwen2.5-7b_openscad-mixed"
if model_exists "$HUB_ID"; then
    echo "SKIP $HUB_ID (exists)"
else
    uv run python src/train.py \
        --mixed \
        --train_path "$TRAIN_PATH" \
        --valid_path "$VALID_PATH" \
        --output_dir outputs/qwen2.5-7b-lora-openscad-mixed \
        --hub_model_id "$HUB_ID" \
        --max_seq_length 2048
fi
