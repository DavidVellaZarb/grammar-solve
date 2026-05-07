#!/usr/bin/env bash
set -euo pipefail

DOMAIN=$1
MODEL_NAME=$2
MODEL_ALIAS=$3
MAX_SEQ_LENGTH=$4
shift 4

TRAIN_PATH="data/smoke_test/${DOMAIN}/train.json"
VALID_PATH="data/smoke_test/${DOMAIN}/valid.json"
TRAIN_EVAL_STRATEGY="${EVAL_STRATEGY:-steps}"

if [[ "${DOMAIN}" == "restricted_graphics" && -z "${EVAL_STRATEGY:-}" ]]; then
    TRAIN_EVAL_STRATEGY="no"
fi

if [[ ! -f "${TRAIN_PATH}" || ! -f "${VALID_PATH}" ]]; then
    echo "Missing ${DOMAIN} data. Run: uv run python -m smoke_test.${DOMAIN}.load"
    exit 1
fi

COMMON_ARGS=(
    --model_name "${MODEL_NAME}"
    --num_train_epochs "${NUM_TRAIN_EPOCHS:-1}"
    --train_path "${TRAIN_PATH}"
    --valid_path "${VALID_PATH}"
    --max_seq_length "${MAX_SEQ_LENGTH}"
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-8}"
    --eval_strategy "${TRAIN_EVAL_STRATEGY}"
    --eval_steps "${EVAL_STEPS:-100}"
    --save_steps "${SAVE_STEPS:-100}"
    --save_total_limit "${SAVE_TOTAL_LIMIT:-1}"
    --report_to "${REPORT_TO:-none}"
    --nopush_to_hub
    --attn_implementation "${ATTN_IMPLEMENTATION:-flash_attention_2}"
)

if [[ -n "${MAX_STEPS:-}" ]]; then
    COMMON_ARGS+=(--max_steps "${MAX_STEPS}")
fi

echo "=== ${DOMAIN}: baseline (${MODEL_ALIAS}) ==="
uv run python src/train.py \
    "${COMMON_ARGS[@]}" \
    --noinclude_grammar \
    --output_dir "outputs/smoke_test/${MODEL_ALIAS}/${DOMAIN}/baseline" \
    "$@"

echo "=== ${DOMAIN}: gold grammar (${MODEL_ALIAS}) ==="
uv run python src/train.py \
    "${COMMON_ARGS[@]}" \
    --include_grammar \
    --output_dir "outputs/smoke_test/${MODEL_ALIAS}/${DOMAIN}/gold" \
    "$@"
