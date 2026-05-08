#!/usr/bin/env bash
set -euo pipefail

DOMAIN=$1
MODEL_NAME=$2
MODEL_ALIAS=$3
MAX_SEQ_LENGTH=$4
shift 4

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

DATA_ROOT="${SMOKE_TEST_V2_DATA_ROOT:-data/smoke_test_v2}"
OUTPUT_ROOT="${SMOKE_TEST_V2_OUTPUT_ROOT:-outputs/smoke_test_v2}"
TRAIN_PATH="${DATA_ROOT}/${DOMAIN}/train.json"
VALID_PATH="${DATA_ROOT}/${DOMAIN}/valid.json"
TRAIN_EVAL_STRATEGY="${EVAL_STRATEGY:-steps}"

if [[ "${DOMAIN}" == "restricted_graphics" && -z "${EVAL_STRATEGY:-}" ]]; then
    TRAIN_EVAL_STRATEGY="no"
fi

if [[ ! -f "${TRAIN_PATH}" || ! -f "${VALID_PATH}" ]]; then
    echo "Missing ${DOMAIN} v2 data. Run: uv run python -m smoke_test.${DOMAIN}.load --output_dir ${DATA_ROOT}/${DOMAIN} --specialize_terminals"
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

echo "=== ${DOMAIN}: v2 gold grammar (${MODEL_ALIAS}) ==="
uv run python src/train.py \
    "${COMMON_ARGS[@]}" \
    --include_grammar \
    --output_dir "${OUTPUT_ROOT}/${MODEL_ALIAS}/${DOMAIN}/gold" \
    "$@"
