#!/usr/bin/env bash
set -euo pipefail

DOMAIN=$1
MODEL_NAME=$2
MODEL_ALIAS=$3
MAX_NEW_TOKENS=$4

TEST_PATH="data/smoke_test/${DOMAIN}/test.json"
RESULT_DIR="results/smoke_test/${MODEL_ALIAS}/${DOMAIN}"
BASELINE_ADAPTER="outputs/smoke_test/${MODEL_ALIAS}/${DOMAIN}/baseline"
GOLD_ADAPTER="outputs/smoke_test/${MODEL_ALIAS}/${DOMAIN}/gold"

if [[ ! -f "${TEST_PATH}" ]]; then
    echo "Missing ${DOMAIN} test data. Run: uv run python -m smoke_test.${DOMAIN}.load"
    exit 1
fi

echo "=== ${DOMAIN}: eval baseline (${MODEL_ALIAS}) ==="
uv run python -m smoke_test.eval \
    --adapter "${BASELINE_ADAPTER}" \
    --model_name "${MODEL_NAME}" \
    --test_path "${TEST_PATH}" \
    --domain "${DOMAIN}" \
    --noinclude_grammar \
    --batch_size "${EVAL_BATCH_SIZE:-16}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --attn_implementation "${ATTN_IMPLEMENTATION:-flash_attention_2}" \
    --output_path "${RESULT_DIR}/baseline.json"

echo "=== ${DOMAIN}: eval gold grammar (${MODEL_ALIAS}) ==="
uv run python -m smoke_test.eval \
    --adapter "${GOLD_ADAPTER}" \
    --model_name "${MODEL_NAME}" \
    --test_path "${TEST_PATH}" \
    --domain "${DOMAIN}" \
    --include_grammar \
    --batch_size "${EVAL_BATCH_SIZE:-16}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --attn_implementation "${ATTN_IMPLEMENTATION:-flash_attention_2}" \
    --output_path "${RESULT_DIR}/gold.json"
