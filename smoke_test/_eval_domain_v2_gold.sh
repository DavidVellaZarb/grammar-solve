#!/usr/bin/env bash
set -euo pipefail

DOMAIN=$1
MODEL_NAME=$2
MODEL_ALIAS=$3
MAX_NEW_TOKENS=$4

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

DATA_ROOT="${SMOKE_TEST_V2_DATA_ROOT:-data/smoke_test_v2}"
OUTPUT_ROOT="${SMOKE_TEST_V2_OUTPUT_ROOT:-outputs/smoke_test_v2}"
RESULT_ROOT="${SMOKE_TEST_V2_RESULT_ROOT:-results/smoke_test_v2}"
TEST_PATH="${DATA_ROOT}/${DOMAIN}/test.json"
RESULT_DIR="${RESULT_ROOT}/${MODEL_ALIAS}/${DOMAIN}"
GOLD_ADAPTER="${OUTPUT_ROOT}/${MODEL_ALIAS}/${DOMAIN}/gold"

if [[ ! -f "${TEST_PATH}" ]]; then
    echo "Missing ${DOMAIN} v2 test data. Run: uv run python -m smoke_test.${DOMAIN}.load --output_dir ${DATA_ROOT}/${DOMAIN} --specialize_terminals"
    exit 1
fi

echo "=== ${DOMAIN}: eval v2 gold grammar (${MODEL_ALIAS}) ==="
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
