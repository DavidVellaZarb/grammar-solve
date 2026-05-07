#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="Qwen/Qwen3-4B-Instruct-2507"
MODEL_ALIAS="qwen3-4b"
DOMAINS=(text_to_sql sparql graphql vega_lite vhdl restricted_graphics selfies)
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [[ -f "${REPO_ROOT}/.env" ]]; then
    set -a
    source "${REPO_ROOT}/.env"
    set +a
fi

for DOMAIN in "${DOMAINS[@]}"; do
    echo "######## LOAD ${DOMAIN} ########"
    uv run python -m "smoke_test.${DOMAIN}.load"
done

for DOMAIN in "${DOMAINS[@]}"; do
    echo "######## TRAIN ${DOMAIN} (${MODEL_ALIAS}) ########"
    "${REPO_ROOT}/smoke_test/${DOMAIN}/train.sh" "${MODEL_NAME}" "${MODEL_ALIAS}" "$@"
done

for DOMAIN in "${DOMAINS[@]}"; do
    echo "######## EVAL ${DOMAIN} (${MODEL_ALIAS}) ########"
    "${REPO_ROOT}/smoke_test/${DOMAIN}/eval.sh" "${MODEL_NAME}" "${MODEL_ALIAS}"
done

