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

FAILURES=()

run_step() {
    local LABEL=$1
    shift

    echo "######## ${LABEL} ########"
    if "$@"; then
        echo "######## ${LABEL}: OK ########"
    else
        local STATUS=$?
        echo "######## ${LABEL}: FAILED (${STATUS}) ########"
        FAILURES+=("${LABEL} (${STATUS})")
    fi
}

for DOMAIN in "${DOMAINS[@]}"; do
    run_step "LOAD ${DOMAIN}" uv run python -m "smoke_test.${DOMAIN}.load"
done

for DOMAIN in "${DOMAINS[@]}"; do
    run_step \
        "TRAIN ${DOMAIN} (${MODEL_ALIAS})" \
        "${REPO_ROOT}/smoke_test/${DOMAIN}/train.sh" "${MODEL_NAME}" "${MODEL_ALIAS}" "$@"
done

for DOMAIN in "${DOMAINS[@]}"; do
    run_step \
        "EVAL ${DOMAIN} (${MODEL_ALIAS})" \
        "${REPO_ROOT}/smoke_test/${DOMAIN}/eval.sh" "${MODEL_NAME}" "${MODEL_ALIAS}"
done

if (( ${#FAILURES[@]} > 0 )); then
    echo "######## FAILURES ########"
    printf '%s\n' "${FAILURES[@]}"
    exit 1
fi

echo "######## ALL STEPS COMPLETED ########"
