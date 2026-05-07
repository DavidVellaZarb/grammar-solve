#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="Qwen/Qwen3-4B-Instruct-2507"
MODEL_ALIAS="qwen3-4b"
LOAD_TRAIN_EVAL_DOMAINS=(sparql graphql vega_lite)
TRAIN_EVAL_DOMAINS=(restricted_graphics)
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [[ -f "${REPO_ROOT}/.env" ]]; then
    set -a
    source "${REPO_ROOT}/.env"
    set +a
fi

FAILURES=()
SKIPS=()
LOAD_OK=()
TRAIN_OK=()

has_domain() {
    local NEEDLE=$1
    shift
    local ITEM
    for ITEM in "$@"; do
        if [[ "${ITEM}" == "${NEEDLE}" ]]; then
            return 0
        fi
    done
    return 1
}

run_step() {
    local LABEL=$1
    shift

    echo "######## ${LABEL} ########"
    if "$@"; then
        echo "######## ${LABEL}: OK ########"
        return 0
    else
        local STATUS=$?
        echo "######## ${LABEL}: FAILED (${STATUS}) ########"
        FAILURES+=("${LABEL} (${STATUS})")
        return "${STATUS}"
    fi
}

skip_step() {
    local LABEL=$1
    local REASON=$2
    echo "######## ${LABEL}: SKIPPED (${REASON}) ########"
    SKIPS+=("${LABEL}: ${REASON}")
}

reset_adapters() {
    local DOMAIN=$1
    local OUTPUT_DIR="${REPO_ROOT}/outputs/smoke_test/${MODEL_ALIAS}/${DOMAIN}"
    if [[ "${RESET_FAILED_ADAPTERS:-1}" == "1" ]]; then
        rm -rf "${OUTPUT_DIR}/baseline" "${OUTPUT_DIR}/gold"
    fi
}

for DOMAIN in "${LOAD_TRAIN_EVAL_DOMAINS[@]}"; do
    if run_step "LOAD ${DOMAIN}" uv run python -m "smoke_test.${DOMAIN}.load"; then
        LOAD_OK+=("${DOMAIN}")
    fi
done

for DOMAIN in "${LOAD_TRAIN_EVAL_DOMAINS[@]}"; do
    if ! has_domain "${DOMAIN}" "${LOAD_OK[@]}"; then
        skip_step "TRAIN ${DOMAIN} (${MODEL_ALIAS})" "load failed"
        continue
    fi
    reset_adapters "${DOMAIN}"
    if run_step \
        "TRAIN ${DOMAIN} (${MODEL_ALIAS})" \
        "${REPO_ROOT}/smoke_test/${DOMAIN}/train.sh" "${MODEL_NAME}" "${MODEL_ALIAS}" "$@"; then
        TRAIN_OK+=("${DOMAIN}")
    fi
done

for DOMAIN in "${TRAIN_EVAL_DOMAINS[@]}"; do
    reset_adapters "${DOMAIN}"
    if run_step \
        "TRAIN ${DOMAIN} (${MODEL_ALIAS})" \
        "${REPO_ROOT}/smoke_test/${DOMAIN}/train.sh" "${MODEL_NAME}" "${MODEL_ALIAS}" "$@"; then
        TRAIN_OK+=("${DOMAIN}")
    fi
done

for DOMAIN in "${LOAD_TRAIN_EVAL_DOMAINS[@]}" "${TRAIN_EVAL_DOMAINS[@]}"; do
    if ! has_domain "${DOMAIN}" "${TRAIN_OK[@]}"; then
        skip_step "EVAL ${DOMAIN} (${MODEL_ALIAS})" "train failed or skipped"
        continue
    fi
    if ! run_step \
        "EVAL ${DOMAIN} (${MODEL_ALIAS})" \
        "${REPO_ROOT}/smoke_test/${DOMAIN}/eval.sh" "${MODEL_NAME}" "${MODEL_ALIAS}"; then
        :
    fi
done

if ! run_step \
    "PLOT ${MODEL_ALIAS}" \
    uv run python -m smoke_test.plot_results \
        --model_alias "${MODEL_ALIAS}" \
        --output_path "outputs/analysis/smoke_test/${MODEL_ALIAS}_baseline_vs_gold.png"; then
    :
fi

if (( ${#SKIPS[@]} > 0 )); then
    echo "######## SKIPS ########"
    printf '%s\n' "${SKIPS[@]}"
fi

if (( ${#FAILURES[@]} > 0 )); then
    echo "######## FAILURES ########"
    printf '%s\n' "${FAILURES[@]}"
    exit 1
fi

echo "######## FAILED-DOMAIN RERUN COMPLETED ########"
