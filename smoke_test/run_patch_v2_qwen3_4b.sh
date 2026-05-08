#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f "${REPO_ROOT}/.env" ]]; then
    set -a
    source "${REPO_ROOT}/.env"
    set +a
fi

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507}"
MODEL_ALIAS="${MODEL_ALIAS:-qwen3-4b}"
DATA_ROOT="${SMOKE_TEST_V2_DATA_ROOT:-data/smoke_test_v2}"
OUTPUT_ROOT="${SMOKE_TEST_V2_OUTPUT_ROOT:-outputs/smoke_test_v2}"
RESULT_ROOT="${SMOKE_TEST_V2_RESULT_ROOT:-results/smoke_test_v2}"
ANALYSIS_DIR="${SMOKE_TEST_V2_ANALYSIS_DIR:-outputs/analysis/smoke_test_v2}"
MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO_ROOT}/outputs/.matplotlib}"

export MPLCONFIGDIR
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

PATCH_DOMAINS=(vega_lite vhdl)
FAILURES=()
TRAIN_OK=()
EVAL_OK=()

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

max_seq_length_for() {
    if [[ -n "${MAX_SEQ_LENGTH:-}" ]]; then
        echo "${MAX_SEQ_LENGTH}"
        return
    fi

    case "$1" in
        vega_lite)
            echo 12288
            ;;
        vhdl)
            echo 16384
            ;;
        *)
            echo 4096
            ;;
    esac
}

max_new_tokens_for() {
    if [[ -n "${MAX_NEW_TOKENS:-}" ]]; then
        echo "${MAX_NEW_TOKENS}"
        return
    fi

    case "$1" in
        vega_lite | vhdl)
            echo 1024
            ;;
        *)
            echo 512
            ;;
    esac
}

ensure_v2_data() {
    local DOMAIN=$1
    local DOMAIN_DIR="${DATA_ROOT}/${DOMAIN}"

    if [[ ! -f "${DOMAIN_DIR}/train.json" || ! -f "${DOMAIN_DIR}/valid.json" || ! -f "${DOMAIN_DIR}/test.json" ]]; then
        run_step \
            "LOAD V2 ${DOMAIN}" \
            uv run python -m "smoke_test.${DOMAIN}.load" \
                --output_dir "${DOMAIN_DIR}" \
                --specialize_terminals || return $?
    fi

    run_step \
        "VALIDATE V2 ${DOMAIN}" \
        uv run python -m smoke_test.validate_specialized \
            --data-root "${DATA_ROOT}" \
            "${DOMAIN}"
}

reset_failed_domain_outputs() {
    local DOMAIN=$1
    if [[ "${RESET_V2_PATCH_ADAPTERS:-1}" == "1" ]]; then
        rm -rf "${OUTPUT_ROOT}/${MODEL_ALIAS}/${DOMAIN}/gold"
    fi
    rm -f "${RESULT_ROOT}/${MODEL_ALIAS}/${DOMAIN}/gold.json"
}

train_v2_domain() {
    local DOMAIN=$1
    shift
    reset_failed_domain_outputs "${DOMAIN}"

    (
        export EVAL_STRATEGY="${PATCH_TRAIN_EVAL_STRATEGY:-no}"
        "${REPO_ROOT}/smoke_test/_train_domain_v2_gold.sh" \
            "${DOMAIN}" \
            "${MODEL_NAME}" \
            "${MODEL_ALIAS}" \
            "$(max_seq_length_for "${DOMAIN}")" \
            "$@"
    )
}

eval_v2_domain() {
    local DOMAIN=$1

    (
        export EVAL_BATCH_SIZE="${PATCH_EVAL_BATCH_SIZE:-${EVAL_BATCH_SIZE:-4}}"
        "${REPO_ROOT}/smoke_test/_eval_domain_v2_gold.sh" \
            "${DOMAIN}" \
            "${MODEL_NAME}" \
            "${MODEL_ALIAS}" \
            "$(max_new_tokens_for "${DOMAIN}")"
    )
}

plot_v2_exact() {
    mkdir -p "${ANALYSIS_DIR}" "${MPLCONFIGDIR}"
    uv run python -m smoke_test.plot_results \
        --model_alias "${MODEL_ALIAS}" \
        --results_root "${RESULT_ROOT}" \
        --methods gold \
        --output_path "${ANALYSIS_DIR}/${MODEL_ALIAS}_v2_gold_multipanel.png"
}

plot_v2_all_metrics() {
    mkdir -p "${ANALYSIS_DIR}" "${MPLCONFIGDIR}"
    uv run python -m smoke_test.plot_results \
        --model_alias "${MODEL_ALIAS}" \
        --results_root "${RESULT_ROOT}" \
        --methods gold \
        --metric_set all \
        --output_path "${ANALYSIS_DIR}/${MODEL_ALIAS}_v2_gold_all_metrics_multipanel.png" \
        --summary_path "${ANALYSIS_DIR}/${MODEL_ALIAS}_v2_gold_all_metrics.json"
}

for DOMAIN in "${PATCH_DOMAINS[@]}"; do
    if ! ensure_v2_data "${DOMAIN}"; then
        continue
    fi

    if run_step \
        "TRAIN V2 PATCH ${DOMAIN} (${MODEL_ALIAS})" \
        train_v2_domain "${DOMAIN}" "$@"; then
        TRAIN_OK+=("${DOMAIN}")
    fi
done

for DOMAIN in "${PATCH_DOMAINS[@]}"; do
    if ! has_domain "${DOMAIN}" "${TRAIN_OK[@]}"; then
        echo "######## EVAL V2 PATCH ${DOMAIN} (${MODEL_ALIAS}): SKIPPED (train failed) ########"
        continue
    fi

    if run_step \
        "EVAL V2 PATCH ${DOMAIN} (${MODEL_ALIAS})" \
        eval_v2_domain "${DOMAIN}"; then
        EVAL_OK+=("${DOMAIN}")
    fi
done

if run_step "PLOT V2 GOLD MULTIPANEL ${MODEL_ALIAS}" plot_v2_exact; then
    :
fi

if run_step "PLOT V2 GOLD ALL-METRICS MULTIPANEL ${MODEL_ALIAS}" plot_v2_all_metrics; then
    :
fi

if (( ${#FAILURES[@]} > 0 )); then
    echo "######## FAILURES ########"
    printf '%s\n' "${FAILURES[@]}"
    exit 1
fi

echo "######## V2 PATCH RERUN COMPLETED ########"
echo "Exact plot:       ${ANALYSIS_DIR}/${MODEL_ALIAS}_v2_gold_multipanel.png"
echo "All-metrics plot: ${ANALYSIS_DIR}/${MODEL_ALIAS}_v2_gold_all_metrics_multipanel.png"
