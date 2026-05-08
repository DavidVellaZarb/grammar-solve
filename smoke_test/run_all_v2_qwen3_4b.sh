#!/usr/bin/env bash
set -euo pipefail

DOMAINS=(text_to_sql sparql graphql vega_lite vhdl restricted_graphics selfies)
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

FAILURES=()
SKIPS=()
LOAD_OK=()
TRAIN_OK=()
EVAL_OK=()

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

max_seq_length_for() {
    if [[ -n "${MAX_SEQ_LENGTH:-}" ]]; then
        echo "${MAX_SEQ_LENGTH}"
        return
    fi

    case "$1" in
        vega_lite)
            echo 12288
            ;;
        vhdl | restricted_graphics)
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
        vega_lite | vhdl | restricted_graphics)
            echo 1024
            ;;
        *)
            echo 512
            ;;
    esac
}

reset_gold_adapter() {
    local DOMAIN=$1
    if [[ "${RESET_V2_ADAPTERS:-1}" == "1" ]]; then
        rm -rf "${OUTPUT_ROOT}/${MODEL_ALIAS}/${DOMAIN}/gold"
    fi
    rm -f "${RESULT_ROOT}/${MODEL_ALIAS}/${DOMAIN}/gold.json"
}

print_summary() {
    if (( ${#EVAL_OK[@]} == 0 )); then
        echo "######## V2 SUMMARY ########"
        echo "No v2 evals completed."
        return
    fi

    uv run python - "${RESULT_ROOT}" "${MODEL_ALIAS}" "${EVAL_OK[@]}" <<'PY'
import json
import sys
from pathlib import Path

result_root = Path(sys.argv[1])
model_alias = sys.argv[2]
domains = sys.argv[3:]

print("######## V2 GOLD SUMMARY ########")
for domain in domains:
    path = result_root / model_alias / domain / "gold.json"
    with path.open() as f:
        data = json.load(f)
    print(
        f"{domain:20s} gold={data['accuracy']:.4f} "
        f"({data['correct']}/{data['total']})"
    )
PY
}

for DOMAIN in "${DOMAINS[@]}"; do
    if run_step \
        "LOAD V2 ${DOMAIN}" \
        uv run python -m "smoke_test.${DOMAIN}.load" \
            --output_dir "${DATA_ROOT}/${DOMAIN}" \
            --specialize_terminals; then
        if run_step \
            "VALIDATE V2 ${DOMAIN}" \
            uv run python -m smoke_test.validate_specialized \
                --data-root "${DATA_ROOT}" \
                "${DOMAIN}"; then
            LOAD_OK+=("${DOMAIN}")
        fi
    fi
done

for DOMAIN in "${DOMAINS[@]}"; do
    if ! has_domain "${DOMAIN}" "${LOAD_OK[@]}"; then
        skip_step "TRAIN V2 ${DOMAIN} (${MODEL_ALIAS})" "load or validation failed"
        continue
    fi

    reset_gold_adapter "${DOMAIN}"
    if run_step \
        "TRAIN V2 ${DOMAIN} (${MODEL_ALIAS})" \
        "${REPO_ROOT}/smoke_test/_train_domain_v2_gold.sh" \
            "${DOMAIN}" \
            "${MODEL_NAME}" \
            "${MODEL_ALIAS}" \
            "$(max_seq_length_for "${DOMAIN}")" \
            "$@"; then
        TRAIN_OK+=("${DOMAIN}")
    fi
done

for DOMAIN in "${DOMAINS[@]}"; do
    if ! has_domain "${DOMAIN}" "${TRAIN_OK[@]}"; then
        skip_step "EVAL V2 ${DOMAIN} (${MODEL_ALIAS})" "train failed or skipped"
        continue
    fi

    rm -f "${RESULT_ROOT}/${MODEL_ALIAS}/${DOMAIN}/gold.json"
    if run_step \
        "EVAL V2 ${DOMAIN} (${MODEL_ALIAS})" \
        "${REPO_ROOT}/smoke_test/_eval_domain_v2_gold.sh" \
            "${DOMAIN}" \
            "${MODEL_NAME}" \
            "${MODEL_ALIAS}" \
            "$(max_new_tokens_for "${DOMAIN}")"; then
        EVAL_OK+=("${DOMAIN}")
    fi
done

print_summary

if (( ${#SKIPS[@]} > 0 )); then
    echo "######## SKIPS ########"
    printf '%s\n' "${SKIPS[@]}"
fi

if (( ${#FAILURES[@]} > 0 )); then
    echo "######## FAILURES ########"
    printf '%s\n' "${FAILURES[@]}"
    exit 1
fi

echo "######## SMOKE TEST V2 COMPLETED ########"
