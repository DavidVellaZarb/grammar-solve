#!/usr/bin/env bash
set -euo pipefail

MODEL_ALIAS="${MODEL_ALIAS:-qwen3-4b}"
RESULTS_ROOT="${RESULTS_ROOT:-results/smoke_test}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/analysis/smoke_test}"
DOMAINS=(text_to_sql sparql graphql vega_lite vhdl restricted_graphics selfies)
METHODS=(baseline gold)

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

OUTPUT_PATH="${OUTPUT_PATH:-${OUTPUT_DIR}/${MODEL_ALIAS}_baseline_vs_gold_all_metrics.png}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_DIR}/${MODEL_ALIAS}_baseline_vs_gold_all_metrics.json}"
MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO_ROOT}/outputs/.matplotlib}"
export MPLCONFIGDIR

missing=()
for domain in "${DOMAINS[@]}"; do
    for method in "${METHODS[@]}"; do
        path="${RESULTS_ROOT}/${MODEL_ALIAS}/${domain}/${method}.json"
        if [[ ! -f "${path}" ]]; then
            missing+=("${path}")
        fi
    done
done

if (( ${#missing[@]} > 0 )); then
    echo "Missing smoke-test result files:"
    printf '  %s\n' "${missing[@]}"
    echo
    echo "Run the smoke-test evals first, then rerun this script."
    exit 1
fi

mkdir -p "${OUTPUT_DIR}" "${MPLCONFIGDIR}"

echo "######## SMOKE METRICS (${MODEL_ALIAS}) ########"
uv run python -m smoke_test.plot_results \
    --model_alias "${MODEL_ALIAS}" \
    --results_root "${RESULTS_ROOT}" \
    --metric_set all \
    --output_path "${OUTPUT_PATH}" \
    --summary_path "${SUMMARY_PATH}"

echo "######## SMOKE METRICS: OK ########"
echo "Plot:    ${OUTPUT_PATH}"
echo "Summary: ${SUMMARY_PATH}"
