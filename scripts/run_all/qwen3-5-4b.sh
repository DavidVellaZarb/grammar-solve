#!/usr/bin/env bash
set -euo pipefail

ALIAS=qwen3-5-4b
DOMAINS=(smcalflow geoquery overnight verilog spice)
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

if [[ -f "${REPO_ROOT}/.env" ]]; then
    set -a
    source "${REPO_ROOT}/.env"
    set +a
fi

source "$(dirname "$0")/_lib.sh"

for DOMAIN in "${DOMAINS[@]}"; do
    echo "######## TRAIN ${DOMAIN} (${ALIAS}) ########"
    "${REPO_ROOT}/scripts/${DOMAIN}/${ALIAS}/train.sh" --nosave_locally "$@"
done

for DOMAIN in "${DOMAINS[@]}"; do
    echo "######## EVAL ${DOMAIN} (${ALIAS}) ########"
    "${REPO_ROOT}/scripts/${DOMAIN}/${ALIAS}/eval.sh" "$@"
    sync_path "results/${DOMAIN}/${ALIAS}"
done

echo "######## MULTI-PANEL PLOT (${ALIAS}) ########"
uv run python src/plot_panel.py panel \
    --model_alias "${ALIAS}" \
    --output_path "outputs/analysis/${ALIAS}_panel.png"
sync_path "outputs/analysis"

stop_pod
