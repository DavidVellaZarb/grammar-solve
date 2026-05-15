#!/usr/bin/env bash
set -euo pipefail

uv sync
MAX_JOBS=8 uv pip install flash-attn --no-build-isolation

ALIAS=gemma-3-12b
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
    "${REPO_ROOT}/scripts/${DOMAIN}/${ALIAS}/train.sh" --nosave_locally
done

for DOMAIN in "${DOMAINS[@]}"; do
    echo "######## EVAL ${DOMAIN} (${ALIAS}) ########"
    "${REPO_ROOT}/scripts/${DOMAIN}/${ALIAS}/eval.sh"
    sync_path "results/${DOMAIN}/${ALIAS}"
done

echo "######## PER-MODEL MULTI-PANEL PLOT (${ALIAS}) ########"
uv run python src/plot_per_model.py per_model --model_alias "${ALIAS}"
sync_path "outputs/analysis/models/${ALIAS}"

stop_pod
