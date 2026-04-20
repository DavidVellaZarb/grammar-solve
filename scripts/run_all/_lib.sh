#!/usr/bin/env bash

HF_RESULTS_REPO="${HF_NAMESPACE}/grammar-solve-results"
WORKSPACE_DIR="/workspace"

sync_path() {
    local rel="$1"
    if [[ ! -d "$rel" ]]; then
        echo "sync_path: skipping '$rel' (not a directory)"
        return 0
    fi

    uv run python -c "
from huggingface_hub import HfApi
import os
HfApi(token=os.getenv('HF_TOKEN')).upload_folder(
    repo_id='${HF_RESULTS_REPO}',
    repo_type='dataset',
    folder_path='${rel}',
    path_in_repo='${rel}',
)
" || echo "warning: HF upload failed for '${rel}'"

    if [[ -d "$WORKSPACE_DIR" ]]; then
        mkdir -p "${WORKSPACE_DIR}/$(dirname "$rel")" \
            && cp -r "$rel" "${WORKSPACE_DIR}/${rel}" \
            || echo "warning: workspace copy failed for '${rel}'"
    else
        echo "warning: ${WORKSPACE_DIR} not present — skipping workspace fallback for '${rel}'"
    fi
}

stop_pod() {
    if [[ -z "${RUNPOD_POD_ID:-}" ]]; then
        echo "stop_pod: RUNPOD_POD_ID not set — not on a Runpod, skipping"
        return 0
    fi
    echo "stop_pod: terminating ${RUNPOD_POD_ID}"
    runpodctl remove pod "$RUNPOD_POD_ID"
}
