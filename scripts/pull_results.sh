#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

STAGE_DIR="$(mktemp -d)"
trap 'rm -rf "$STAGE_DIR"' EXIT

uv run python -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id=f\"{os.environ['HF_NAMESPACE']}/grammar-solve-results\",
    repo_type='dataset',
    local_dir='${STAGE_DIR}',
    token=os.getenv('HF_TOKEN'),
)
"

for sub in results outputs/analysis; do
    if [[ -d "${STAGE_DIR}/${sub}" ]]; then
        mkdir -p "${sub}"
        cp -r "${STAGE_DIR}/${sub}/." "${sub}/"
        echo "pulled ${sub}"
    fi
done
