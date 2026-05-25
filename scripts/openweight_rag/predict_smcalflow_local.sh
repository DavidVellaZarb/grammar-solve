#!/usr/bin/env bash
# Predict SMCalFlow test-set grammars locally on the pod using an open-weight
# Qwen3.5-9B-Instruct model via HuggingFace transformers (RAG-CoT). Uploads the
# predictions to the HF dataset repo and mirrors to /workspace/ for persistence.
set -euo pipefail

if [[ -f ".env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source ".env"
    set +a
fi

: "${HF_NAMESPACE:?Set HF_NAMESPACE in your environment or .env}"
: "${HF_TOKEN:?Set HF_TOKEN in your environment or .env}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-9B-Instruct}"
MODEL_ALIAS="${MODEL_ALIAS:-qwen3-5-9b}"
DOMAIN="smcalflow"
K="${K:-64}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"

OUT_DIR="outputs/predicted_grammars/openweight_rag"
OUT_FILE="${OUT_DIR}/${DOMAIN}_test_k${K}_${MODEL_ALIAS}.json"
CACHE_FILE="cache/openweight_rag_${DOMAIN}_${MODEL_ALIAS}.json"

mkdir -p "$OUT_DIR" cache

uv run python src/rag_grammar_local.py predict \
    --test_path "data/${DOMAIN}/test.json" \
    --train_path "data/${DOMAIN}/train.json" \
    --grammar_path "grammars/${DOMAIN}.lark" \
    --output_path "$OUT_FILE" \
    --cache_path "$CACHE_FILE" \
    --model "$MODEL_NAME" \
    --k "$K" \
    --prompt_style cot \
    --batch_size "$BATCH_SIZE" \
    --max_new_tokens "$MAX_NEW_TOKENS"

echo "=== Uploading ${OUT_FILE} to HF dataset repo ==="
uv run python - "$OUT_FILE" <<'PY'
import os
import sys

from huggingface_hub import HfApi

out_file = sys.argv[1]
HfApi(token=os.environ["HF_TOKEN"]).upload_file(
    repo_id=f"{os.environ['HF_NAMESPACE']}/grammar-solve-results",
    repo_type="dataset",
    path_or_fileobj=out_file,
    path_in_repo=out_file,
)
print(f"Uploaded {out_file}")
PY

if [[ -d /workspace ]]; then
    mkdir -p "/workspace/$(dirname "$OUT_FILE")"
    cp "$OUT_FILE" "/workspace/$OUT_FILE"
    echo "Copied to /workspace/$OUT_FILE"
fi

echo "Done. Predictions at: $OUT_FILE"
