#!/usr/bin/env bash
# Evaluate the SMCalFlow Qwen3.5-9B parser against open-weight RAG-CoT
# grammars (predicted by predict_smcalflow_local.sh on the same pod, or
# previously uploaded to the HF dataset repo). Uses the local predictions
# file if present; otherwise falls back to downloading from HF. Syncs the
# result folder to HF and /workspace/. Does NOT stop the pod when finished.
set -euo pipefail

# shellcheck source=scripts/run_all/_lib.sh
source "scripts/run_all/_lib.sh"

if [[ -f ".env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source ".env"
    set +a
fi

: "${HF_NAMESPACE:?Set HF_NAMESPACE in your environment or .env}"
: "${HF_TOKEN:?Set HF_TOKEN in your environment or .env}"

MODEL_ALIAS="${MODEL_ALIAS:-qwen3-5-9b}"
DOMAIN="smcalflow"
K="${K:-64}"
RATIO="${RATIO:-0.1}"

ADAPTER="${HF_NAMESPACE}/${MODEL_ALIAS}_${DOMAIN}-mixed-r${RATIO}"
PRED_REL="outputs/predicted_grammars/openweight_rag/${DOMAIN}_test_k${K}_${MODEL_ALIAS}.json"
RESULT_REL="results/openweight_rag/${DOMAIN}/${MODEL_ALIAS}"
RESULT_FILE="${RESULT_REL}/rag_r${RATIO}.json"

mkdir -p "$(dirname "$PRED_REL")" "$RESULT_REL"

if [[ -f "$PRED_REL" ]]; then
    echo "=== Using local predictions: $PRED_REL ==="
else
    echo "=== Fetching predictions from HF dataset repo ==="
    uv run python - "$PRED_REL" <<'PY'
import os
import shutil
import sys

from huggingface_hub import hf_hub_download

pred_rel = sys.argv[1]
local = hf_hub_download(
    repo_id=f"{os.environ['HF_NAMESPACE']}/grammar-solve-results",
    repo_type="dataset",
    filename=pred_rel,
    token=os.environ["HF_TOKEN"],
)
os.makedirs(os.path.dirname(pred_rel), exist_ok=True)
shutil.copy(local, pred_rel)
print(f"Downloaded predictions to {pred_rel}")
PY
fi

echo "=== Eval ${ADAPTER} with open-weight RAG grammars ==="
uv run python src/eval.py \
    --adapter "$ADAPTER" \
    --test_path "data/${DOMAIN}/test.json" \
    --include_grammar \
    --grammar_file "$PRED_REL" \
    --output_path "$RESULT_FILE"

sync_path "$RESULT_REL"

echo "Done. Result: $RESULT_FILE (pod left running)."
