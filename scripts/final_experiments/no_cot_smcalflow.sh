#!/usr/bin/env bash
set -euo pipefail

MODEL_ALIAS="qwen3-5-9b"
DOMAIN="smcalflow"

TEST_PATH="data/smcalflow/test.json"
GRAMMAR_FILE="outputs/predicted_grammars/rag/test_k64.json"
RESULT_DIR="results/no-cot/${DOMAIN}/${MODEL_ALIAS}"
OUTPUT_PATH="${RESULT_DIR}/rag.json"

if [[ -f ".env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source ".env"
    set +a
fi

: "${HF_NAMESPACE:?Set HF_NAMESPACE in your environment or .env}"

for path in "$TEST_PATH" "$GRAMMAR_FILE"; do
    if [[ ! -f "$path" ]]; then
        echo "Missing required file: $path" >&2
        exit 1
    fi
done

mkdir -p "$RESULT_DIR"

ADAPTER="${HF_NAMESPACE}/${MODEL_ALIAS}_${DOMAIN}-mixed-r0.1"

echo "=== SMCalFlow RAG eval (no CoT) on ${ADAPTER} ==="
uv run python src/eval.py \
    --adapter "$ADAPTER" \
    --test_path "$TEST_PATH" \
    --include_grammar \
    --grammar_file "$GRAMMAR_FILE" \
    --output_path "$OUTPUT_PATH"

echo "Wrote $OUTPUT_PATH"
