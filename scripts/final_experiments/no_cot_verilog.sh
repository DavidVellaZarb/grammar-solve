#!/usr/bin/env bash
set -euo pipefail

MODEL_ALIAS="qwen3-5-9b"

PROBLEM_FILE="data/verilog_eval/VerilogEval_Human.jsonl"
GRAMMAR_FILE="outputs/predicted_grammars/rag/verilog_test_k64.json"
RESULT_DIR="results/no-cot/verilog/${MODEL_ALIAS}"
OUTPUT_PATH="${RESULT_DIR}/rag.json"

if [[ -f ".env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source ".env"
    set +a
fi

: "${HF_NAMESPACE:?Set HF_NAMESPACE in your environment or .env}"

if ! command -v iverilog &> /dev/null; then
    echo "Error: iverilog not found. Install: apt-get install -y iverilog" >&2
    exit 1
fi

for path in "$PROBLEM_FILE" "$GRAMMAR_FILE"; do
    if [[ ! -f "$path" ]]; then
        echo "Missing required file: $path" >&2
        exit 1
    fi
done

mkdir -p "$RESULT_DIR"

ADAPTER="${HF_NAMESPACE}/${MODEL_ALIAS}_mg-verilog-mixed-r0.1"

echo "=== Verilog RAG eval (no CoT) on ${ADAPTER} ==="
uv run python src/eval_verilog.py \
    --adapter "$ADAPTER" \
    --problem_file "$PROBLEM_FILE" \
    --include_grammar \
    --grammar_file "$GRAMMAR_FILE" \
    --n_samples 5 \
    --temperature 0.8 \
    --output_path "$OUTPUT_PATH"

echo "Wrote $OUTPUT_PATH"
