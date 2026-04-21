#!/usr/bin/env bash
set -euo pipefail

if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

if ! command -v iverilog &> /dev/null; then
    echo "Error: iverilog not found. Install: apt-get install -y iverilog"
    exit 1
fi

MODEL="claude-opus-4-7"
MODEL_ALIAS="opus-4-7"
K=16
POLL=60

PROBLEM_FILE="data/verilog_eval/VerilogEval_Human.jsonl"
TRAIN_PLAIN="data/mg_verilog/train_detailed.json"
TRAIN_COT="data/mg_verilog/train_detailed_cot.json"
PRED_GRAMMAR="outputs/predicted_grammars/rag_cot/verilog_test_k64.json"

OUT_DIR="outputs/icl/${MODEL_ALIAS}/verilog"
RES_DIR="results/icl/${MODEL_ALIAS}/verilog"
mkdir -p "$OUT_DIR" "$RES_DIR"

MODES=(zero_shot baseline grammar_first grammar_knn rag_cot rag_cot_with_grammar)

train_path_for() {
    case "$1" in
        zero_shot|baseline|grammar_first|grammar_knn) echo "$TRAIN_PLAIN" ;;
        rag_cot|rag_cot_with_grammar) echo "$TRAIN_COT" ;;
        *) echo "unknown mode: $1" >&2; exit 1 ;;
    esac
}

run_submit() {
    local mode="$1"
    local train_path
    train_path=$(train_path_for "$mode")
    local pred="${OUT_DIR}/${mode}.json"
    if [[ "$mode" == "rag_cot_with_grammar" ]]; then
        uv run python src/icl.py submit \
            --mode "$mode" --dataset verilog \
            --test_path "$PROBLEM_FILE" --train_path "$train_path" \
            --predicted_grammar_path "$PRED_GRAMMAR" \
            --output_path "$pred" \
            --model "$MODEL" --api anthropic --k "$K" \
            --max_tokens 8192
    else
        uv run python src/icl.py submit \
            --mode "$mode" --dataset verilog \
            --test_path "$PROBLEM_FILE" --train_path "$train_path" \
            --output_path "$pred" \
            --model "$MODEL" --api anthropic --k "$K" \
            --max_tokens 8192
    fi
}

run_collect() {
    local mode="$1"
    local train_path
    train_path=$(train_path_for "$mode")
    local pred="${OUT_DIR}/${mode}.json"
    if [[ "$mode" == "rag_cot_with_grammar" ]]; then
        uv run python src/icl.py collect \
            --mode "$mode" --dataset verilog \
            --test_path "$PROBLEM_FILE" --train_path "$train_path" \
            --predicted_grammar_path "$PRED_GRAMMAR" \
            --output_path "$pred" \
            --model "$MODEL" --api anthropic --k "$K" \
            --poll_interval "$POLL"
    else
        uv run python src/icl.py collect \
            --mode "$mode" --dataset verilog \
            --test_path "$PROBLEM_FILE" --train_path "$train_path" \
            --output_path "$pred" \
            --model "$MODEL" --api anthropic --k "$K" \
            --poll_interval "$POLL"
    fi
}

echo "=== Phase 1: submit all batches (non-blocking) ==="
for mode in "${MODES[@]}"; do
    echo "--- submit: $mode ---"
    run_submit "$mode"
done

echo "=== Phase 2: sequential collect + eval ==="
for mode in "${MODES[@]}"; do
    pred="${OUT_DIR}/${mode}.json"
    res="${RES_DIR}/${mode}.json"

    if [[ ! -f "$pred" ]]; then
        echo "--- collect: $mode ---"
        run_collect "$mode"
    else
        echo "--- collect: $mode (predictions already present) ---"
    fi

    if [[ ! -f "$res" ]]; then
        echo "--- eval: $mode ---"
        uv run python src/icl.py eval_predictions \
            --dataset verilog \
            --predictions_path "$pred" \
            --problem_file "$PROBLEM_FILE" \
            --output_path "$res" \
            --k 1
    else
        echo "--- eval: $mode (results already present) ---"
    fi
done

echo "=== Plotting ==="
uv run python src/icl.py plot \
    --dataset verilog \
    --results_dir "$RES_DIR" \
    --output_path "${RES_DIR}/comparison.png" \
    --title "VerilogEval — ${MODEL_ALIAS} (k=${K})"

echo "=== Done ==="
