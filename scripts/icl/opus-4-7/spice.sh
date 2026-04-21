#!/usr/bin/env bash
set -euo pipefail

if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

MODEL="claude-opus-4-7"
MODEL_ALIAS="opus-4-7"
K=16
POLL=60

TEST="data/spice/test.json"
TRAIN_PLAIN="data/spice/train.json"
TRAIN_COT="data/spice/train_cot.json"
PRED_GRAMMAR="outputs/predicted_grammars/rag_cot/spice_test_k64.json"

OUT_DIR="outputs/icl/${MODEL_ALIAS}/spice"
RES_DIR="results/icl/${MODEL_ALIAS}/spice"
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
            --mode "$mode" --dataset spice \
            --test_path "$TEST" --train_path "$train_path" \
            --predicted_grammar_path "$PRED_GRAMMAR" \
            --output_path "$pred" \
            --model "$MODEL" --api anthropic --k "$K" \
            --max_tokens 8192
    else
        uv run python src/icl.py submit \
            --mode "$mode" --dataset spice \
            --test_path "$TEST" --train_path "$train_path" \
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
            --mode "$mode" --dataset spice \
            --test_path "$TEST" --train_path "$train_path" \
            --predicted_grammar_path "$PRED_GRAMMAR" \
            --output_path "$pred" \
            --model "$MODEL" --api anthropic --k "$K" \
            --poll_interval "$POLL"
    else
        uv run python src/icl.py collect \
            --mode "$mode" --dataset spice \
            --test_path "$TEST" --train_path "$train_path" \
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
            --dataset spice \
            --predictions_path "$pred" \
            --output_path "$res"
    else
        echo "--- eval: $mode (results already present) ---"
    fi
done

echo "=== Plotting ==="
uv run python src/icl.py plot \
    --dataset spice \
    --results_dir "$RES_DIR" \
    --output_path "${RES_DIR}/comparison.png" \
    --title "SPICE — ${MODEL_ALIAS} (k=${K})"

echo "=== Done ==="
