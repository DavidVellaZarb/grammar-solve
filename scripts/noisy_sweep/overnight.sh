#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
MODEL_ALIAS="llama-3.1-8b"

model_exists() {
    uv run python -c "from huggingface_hub import repo_exists; print(repo_exists('$1', repo_type='model'))" 2>/dev/null | grep -q "True"
}

TRAIN_PATH=data/overnight/train.json
VALID_PATH=data/overnight/valid.json
TEST_PATH=data/overnight/test.json
GRAMMAR_FILE=grammars/overnight_blocks.lark
RAG_FILE=outputs/predicted_grammars/rag_cot/overnight_test_k64.json
RESULT_DIR="results/overnight/${MODEL_ALIAS}/noisy_sweep"
NOISY_DATA_DIR="data/overnight/noisy_sweep"
SEED=42

mkdir -p "$NOISY_DATA_DIR" "$RESULT_DIR"

BASE_TAG="p0.0"
BASE_HUB="${HF_NAMESPACE}/${MODEL_ALIAS}_overnight-noisy-${BASE_TAG}"
if model_exists "$BASE_HUB"; then
    echo "SKIP $BASE_HUB (exists)"
else
    echo "=== Train baseline (unmodified gold) ==="
    uv run python src/train.py \
        --model_name "$MODEL_NAME" \
        --num_train_epochs 1 \
        --train_path "$TRAIN_PATH" \
        --valid_path "$VALID_PATH" \
        --output_dir "outputs/${MODEL_ALIAS}-lora-overnight-noisy-${BASE_TAG}" \
        --hub_model_id "$BASE_HUB"
fi

echo "=== Eval baseline (RAG grammar) ==="
uv run python src/eval_overnight.py \
    --adapter "$BASE_HUB" \
    --test_path "$TEST_PATH" \
    --include_grammar \
    --grammar_file "$RAG_FILE" \
    --output_path "${RESULT_DIR}/${BASE_TAG}.json"

for RANGE in "2 5" "3 6" "4 7"; do
    set -- $RANGE
    LO=$1; HI=$2
    for PROP in 0.1 0.2 0.3; do
        TAG="r${LO}_${HI}-p${PROP}"
        NOISY_TRAIN="${NOISY_DATA_DIR}/train_${TAG}.json"
        NOISY_VALID="${NOISY_DATA_DIR}/valid_${TAG}.json"
        HUB_ID="${HF_NAMESPACE}/${MODEL_ALIAS}_overnight-noisy-${TAG}"

        uv run python src/modify_grammar.py \
            --input_path "$TRAIN_PATH" \
            --output_path "$NOISY_TRAIN" \
            --operations '["add", "remove"]' \
            --proportion "$PROP" \
            --n_ops "[${LO}, ${HI}]" \
            --grammar_file "$GRAMMAR_FILE" \
            --seed "$SEED"
        uv run python src/modify_grammar.py \
            --input_path "$VALID_PATH" \
            --output_path "$NOISY_VALID" \
            --operations '["add", "remove"]' \
            --proportion "$PROP" \
            --n_ops "[${LO}, ${HI}]" \
            --grammar_file "$GRAMMAR_FILE" \
            --seed "$SEED"

        if model_exists "$HUB_ID"; then
            echo "SKIP $HUB_ID (exists)"
        else
            echo "=== Train ${TAG} ==="
            uv run python src/train.py \
                --model_name "$MODEL_NAME" \
                --num_train_epochs 1 \
                --train_path "$NOISY_TRAIN" \
                --valid_path "$NOISY_VALID" \
                --output_dir "outputs/${MODEL_ALIAS}-lora-overnight-noisy-${TAG}" \
                --hub_model_id "$HUB_ID"
        fi

        echo "=== Eval ${TAG} (RAG grammar) ==="
        uv run python src/eval_overnight.py \
            --adapter "$HUB_ID" \
            --test_path "$TEST_PATH" \
            --include_grammar \
            --grammar_file "$RAG_FILE" \
            --output_path "${RESULT_DIR}/${TAG}.json"
    done
done

echo "=== Plotting ==="
for PROP in 0.1 0.2 0.3; do
    FILES="[\"${RESULT_DIR}/p0.0.json\""
    LABELS='["Gold (p=0.0)"'
    for RANGE in "2 5" "3 6" "4 7"; do
        set -- $RANGE
        LO=$1; HI=$2
        FILES="${FILES}, \"${RESULT_DIR}/r${LO}_${HI}-p${PROP}.json\""
        LABELS="${LABELS}, \"[${LO},${HI})\""
    done
    FILES="${FILES}]"
    LABELS="${LABELS}]"

    uv run python src/plot.py plot_paper_results \
        --result_files "$FILES" \
        --labels "$LABELS" \
        --metrics '["accuracy", "execution_accuracy"]' \
        --metric_labels '{"accuracy": "Exact Match", "execution_accuracy": "Execution Accuracy"}' \
        --per_example_fields '{"accuracy": "match", "execution_accuracy": "execution_match"}' \
        --output_path "${RESULT_DIR}/comparison_p${PROP}.png" \
        --title "Overnight-Blocks noisy-sweep (${MODEL_ALIAS}), proportion=${PROP}"
done
