#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME=$1
MODEL_ALIAS=$2

model_exists() {
    uv run python -c "from huggingface_hub import repo_exists; print(repo_exists('$1', repo_type='model'))" 2>/dev/null | grep -q "True"
}

TRAIN_PATH=data/geoquery/train.json
VALID_PATH=data/geoquery/valid.json
TEST_PATH=data/geoquery/test.json
RAG_FILE=outputs/predicted_grammars/rag_cot/geoquery_test_k64.json
RESULT_DIR="results/geoquery/${MODEL_ALIAS}/mixed_sweep"

for RATIO in 0.0 0.1 0.2 0.3; do
    HUB_ID="${HF_NAMESPACE}/${MODEL_ALIAS}_geoquery-mixed-r${RATIO}"

    if model_exists "$HUB_ID"; then
        echo "SKIP $HUB_ID (exists)"
    else
        echo "=== Train ratio=${RATIO} ==="
        uv run python src/train.py \
            --model_name "$MODEL_NAME" \
            --mixed_ratio "$RATIO" \
            --num_train_epochs 1 \
            --train_path "$TRAIN_PATH" \
            --valid_path "$VALID_PATH" \
            --output_dir "outputs/${MODEL_ALIAS}-lora-geoquery-mixed-r${RATIO}" \
            --hub_model_id "$HUB_ID"
    fi

    echo "=== Eval ratio=${RATIO} (RAG grammar) ==="
    uv run python src/eval_geoquery.py \
        --adapter "$HUB_ID" \
        --test_path "$TEST_PATH" \
        --include_grammar \
        --grammar_file "$RAG_FILE" \
        --output_path "${RESULT_DIR}/rag_r${RATIO}.json"
done

echo "=== Plotting ==="
uv run python src/plot.py plot_paper_results \
    --result_files "[\"${RESULT_DIR}/rag_r0.0.json\", \"${RESULT_DIR}/rag_r0.1.json\", \"${RESULT_DIR}/rag_r0.2.json\", \"${RESULT_DIR}/rag_r0.3.json\"]" \
    --labels '["r=0.0", "r=0.1", "r=0.2", "r=0.3"]' \
    --metrics '["accuracy", "execution_accuracy"]' \
    --metric_labels '{"accuracy": "Exact Match", "execution_accuracy": "Execution Accuracy"}' \
    --per_example_fields '{"accuracy": "match", "execution_accuracy": "execution_match"}' \
    --output_path "${RESULT_DIR}/comparison.png" \
    --title "GeoQuery mixed-ratio sweep (${MODEL_ALIAS})"
