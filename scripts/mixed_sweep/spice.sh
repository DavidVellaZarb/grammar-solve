#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
MODEL_ALIAS="llama-3.1-8b"

TRAIN_PATH=data/spice/train.json
VALID_PATH=data/spice/valid.json
TEST_PATH=data/spice/test.json
RAG_FILE=outputs/predicted_grammars/rag_cot/spice_test_k64.json
RESULT_DIR="results/mixed_sweep/spice"

for RATIO in 0.0 0.1 0.2 0.3; do
    HUB_ID="${HF_NAMESPACE}/${MODEL_ALIAS}_spice-mixed-r${RATIO}"

    echo "=== Train ratio=${RATIO} ==="
    uv run python src/train.py \
        --model_name "$MODEL_NAME" \
        --mixed_ratio "$RATIO" \
        --num_train_epochs 1 \
        --train_path "$TRAIN_PATH" \
        --valid_path "$VALID_PATH" \
        --output_dir "outputs/${MODEL_ALIAS}-lora-spice-mixed-r${RATIO}" \
        --hub_model_id "$HUB_ID" \
        --max_seq_length 2048

    echo "=== Eval ratio=${RATIO} (RAG grammar) ==="
    uv run python src/eval_spice.py \
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
    --metrics '["ged_similarity", "component_f1"]' \
    --metric_labels '{"ged_similarity": "GED Similarity", "component_f1": "Component F1"}' \
    --per_example_fields '{"ged_similarity": "ged_similarity", "component_f1": "component_f1"}' \
    --output_path "${RESULT_DIR}/comparison.png" \
    --title "SPICE mixed-ratio sweep"
