#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
MODEL_ALIAS="llama-3.1-8b"

if ! command -v iverilog &> /dev/null; then
    echo "Error: iverilog not found. Install: apt-get install -y iverilog"
    exit 1
fi

TRAIN_PATH=data/mg_verilog/train_detailed.json
VALID_PATH=data/mg_verilog/valid_detailed.json
PROBLEM_FILE=data/verilog_eval/VerilogEval_Human.jsonl
RAG_FILE=outputs/predicted_grammars/rag_cot/verilog_test_k64.json
RESULT_DIR="results/mixed_sweep/verilog"

model_exists() {
    uv run python -c "from huggingface_hub import repo_exists; print(repo_exists('$1', repo_type='model'))" 2>/dev/null | grep -q "True"
}

for RATIO in 0.0 0.1 0.2 0.3; do
    HUB_ID="${HF_NAMESPACE}/${MODEL_ALIAS}_mg-verilog-mixed-r${RATIO}"

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
            --output_dir "outputs/${MODEL_ALIAS}-lora-verilog-mixed-r${RATIO}" \
            --hub_model_id "$HUB_ID" \
            --max_seq_length 2048
    fi

    echo "=== Eval ratio=${RATIO} (RAG grammar, pass 1) ==="
    uv run python src/eval_verilog.py \
        --adapter "$HUB_ID" \
        --problem_file "$PROBLEM_FILE" \
        --include_grammar \
        --grammar_file "$RAG_FILE" \
        --n_samples 1 \
        --temperature 0.0 \
        --output_path "${RESULT_DIR}/rag_r${RATIO}.json"
done

echo "=== Plotting ==="
uv run python src/plot.py plot_paper_pass_at_k \
    --result_files "[\"${RESULT_DIR}/rag_r0.0.json\", \"${RESULT_DIR}/rag_r0.1.json\", \"${RESULT_DIR}/rag_r0.2.json\", \"${RESULT_DIR}/rag_r0.3.json\"]" \
    --labels '["r=0.0", "r=0.1", "r=0.2", "r=0.3"]' \
    --output_path "${RESULT_DIR}/comparison.png" \
    --title "VerilogEval mixed-ratio sweep"
