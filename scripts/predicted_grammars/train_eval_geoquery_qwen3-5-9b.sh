#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="Qwen/Qwen3.5-9B"
MODEL_ALIAS="qwen3-5-9b"

PRED_TRAIN="outputs/predicted_grammars/rag_cot/geoquery_train_k64.json"
PRED_TEST="outputs/predicted_grammars/rag_cot/geoquery_test_k64.json"
TRAIN_RAG="data/geoquery/train_rag.json"
VALID_PATH="data/geoquery/valid.json"
RESULT_DIR="results/predicted_grammars/${MODEL_ALIAS}/geoquery"

HUB_ID="${HF_NAMESPACE}/${MODEL_ALIAS}_geoquery-rag-r0.1"

uv run python src/extract_rag_grammars.py extract \
    --predicted_path "$PRED_TRAIN" \
    --output_path "$TRAIN_RAG"

uv run python src/train.py \
    --model_name "$MODEL_NAME" \
    --mixed_ratio 0.1 \
    --num_train_epochs 1 \
    --train_path "$TRAIN_RAG" \
    --valid_path "$VALID_PATH" \
    --output_dir "outputs/${MODEL_ALIAS}-lora-geoquery-rag-r0.1" \
    --hub_model_id "$HUB_ID"

mkdir -p "$RESULT_DIR"

echo "=== Eval: trained-on-RAG, eval-on-RAG ==="
uv run python src/eval_geoquery.py \
    --adapter "$HUB_ID" \
    --test_path data/geoquery/test.json \
    --include_grammar \
    --grammar_file "$PRED_TEST" \
    --output_path "${RESULT_DIR}/rag.json"

echo "=== Eval: trained-on-RAG, eval-on-gold (reference) ==="
uv run python src/eval_geoquery.py \
    --adapter "$HUB_ID" \
    --test_path data/geoquery/test.json \
    --include_grammar \
    --output_path "${RESULT_DIR}/gold.json"
