#!/usr/bin/env bash
# Ablation 2: train the no-grammar baseline for 2 epochs (the paper baseline trains 1).
# Eval is identical to the paper baseline: query-only prompt, substring-match accuracy.
set -euo pipefail

MODEL_NAME="Qwen/Qwen3.5-9B"
MODEL_ALIAS="qwen3-5-9b"

# ---------- SMCalFlow ----------
SMCAL_HUB_ID="${HF_NAMESPACE}/${MODEL_ALIAS}_smcalflow-baseline-e2"

echo "######## TRAIN smcalflow baseline-2epoch (${MODEL_ALIAS}) ########"
uv run python src/train.py \
    --model_name "$MODEL_NAME" \
    --noinclude_grammar \
    --num_train_epochs 2 \
    --train_path data/smcalflow/train.json \
    --valid_path data/smcalflow/valid.json \
    --output_dir "outputs/${MODEL_ALIAS}-lora-smcalflow-baseline-e2" \
    --hub_model_id "$SMCAL_HUB_ID" \
    --nosave_locally

echo "######## EVAL smcalflow baseline-2epoch (${MODEL_ALIAS}) ########"
mkdir -p "results/ablations/smcalflow/${MODEL_ALIAS}"
uv run python src/eval.py \
    --adapter "$SMCAL_HUB_ID" \
    --test_path data/smcalflow/test.json \
    --noinclude_grammar \
    --output_path "results/ablations/smcalflow/${MODEL_ALIAS}/baseline_2epoch.json"

# ---------- Verilog ----------
VERILOG_HUB_ID="${HF_NAMESPACE}/${MODEL_ALIAS}_mg-verilog-baseline-e2"

echo "######## TRAIN verilog baseline-2epoch (${MODEL_ALIAS}) ########"
uv run python src/train.py \
    --model_name "$MODEL_NAME" \
    --noinclude_grammar \
    --num_train_epochs 2 \
    --train_path data/mg_verilog/train_detailed.json \
    --valid_path data/mg_verilog/valid_detailed.json \
    --output_dir "outputs/${MODEL_ALIAS}-lora-verilog-baseline-e2" \
    --hub_model_id "$VERILOG_HUB_ID" \
    --max_seq_length 2048 \
    --nosave_locally

echo "######## EVAL verilog baseline-2epoch (${MODEL_ALIAS}) ########"
mkdir -p "results/ablations/verilog/${MODEL_ALIAS}"
uv run python src/eval_verilog.py \
    --adapter "$VERILOG_HUB_ID" \
    --problem_file data/verilog_eval/VerilogEval_Human.jsonl \
    --noinclude_grammar \
    --n_samples 5 \
    --temperature 0.8 \
    --output_path "results/ablations/verilog/${MODEL_ALIAS}/baseline_2epoch.json"
