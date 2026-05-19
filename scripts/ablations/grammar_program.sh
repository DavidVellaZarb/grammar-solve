#!/usr/bin/env bash
# Ablation 1: train query -> minimal_grammar + program (single sequence target).
# At inference the model emits the grammar and then the program; eval substring-matches
# the gold program inside the output (eval.py:check_match / verilog functional harness
# after stripping the grammar prefix).
set -euo pipefail

MODEL_NAME="Qwen/Qwen3.5-9B"
MODEL_ALIAS="qwen3-5-9b"

# ---------- SMCalFlow ----------
SMCAL_HUB_ID="${HF_NAMESPACE}/${MODEL_ALIAS}_smcalflow-grammar-program"

echo "######## TRAIN smcalflow grammar_program (${MODEL_ALIAS}) ########"
uv run python src/train.py \
    --model_name "$MODEL_NAME" \
    --task grammar_program \
    --num_train_epochs 1 \
    --train_path data/smcalflow/train.json \
    --valid_path data/smcalflow/valid.json \
    --output_dir "outputs/${MODEL_ALIAS}-lora-smcalflow-grammar-program" \
    --hub_model_id "$SMCAL_HUB_ID" \
    --nosave_locally

echo "######## EVAL smcalflow grammar_program (${MODEL_ALIAS}) ########"
mkdir -p "results/ablations/smcalflow/${MODEL_ALIAS}"
uv run python src/eval.py \
    --adapter "$SMCAL_HUB_ID" \
    --test_path data/smcalflow/test.json \
    --task grammar_program \
    --output_path "results/ablations/smcalflow/${MODEL_ALIAS}/grammar_program.json"

# ---------- Verilog ----------
VERILOG_HUB_ID="${HF_NAMESPACE}/${MODEL_ALIAS}_mg-verilog-grammar-program"

echo "######## TRAIN verilog grammar_program (${MODEL_ALIAS}) ########"
uv run python src/train.py \
    --model_name "$MODEL_NAME" \
    --task grammar_program \
    --num_train_epochs 1 \
    --train_path data/mg_verilog/train_detailed.json \
    --valid_path data/mg_verilog/valid_detailed.json \
    --output_dir "outputs/${MODEL_ALIAS}-lora-verilog-grammar-program" \
    --hub_model_id "$VERILOG_HUB_ID" \
    --max_seq_length 2048 \
    --nosave_locally

echo "######## EVAL verilog grammar_program (${MODEL_ALIAS}) ########"
mkdir -p "results/ablations/verilog/${MODEL_ALIAS}"
uv run python src/eval_verilog.py \
    --adapter "$VERILOG_HUB_ID" \
    --problem_file data/verilog_eval/VerilogEval_Human.jsonl \
    --task grammar_program \
    --n_samples 5 \
    --temperature 0.8 \
    --max_new_tokens 2048 \
    --output_path "results/ablations/verilog/${MODEL_ALIAS}/grammar_program.json"
