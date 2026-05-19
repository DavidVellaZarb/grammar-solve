#!/usr/bin/env bash
# Ablation 3: evaluate the existing mixed-r0.1 model (10% query->program, 90% query+grammar->program)
# WITHOUT grammars at inference time. The adapters already exist on HF; no training here.
set -euo pipefail

MODEL_ALIAS="qwen3-5-9b"

# ---------- SMCalFlow ----------
SMCAL_HUB_ID="${HF_NAMESPACE}/${MODEL_ALIAS}_smcalflow-mixed-r0.1"

echo "######## EVAL smcalflow mixed-nogrammar (${MODEL_ALIAS}) ########"
mkdir -p "results/ablations/smcalflow/${MODEL_ALIAS}"
uv run python src/eval.py \
    --adapter "$SMCAL_HUB_ID" \
    --test_path data/smcalflow/test.json \
    --noinclude_grammar \
    --output_path "results/ablations/smcalflow/${MODEL_ALIAS}/mixed_nogrammar.json"

# ---------- Verilog ----------
VERILOG_HUB_ID="${HF_NAMESPACE}/${MODEL_ALIAS}_mg-verilog-mixed-r0.1"

echo "######## EVAL verilog mixed-nogrammar (${MODEL_ALIAS}) ########"
mkdir -p "results/ablations/verilog/${MODEL_ALIAS}"
uv run python src/eval_verilog.py \
    --adapter "$VERILOG_HUB_ID" \
    --problem_file data/verilog_eval/VerilogEval_Human.jsonl \
    --noinclude_grammar \
    --n_samples 5 \
    --temperature 0.8 \
    --output_path "results/ablations/verilog/${MODEL_ALIAS}/mixed_nogrammar.json"
