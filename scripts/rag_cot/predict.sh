#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR=outputs/predicted_grammars/rag_cot

uv run python src/rag_grammar.py predict \
    --test_path data/smiles/test.json \
    --train_path data/smiles/train_cot.json \
    --grammar_path grammars/smiles.lark \
    --k 64 \
    --output_path "${OUTPUT_DIR}/smiles_test_k64.json" \
    --cache_path cache/rag_cot_smiles_cache.json \
    --max_tokens 4096 \
    --prompt_style cot \
    --mode batch \
    "$@" &
PID_SMILES=$!

uv run python src/rag_grammar.py predict \
    --test_path data/spice/test.json \
    --train_path data/spice/train_cot.json \
    --grammar_path grammars/spice.lark \
    --k 64 \
    --output_path "${OUTPUT_DIR}/spice_test_k64.json" \
    --cache_path cache/rag_cot_spice_cache.json \
    --max_tokens 4096 \
    --prompt_style cot \
    --mode batch \
    "$@" &
PID_SPICE=$!

uv run python src/rag_grammar.py predict \
    --test_path data/openscad/test.json \
    --train_path data/openscad/train_cot.json \
    --grammar_path grammars/openscad.lark \
    --k 64 \
    --output_path "${OUTPUT_DIR}/openscad_test_k64.json" \
    --cache_path cache/rag_cot_openscad_cache.json \
    --max_tokens 4096 \
    --prompt_style cot \
    --mode batch \
    "$@" &
PID_OPENSCAD=$!

uv run python src/rag_grammar.py predict \
    --test_path data/verilog_eval/VerilogEval_Human.jsonl \
    --train_path data/mg_verilog/train_detailed_cot.json \
    --grammar_path grammars/verilog.lark \
    --k 64 \
    --output_path "${OUTPUT_DIR}/verilog_test_k64.json" \
    --cache_path cache/rag_cot_verilog_cache.json \
    --max_tokens 4096 \
    --prompt_style cot \
    --mode batch \
    "$@" &
PID_VERILOG=$!

echo "All batches submitted. Waiting for completion..."

FAILED=0

wait $PID_SMILES  || { echo "SMILES prediction failed"; FAILED=1; }
wait $PID_SPICE   || { echo "SPICE prediction failed"; FAILED=1; }
wait $PID_OPENSCAD || { echo "OpenSCAD prediction failed"; FAILED=1; }
wait $PID_VERILOG  || { echo "Verilog prediction failed"; FAILED=1; }

if [ $FAILED -ne 0 ]; then
    echo "One or more predictions failed."
    exit 1
fi

echo "All predictions complete. Running grammar evaluation..."

uv run python src/eval_grammar.py \
    --predicted_path "${OUTPUT_DIR}/smiles_test_k64.json" \
    --gold_path data/smiles/test.json \
    --write

uv run python src/eval_grammar.py \
    --predicted_path "${OUTPUT_DIR}/spice_test_k64.json" \
    --gold_path data/spice/test.json \
    --write

uv run python src/eval_grammar.py \
    --predicted_path "${OUTPUT_DIR}/openscad_test_k64.json" \
    --gold_path data/openscad/test.json \
    --write

uv run python src/eval_grammar.py \
    --predicted_path "${OUTPUT_DIR}/verilog_test_k64.json" \
    --gold_path data/verilog_eval/VerilogEval_Human.jsonl \
    --write

echo "All grammar evaluations complete."
