#!/usr/bin/env bash
set -euo pipefail

uv run python src/rag_grammar.py predict \
    --test_path data/verilog_eval/VerilogEval_Human.jsonl \
    --train_path data/mg_verilog/train_detailed.json \
    --grammar_path grammars/verilog.lark \
    --k 64 \
    --output_path outputs/predicted_grammars/rag/verilog_test_k64.json \
    --cache_path cache/rag_verilog_cache.json \
    --mode batch
