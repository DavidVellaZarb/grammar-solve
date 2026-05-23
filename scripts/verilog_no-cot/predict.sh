#!/usr/bin/env bash
# Regenerate Verilog RAG grammars without CoT, using the current domain-specific
# `verilog` system prompt template (rag_grammar.py:201). This isolates the effect
# of CoT scaffolding for the no-CoT ablation — see the no_cot_verilog.sh comment
# on the old generic-prompt file at outputs/predicted_grammars/rag/verilog_test_k64.json.
set -euo pipefail

OUTPUT_DIR=outputs/predicted_grammars/rag_no_cot
mkdir -p "$OUTPUT_DIR"

uv run python src/rag_grammar.py predict \
    --test_path data/verilog_eval/VerilogEval_Human.jsonl \
    --train_path data/mg_verilog/train_detailed.json \
    --grammar_path grammars/verilog.lark \
    --k 64 \
    --model claude-opus-4-6 \
    --output_path "${OUTPUT_DIR}/verilog_test_k64.json" \
    --cache_path cache/rag_no_cot_verilog_cache.json \
    --mode batch
