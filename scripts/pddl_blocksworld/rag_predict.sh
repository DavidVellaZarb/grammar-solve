#!/usr/bin/env bash
set -euo pipefail

uv run python src/rag_grammar.py predict \
    --test_path data/pddl_blocksworld/test.json \
    --train_path data/pddl_blocksworld/train_cot.json \
    --grammar_path grammars/pddl_blocksworld.lark \
    --k 64 \
    --output_path outputs/predicted_grammars/rag_cot/pddl_blocksworld_test_k64.json \
    --cache_path cache/rag_pddl_blocksworld_cache.json \
    --max_tokens 4096 \
    --prompt_style cot \
    --mode batch

uv run python src/eval_grammar.py \
    --predicted_path outputs/predicted_grammars/rag_cot/pddl_blocksworld_test_k64.json \
    --gold_path data/pddl_blocksworld/test.json \
    --write
