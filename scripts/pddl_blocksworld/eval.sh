#!/usr/bin/env bash
set -euo pipefail

uv run python src/eval_pddl.py evaluate_gbfs_only \
    --test_path data/pddl_blocksworld/test.json \
    --domain_file pddl_domains/blocksworld/domain.pddl \
    --output_path results/pddl_blocksworld/gbfs_only/test.json

uv run python src/eval_pddl.py evaluate \
    --adapter "${HF_NAMESPACE}/qwen2.5-7b_pddl-blocksworld-baseline" \
    --test_path data/pddl_blocksworld/test.json \
    --domain_file pddl_domains/blocksworld/domain.pddl \
    --noinclude_grammar \
    --output_path results/pddl_blocksworld/baseline/test.json

uv run python src/eval_pddl.py evaluate \
    --adapter "${HF_NAMESPACE}/qwen2.5-7b_pddl-blocksworld" \
    --test_path data/pddl_blocksworld/test.json \
    --domain_file pddl_domains/blocksworld/domain.pddl \
    --include_grammar \
    --output_path results/pddl_blocksworld/grammar/test.json

uv run python src/eval_pddl.py evaluate \
    --adapter "${HF_NAMESPACE}/qwen2.5-7b_pddl-blocksworld" \
    --test_path data/pddl_blocksworld/test.json \
    --domain_file pddl_domains/blocksworld/domain.pddl \
    --include_grammar \
    --grammar_file outputs/predicted_grammars/rag_cot/pddl_blocksworld_test_k64.json \
    --output_path results/pddl_blocksworld/rag/test.json

uv run python src/eval_pddl.py plot \
    --result_files '["results/pddl_blocksworld/gbfs_only/test.json", "results/pddl_blocksworld/baseline/test.json", "results/pddl_blocksworld/rag/test.json", "results/pddl_blocksworld/grammar/test.json"]' \
    --labels '["GBFS Only", "Baseline", "RAG", "Gold Grammar"]' \
    --output_path results/pddl_blocksworld/comparison.png \
    --title "Blocksworld: Seeded GBFS Performance"
