#!/usr/bin/env bash
set -euo pipefail

uv run python src/eval_pddl.py evaluate_gbfs_only \
    --test_path data/pddl_depot/test.json \
    --domain_file pddl_domains/depot/domain.pddl \
    --output_path results/pddl_depot/gbfs_only/test.json

uv run python src/eval_pddl.py evaluate \
    --adapter "${HF_NAMESPACE}/qwen2.5-7b_pddl-depot-baseline" \
    --test_path data/pddl_depot/test.json \
    --domain_file pddl_domains/depot/domain.pddl \
    --noinclude_grammar \
    --output_path results/pddl_depot/baseline/test.json

uv run python src/eval_pddl.py evaluate \
    --adapter "${HF_NAMESPACE}/qwen2.5-7b_pddl-depot" \
    --test_path data/pddl_depot/test.json \
    --domain_file pddl_domains/depot/domain.pddl \
    --include_grammar \
    --output_path results/pddl_depot/grammar/test.json

uv run python src/eval_pddl.py evaluate \
    --adapter "${HF_NAMESPACE}/qwen2.5-7b_pddl-depot" \
    --test_path data/pddl_depot/test.json \
    --domain_file pddl_domains/depot/domain.pddl \
    --include_grammar \
    --grammar_file outputs/predicted_grammars/rag_cot/pddl_depot_test_k64.json \
    --output_path results/pddl_depot/rag/test.json

uv run python src/eval_pddl.py plot \
    --result_files '["results/pddl_depot/gbfs_only/test.json", "results/pddl_depot/baseline/test.json", "results/pddl_depot/rag/test.json", "results/pddl_depot/grammar/test.json"]' \
    --labels '["GBFS Only", "Baseline", "RAG", "Gold Grammar"]' \
    --output_path results/pddl_depot/comparison.png \
    --title "Depot: Seeded GBFS Performance"
