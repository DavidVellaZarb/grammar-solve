#!/usr/bin/env bash
set -euo pipefail

uv run python src/eval_pddl.py evaluate_gbfs_only \
    --test_path data/pddl_satellite/test.json \
    --domain_file pddl_domains/satellite/domain.pddl \
    --output_path results/pddl_satellite/gbfs_only/test.json

uv run python src/eval_pddl.py evaluate \
    --adapter "${HF_NAMESPACE}/qwen2.5-7b_pddl-satellite-baseline" \
    --test_path data/pddl_satellite/test.json \
    --domain_file pddl_domains/satellite/domain.pddl \
    --noinclude_grammar \
    --output_path results/pddl_satellite/baseline/test.json

uv run python src/eval_pddl.py evaluate \
    --adapter "${HF_NAMESPACE}/qwen2.5-7b_pddl-satellite" \
    --test_path data/pddl_satellite/test.json \
    --domain_file pddl_domains/satellite/domain.pddl \
    --include_grammar \
    --output_path results/pddl_satellite/grammar/test.json

uv run python src/eval_pddl.py evaluate \
    --adapter "${HF_NAMESPACE}/qwen2.5-7b_pddl-satellite" \
    --test_path data/pddl_satellite/test.json \
    --domain_file pddl_domains/satellite/domain.pddl \
    --include_grammar \
    --grammar_file outputs/predicted_grammars/rag_cot/pddl_satellite_test_k64.json \
    --output_path results/pddl_satellite/rag/test.json

uv run python src/eval_pddl.py plot \
    --result_files '["results/pddl_satellite/gbfs_only/test.json", "results/pddl_satellite/baseline/test.json", "results/pddl_satellite/rag/test.json", "results/pddl_satellite/grammar/test.json"]' \
    --labels '["GBFS Only", "Baseline", "RAG", "Gold Grammar"]' \
    --output_path results/pddl_satellite/comparison.png \
    --title "Satellite: Seeded GBFS Performance"
