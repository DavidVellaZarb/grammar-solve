#!/usr/bin/env bash
set -euo pipefail

GOLD_GENERIC="data/smcalflow/test_generic.json"
GOLD_SPECIALIZED="data/smcalflow/test.json"

uv run python src/knn.py predict \
    --k 1 \
    --output_path outputs/predicted_grammars/knn_k1_generic.json

uv run python src/specialize_grammar.py \
    --test_path outputs/predicted_grammars/knn_k1_generic.json \
    --output_path outputs/predicted_grammars/knn_k1_specialized.json

uv run python src/eval_grammar.py \
    --predicted_path outputs/predicted_grammars/knn_k1_generic.json \
    --gold_path "$GOLD_GENERIC" \
    --write

uv run python src/eval_grammar.py \
    --predicted_path outputs/predicted_grammars/knn_k1_specialized.json \
    --gold_path "$GOLD_SPECIALIZED" \
    --write

uv run python src/knn.py predict \
    --k 3 \
    --strategy union \
    --output_path outputs/predicted_grammars/knn_k3_union_generic.json

uv run python src/specialize_grammar.py \
    --test_path outputs/predicted_grammars/knn_k3_union_generic.json \
    --output_path outputs/predicted_grammars/knn_k3_union_specialized.json

uv run python src/eval_grammar.py \
    --predicted_path outputs/predicted_grammars/knn_k3_union_generic.json \
    --gold_path "$GOLD_GENERIC" \
    --write

uv run python src/eval_grammar.py \
    --predicted_path outputs/predicted_grammars/knn_k3_union_specialized.json \
    --gold_path "$GOLD_SPECIALIZED" \
    --write

uv run python src/knn.py predict \
    --k 3 \
    --strategy intersection \
    --output_path outputs/predicted_grammars/knn_k3_intersection_generic.json

uv run python src/specialize_grammar.py \
    --test_path outputs/predicted_grammars/knn_k3_intersection_generic.json \
    --output_path outputs/predicted_grammars/knn_k3_intersection_specialized.json

uv run python src/eval_grammar.py \
    --predicted_path outputs/predicted_grammars/knn_k3_intersection_generic.json \
    --gold_path "$GOLD_GENERIC" \
    --write

uv run python src/eval_grammar.py \
    --predicted_path outputs/predicted_grammars/knn_k3_intersection_specialized.json \
    --gold_path "$GOLD_SPECIALIZED" \
    --write
