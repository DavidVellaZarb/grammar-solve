#!/usr/bin/env bash
set -euo pipefail

# k=1
uv run python src/knn.py predict \
    --k 1 \
    --output_path outputs/predicted_grammars/knn_k1_generic.json

uv run python src/specialize_grammar.py \
    --test_path outputs/predicted_grammars/knn_k1_generic.json \
    --output_path outputs/predicted_grammars/knn_k1_specialized.json

# k=3, union
uv run python src/knn.py predict \
    --k 3 \
    --strategy union \
    --output_path outputs/predicted_grammars/knn_k3_union_generic.json

uv run python src/specialize_grammar.py \
    --test_path outputs/predicted_grammars/knn_k3_union_generic.json \
    --output_path outputs/predicted_grammars/knn_k3_union_specialized.json

# k=3, intersection
uv run python src/knn.py predict \
    --k 3 \
    --strategy intersection \
    --output_path outputs/predicted_grammars/knn_k3_intersection_generic.json

uv run python src/specialize_grammar.py \
    --test_path outputs/predicted_grammars/knn_k3_intersection_generic.json \
    --output_path outputs/predicted_grammars/knn_k3_intersection_specialized.json
