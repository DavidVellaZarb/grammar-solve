#!/usr/bin/env bash
set -euo pipefail

uv run python src/knn.py predict \
    --k 1 \
    --output_path outputs/predicted_grammars/knn_k1.json \
    "$@"

uv run python src/knn.py predict \
    --k 3 \
    --strategy union \
    --output_path outputs/predicted_grammars/knn_k3_union.json \
    "$@"

uv run python src/knn.py predict \
    --k 3 \
    --strategy intersection \
    --output_path outputs/predicted_grammars/knn_k3_intersection.json \
    "$@"
