#!/usr/bin/env bash
set -euo pipefail

uv run python src/classifier.py train \
    --train_path data/smcalflow/train_balanced_generic.json \
    --val_path data/smcalflow/valid_balanced_generic.json \
    --output_dir outputs/classifier_balanced \
    --hub_model_id "${HF_NAMESPACE}/deberta-v3-base_smcalflow_balanced-classifier"
