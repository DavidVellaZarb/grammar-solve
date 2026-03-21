#!/usr/bin/env bash
set -euo pipefail

echo "Submitting CoT batches for all domains..."

uv run python src/generate_cot.py submit \
    --input_path "data/mg_verilog/train_detailed.json" \
    --output_path "data/mg_verilog/train_detailed_cot.json" \
    --grammar_path "grammars/verilog.lark" \
    --cache_path "cache/cot_verilog_cache.json" \
    --task_name "cot_verilog" \
    --domain_description "a Verilog hardware description language"

uv run python src/generate_cot.py submit \
    --input_path "data/openscad/train.json" \
    --output_path "data/openscad/train_cot.json" \
    --grammar_path "grammars/openscad.lark" \
    --cache_path "cache/cot_openscad_cache.json" \
    --task_name "cot_openscad" \
    --domain_description "an OpenSCAD 3D modeling language"

uv run python src/generate_cot.py submit \
    --input_path "data/smiles/train.json" \
    --output_path "data/smiles/train_cot.json" \
    --grammar_path "grammars/smiles.lark" \
    --cache_path "cache/cot_smiles_cache.json" \
    --task_name "cot_smiles" \
    --domain_description "a SMILES chemical notation language"

uv run python src/generate_cot.py submit \
    --input_path "data/spice/train.json" \
    --output_path "data/spice/train_cot.json" \
    --grammar_path "grammars/spice.lark" \
    --cache_path "cache/cot_spice_cache.json" \
    --task_name "cot_spice" \
    --domain_description "a SPICE circuit simulation language"

echo "All CoT batches submitted."
