#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "########## [1/4] RAG no-CoT eval: SMCalFlow ##########"
"${SCRIPT_DIR}/no_cot_smcalflow.sh"

echo "########## [2/4] RAG no-CoT eval: Verilog ##########"
"${SCRIPT_DIR}/no_cot_verilog.sh"

echo "########## [3/4] Full sweep: SMCalFlow ##########"
"${SCRIPT_DIR}/full_sweep_smcalflow_qwen3-5-9b.sh"

echo "########## [4/4] Full sweep: Overnight ##########"
"${SCRIPT_DIR}/full_sweep_overnight_qwen3-5-9b.sh"

echo "########## All final experiments complete. ##########"
