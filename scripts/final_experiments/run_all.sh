#!/usr/bin/env bash
# Run all final paper experiments end-to-end.
# After each experiment, results are uploaded to HF AND mirrored to /workspace/
# (via sync_path) so they survive a pod restart. The pod is then STOPPED
# (not removed) so it can be resumed later.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

source scripts/run_all/_lib.sh   # provides sync_path() (HF upload + /workspace copy)

echo "########## [1/4] RAG no-CoT eval: SMCalFlow ##########"
bash scripts/final_experiments/no_cot_smcalflow.sh
sync_path "results/no-cot/smcalflow/qwen3-5-9b"

echo "########## [2/4] RAG no-CoT eval: Verilog ##########"
bash scripts/final_experiments/no_cot_verilog.sh
sync_path "results/no-cot/verilog/qwen3-5-9b"

echo "########## [3/4] Full sweep: SMCalFlow ##########"
bash scripts/final_experiments/full_sweep_smcalflow_qwen3-5-9b.sh
sync_path "results/full_sweep/smcalflow/qwen3-5-9b"

echo "########## [4/4] Full sweep: Overnight ##########"
bash scripts/final_experiments/full_sweep_overnight_qwen3-5-9b.sh
sync_path "results/full_sweep/overnight/qwen3-5-9b"

echo "########## All final experiments complete. ##########"

# Stop (not remove) the pod so /workspace/ state is preserved and the pod can resume.
# scripts/run_all/_lib.sh:stop_pod uses `runpodctl remove pod`, which destroys the pod;
# we deliberately use `runpodctl stop pod` here instead.
if [[ -n "${RUNPOD_POD_ID:-}" ]]; then
    echo "Stopping pod ${RUNPOD_POD_ID} (preserving state for restart)..."
    runpodctl stop pod "$RUNPOD_POD_ID"
else
    echo "RUNPOD_POD_ID not set — not on a Runpod, skipping pod stop"
fi
