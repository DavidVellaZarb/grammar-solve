#!/usr/bin/env bash
# Run all three paper ablations end-to-end on a Runpod pod.
# Order: cheapest first (eval-only) -> baseline 2-epoch -> grammar_program.
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

run_and_sync() {
    local script="$1"
    bash "$script"
    sync_path "results/ablations/smcalflow/qwen3-5-9b"
    sync_path "results/ablations/verilog/qwen3-5-9b"
}

echo "############ ABLATION 3: mixed_nogrammar (eval-only) ############"
run_and_sync scripts/ablations/mixed_nogrammar.sh

echo "############ ABLATION 2: baseline_2epoch ############"
run_and_sync scripts/ablations/baseline_2epoch.sh

echo "############ ABLATION 1: grammar_program ############"
run_and_sync scripts/ablations/grammar_program.sh

# Stop (not remove) the pod so /workspace/ state is preserved and the pod can resume.
# scripts/run_all/_lib.sh:stop_pod uses `runpodctl remove pod`, which destroys the pod;
# we deliberately use `runpodctl stop pod` here instead.
if [[ -n "${RUNPOD_POD_ID:-}" ]]; then
    echo "Stopping pod ${RUNPOD_POD_ID} (preserving state for restart)..."
    runpodctl stop pod "$RUNPOD_POD_ID"
else
    echo "RUNPOD_POD_ID not set — not on a Runpod, skipping pod stop"
fi
