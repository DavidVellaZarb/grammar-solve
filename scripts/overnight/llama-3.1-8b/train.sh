#!/usr/bin/env bash
set -euo pipefail
"$(dirname "$0")/../train.sh" "meta-llama/Llama-3.1-8B-Instruct" "llama-3.1-8b"
