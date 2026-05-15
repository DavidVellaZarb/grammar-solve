#!/usr/bin/env bash
set -euo pipefail
"$(dirname "$0")/../train.sh" "google/gemma-3-12b-it" "gemma-3-12b"
