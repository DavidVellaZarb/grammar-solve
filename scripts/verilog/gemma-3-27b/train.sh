#!/usr/bin/env bash
set -euo pipefail
"$(dirname "$0")/../train.sh" "google/gemma-3-27b-it" "gemma-3-27b"
