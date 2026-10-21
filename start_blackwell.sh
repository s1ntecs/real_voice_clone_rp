#!/usr/bin/env bash
set -euo pipefail
cd /workspace
python3 -u scripts/verify_gpu_runtime.py
exec /start.sh
