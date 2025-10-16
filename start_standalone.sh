#!/usr/bin/env bash
set -e

echo "Worker Initiated"

# Подгружаем tcmalloc, если найден
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1 || true)"
if [[ -n "$TCMALLOC" ]]; then
  export LD_PRELOAD="${TCMALLOC}"
  echo "Using LD_PRELOAD=${LD_PRELOAD}"
else
  echo "libtcmalloc not found, continuing without LD_PRELOAD"
fi

export PYTHONUNBUFFERED=1
cd /workspace

echo "Starting RunPod Handler"
python3 -u rp_handler.py
