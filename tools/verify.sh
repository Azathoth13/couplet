#!/usr/bin/env bash
# CPU contracts by default. GPU checks require an explicit stage.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
STAGE="${1:-unit}"
PY="${LLAMA3_FHE_PYTHON:-python3}"
export PYTHONPATH="$ROOT:${EASYFHE_SOURCE_DIR:-}:${PYTHONPATH:-}"
case "$STAGE" in
  unit)
    "$PY" -m compileall -q llama3fhe bench baselines/sc tools run_benchmark.py
    "$PY" -m pytest tests -q -m 'not gpu'
    "$PY" bench/single_gpu/layer.py --audit-only
    ;;
  gpu)
    "$PY" tools/backend/check.py
    "$PY" -m pytest tests -q -m gpu
    ;;
  *)
    echo "Usage: tools/verify.sh [unit|gpu]" >&2
    exit 2
    ;;
esac
