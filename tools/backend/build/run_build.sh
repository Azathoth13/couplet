#!/usr/bin/env bash
set -eo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
cd "$SRC"
echo "=== build start $(date) ==="
echo "version=$PYTORCH_BUILD_VERSION MAX_JOBS=$MAX_JOBS arch=$TORCH_CUDA_ARCH_LIST"
echo "CPATH=$CPATH LDFLAGS=$LDFLAGS TORCH_NVCC_FLAGS=$TORCH_NVCC_FLAGS"
mkdir -p dist
rc=0
time python setup.py bdist_wheel || rc=$?
echo "=== build end $(date) rc=$rc ==="
ls -la dist/ 2>/dev/null
exit $rc
