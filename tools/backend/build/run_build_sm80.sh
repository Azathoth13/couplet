#!/usr/bin/env bash
# Same recipe as run_build.sh, but the CUDA arch list also carries sm_80 (A100 / A800 80 GB).
# The supplied reference wheel has sm_90 + sm_120 cubins only (no PTX), so it cannot run on Ampere.
# Version tag gets ".sm80" so the two wheels never collide in dist/.
#   bash tools/backend/build/run_build_sm80.sh 2>&1 | tee build_sm80.log      (~1-1.5 h, MAX_JOBS=16)
set -eo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST_SM80:-8.0;9.0;12.0}"
export PYTORCH_BUILD_VERSION="${PYTORCH_BUILD_VERSION}.sm80"
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
