# EasyFHE build env: Ubuntu 22.04 (glibc 2.35) compatible wheel, built on an Ubuntu 24.04 host.
# source this file.  Every path below can be overridden from the environment.
#
# This is the recipe that produced the production wheel
#   easyfhe-0.1.1+cu128.git949c3bd.u2204-cp312-cp312-linux_x86_64.whl
# See README.md next to this file.
SRC=${EASYFHE_U22_SRC:?Set EASYFHE_U22_SRC to the pinned EasyFHE source directory}
EASYFHE_SOURCE_PIN=949c3bdebacdaeb885c45d61ba28436e7f959cd3
if [[ -d "$SRC/.git" || -f "$SRC/.git" ]]; then
  [[ "$(git -C "$SRC" rev-parse HEAD)" = "$EASYFHE_SOURCE_PIN" ]] || {
    echo "Expected EasyFHE source $EASYFHE_SOURCE_PIN" >&2; return 2;
  }
elif [[ ! -f "$SRC/COUPLET_SOURCE_PROVENANCE.json" ]]; then
  echo "Use the pinned Git checkout or the accompanying source archive" >&2
  return 2
fi
COMPAT=${EASYFHE_U22_COMPAT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/compat}
export TMPDIR=${EASYFHE_U22_TMPDIR:-${TMPDIR:-/tmp}}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
export PATH=$CUDA_HOME/bin:$PATH
[[ "$("$CUDA_HOME/bin/nvcc" --version)" == *"release 12.8"* ]] || {
  echo "This cu128 build profile requires CUDA Toolkit 12.8" >&2; return 2;
}
export CC=${CC:-/usr/bin/gcc-13} CXX=${CXX:-/usr/bin/g++-13}
export CUDAHOSTCXX=${CUDAHOSTCXX:-$CXX} CMAKE_CUDA_HOST_COMPILER=${CMAKE_CUDA_HOST_COMPILER:-$CXX} CUDACXX=$CUDA_HOME/bin/nvcc
# --- ABI compat: (1) header overlay disables glibc-2.38 __isoc23_strto* redirection for every TU
export CPATH=$COMPAT
# NVTX3 C++ header: the fork dropped third_party/NVTX but profiler/stubs/cuda.cpp + cuda/shared/nvtx.cpp still include
# nvtx3/nvtx3.hpp when CMake finds a "nvtx3" dir in the CUDA include (which only has the C API). Header-only, no ABI impact.
NVTX_INC=${EASYFHE_U22_NVTX_INC:-$SRC/third_party/NVTX/c/include}
[[ -f "$NVTX_INC/nvtx3/nvtx3.hpp" ]] || {
  echo "Missing the pinned NVTX C++ headers; see README.md" >&2; return 2;
}
export CFLAGS="-I$COMPAT" CXXFLAGS="-I$COMPAT -isystem $NVTX_INC"
# CUDAFLAGS gets lost (CMAKE_CUDA_FLAGS is re-initialised by enable_language(CUDA)); PyTorch appends TORCH_NVCC_FLAGS instead.
export TORCH_NVCC_FLAGS="-I$COMPAT"
# --- ABI compat: (2) static libstdc++ (symbols hidden) + shim for the 2 >glibc-2.35 symbols libstdc++.a itself needs.
# Passed as a gcc @response-file: CMakeLists.txt:659 feeds $ENV{LDFLAGS} into a regex, and the "++" in
# -static-libstdc++ makes that regex fail to compile. The rsp path has no regex metachars.
# ldflags.rsp is (re)written here because it must carry absolute paths:
#   -static-libstdc++ -Wl,--exclude-libs,libstdc++.a <COMPAT>/glibc235_compat.o -L<cuda>/targets/x86_64-linux/lib
# (the -L is needed because with an @rsp on the link line gcc rewrites collect2 args into temp @files, so CMake's
#  CUDA ABI detection cannot see nvcc's -L and CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES loses the toolkit lib dir)
[ -f "$COMPAT/glibc235_compat.o" ] || $CC -O2 -fPIC -c "$COMPAT/glibc235_compat.c" -o "$COMPAT/glibc235_compat.o"
printf '%s\n' "-static-libstdc++ -Wl,--exclude-libs,libstdc++.a $COMPAT/glibc235_compat.o -L$CUDA_HOME/targets/x86_64-linux/lib" > "$COMPAT/ldflags.rsp"
export LDFLAGS="@$COMPAT/ldflags.rsp"
unset CMAKE_SHARED_LINKER_FLAGS CMAKE_MODULE_LINKER_FLAGS CMAKE_EXE_LINKER_FLAGS
# --- build profile (same as packaging/manylinux/build_inside_container.sh + previous local build)
export CMAKE_BUILD_TYPE=Release
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0;12.0}"
export MAX_JOBS=${MAX_JOBS:-16}
export USE_CUDA=1 USE_CUDNN=0 USE_CUSPARSELT=0 USE_CUDSS=0 USE_CUFILE=0
export BUILD_TEST=0 USE_AVX=0
export USE_EASYFHE_FAST_BUILD=1 USE_EASYFHE_FAST_INFERENCE=1
export BUILD_CUSTOM_PROTOBUF=0 USE_DISTRIBUTED=0 USE_NCCL=0
export PYTORCH_BUILD_VERSION="0.1.1+cu128.git949c3bd.u2204"
export PYTORCH_BUILD_NUMBER=1
# build venv: python 3.12 with the PyTorch build requirements (setuptools, wheel, cmake>=3.27, ninja, pyyaml, typing_extensions)
if [[ -n "${EASYFHE_U22_BUILD_VENV:-}" ]]; then
  source "$EASYFHE_U22_BUILD_VENV/bin/activate"
fi
