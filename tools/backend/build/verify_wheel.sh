#!/usr/bin/env bash
# Verify an easyfhe wheel is loadable on Ubuntu 22.04 (glibc 2.35, libstdc++ from gcc-12) without extra runtime libs.
# usage: verify_wheel.sh <wheel>
set -uo pipefail
W="$1"; [ -f "$W" ] || { echo "no wheel: $W"; exit 2; }
D=$(mktemp -d "${TMPDIR:-/tmp}/whlchk.XXXXXX"); trap 'rm -rf "$D"' EXIT
unzip -q -o "$W" 'easyfhe/lib/*' 'easyfhe/_C*' 'easyfhe/version.py' -d "$D"
echo "=== $W ==="; cat "$D/easyfhe/version.py" | grep -E "__version__|cuda|git_version"
fail=0
for f in "$D"/easyfhe/lib/*.so* "$D"/easyfhe/_C*.so; do
  b=$(basename "$f")
  glibc_max=$(objdump -T "$f" | grep -oE 'GLIBC_[0-9.]+' | sort -uV | tail -1)
  glibcxx_max=$(objdump -T "$f" | grep -oE 'GLIBCXX_[0-9.]+' | sort -uV | tail -1)
  cxxabi_max=$(objdump -T "$f" | grep -oE 'CXXABI_[0-9.]+' | sort -uV | tail -1)
  gcc_max=$(objdump -T "$f" | grep -oE 'GCC_[0-9.]+' | sort -uV | tail -1)
  gomp_max=$(objdump -T "$f" | grep -oE 'GOMP_[0-9.]+|OMP_[0-9.]+' | sort -uV | tail -1)
  bad=$(objdump -T "$f" | grep -cE '__isoc23_|arc4random|strlcpy|strlcat')
  needs_stdcxx=$(readelf -d "$f" | grep -c 'libstdc++')
  exported_std=$(nm -D --defined-only "$f" 2>/dev/null | grep -c ' _ZSt\| _ZNSt\| _ZNKSt')
  status=OK
  # glibc 2.35 -> anything > 2.35 is bad
  if [ -n "$glibc_max" ] && [ "$(printf '%s\n' "$glibc_max" GLIBC_2.35 | sort -V | tail -1)" != "GLIBC_2.35" ]; then status=BAD; fi
  if [ -n "$glibcxx_max" ] && [ "$(printf '%s\n' "$glibcxx_max" GLIBCXX_3.4.30 | sort -V | tail -1)" != "GLIBCXX_3.4.30" ]; then status=BAD; fi
  [ "$bad" != 0 ] && status=BAD
  [ $status = BAD ] && fail=1
  printf "%-6s %-32s glibc<=%-11s glibcxx=%-15s cxxabi=%-12s gcc=%-10s omp=%-9s isoc23/arc4=%s needs_libstdc++=%s exported_std=%s\n" "$status" "$b" "$glibc_max" "${glibcxx_max:-none}" "${cxxabi_max:-none}" "${gcc_max:-none}" "${gomp_max:-none}" "$bad" "$needs_stdcxx" "$exported_std"
done
echo "=== NEEDED (union) ==="; for f in "$D"/easyfhe/lib/*.so* "$D"/easyfhe/_C*.so; do readelf -d "$f" | grep NEEDED | awk '{print $NF}'; done | sort -u | tr '\n' ' '; echo
echo "=== CUDA arch in libtorch_cuda.so ==="; ${CUDA_HOME:-/usr/local/cuda-12.8}/bin/cuobjdump --list-elf "$D/easyfhe/lib/libtorch_cuda.so" 2>/dev/null | grep -oE 'sm_[0-9]+' | sort -u | tr '\n' ' '; echo
[ $fail = 0 ] && echo "RESULT: PASS (Ubuntu 22.04 compatible)" || echo "RESULT: FAIL"
exit $fail
