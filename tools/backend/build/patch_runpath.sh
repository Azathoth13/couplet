#!/usr/bin/env bash
# Rewrite the CUDA toolkit RUNPATH of a freshly built wheel from the build host's /usr/local/cuda-12.8/lib64 to the
# generic /usr/local/cuda/lib64 (the reference wheel setting), and repack with a fresh RECORD.
#   bash tools/backend/build/patch_runpath.sh <in.whl> [<out dir>]      (needs `wheel` + `patchelf` on PATH / in the venv)
set -euo pipefail
IN=$(readlink -f "$1")
OUT=${2:-$(dirname "$IN")/patched}
PY=${PYTHON:-python}
FROM=${RUNPATH_FROM:-/usr/local/cuda-12.8/lib64}
TO=${RUNPATH_TO:-/usr/local/cuda/lib64}
WORK=$(mktemp -d "${TMPDIR:-/tmp}/wheelpatch.XXXX")
trap 'rm -rf "$WORK"' EXIT
$PY -m wheel unpack "$IN" -d "$WORK" >/dev/null
while IFS= read -r -d '' so; do
  rp=$(patchelf --print-rpath "$so" 2>/dev/null || true)
  [[ "$rp" == *"$FROM"* ]] || continue
  patchelf --set-rpath "${rp//$FROM/$TO}" "$so"
  echo "patched $(basename "$so"): $rp -> $(patchelf --print-rpath "$so")"
done < <(find "$WORK" -name '*.so' -print0)
mkdir -p "$OUT"
$PY -m wheel pack "$(find "$WORK" -mindepth 1 -maxdepth 1 -type d)" -d "$OUT" | tail -1
rm -rf "$WORK"
