#!/usr/bin/env bash
# KV emit A/B from the same encrypted Prefill chain into the same Decode path.
# One card: ARMS='legacy direct overlap' (owner routing needs >1 card).
# Multi-card: ARMS='direct owner overlap stream' after the host is prepared.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"

PY=${PY:-python}
FIXTURES=${FIXTURES:-$ROOT/fixtures/model}
MODEL_DIR=${MODEL_DIR:-$ROOT/assets/weights/Meta-Llama-3-8B-QuaRot}
POLY_KV=${POLY_KV:-$ROOT/fixtures/model/prefill_kv_poly.npz}
OUT=${OUT:-$ROOT/artifacts/exp6_kv_emit}
LAYERS=${LAYERS:-0-1}
DECODE_TOKENS=${DECODE_TOKENS:-1}
WORLD_SIZE=${WORLD_SIZE:-1}
DEVICES=${DEVICES:-0}
ARMS=${ARMS:-legacy direct overlap}
VERIFY=${VERIFY:-0}
WARMUP=${WARMUP:-1}
WARM_KEYS=${WARM_KEYS:-1}
ROWS=${ROWS:-}
VERIFY_EVERY=${VERIFY_EVERY:-0}

export SYLPH_CUDA_ARCH=${SYLPH_CUDA_ARCH:-sm_120}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export ODIN_SYLPH_KERNELS=1
export ODIN_SYLPH_HOSTCACHE=1
export ODIN_BATCH_BS=1
export ODIN_MULTI_MAC=1
export ODIN_MULTI_GIANT_HOIST=1
export ODIN_SPARSE_NONEXT_BABY=1
export ODIN_SPARSE_GPU_MASKS=1
export ODIN_SPARSE_FUSED=1
export ODIN_FAST_STAGE2=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export T2_PD_DROP_HOST_KEYS=${T2_PD_DROP_HOST_KEYS:-0}
if [ "$DECODE_TOKENS" = 0 ]; then
  export T2_PD_PREFILL_ONLY_KEYS=1
fi
unset SYLPH_HALFBOOT_BATCH

mkdir -p "$OUT"
if [ ! -f "$FIXTURES/hidden_trajectory.npz" ]; then
  echo "missing hidden trajectory under $FIXTURES" >&2
  exit 1
fi
first_layer=${LAYERS%%-*}
if [ ! -f "$FIXTURES/layer${first_layer}_weights.npz" ]; then
  echo "missing model weights under $FIXTURES" >&2
  exit 1
fi
read -r -a device_array <<< "$DEVICES"
if [ "${#device_array[@]}" -lt "$WORLD_SIZE" ]; then
  echo "DEVICES must list at least WORLD_SIZE GPU indices" >&2
  exit 2
fi

{
  date -u +'%Y-%m-%dT%H:%M:%SZ'
  printf 'ROOT=%s PY=%s FIXTURES=%s MODEL_DIR=%s POLY_KV=%s\n' \
    "$ROOT" "$PY" "$FIXTURES" "$MODEL_DIR" "$POLY_KV"
  printf 'LAYERS=%s DECODE_TOKENS=%s WORLD_SIZE=%s DEVICES=%s ARMS=%s VERIFY=%s WARMUP=%s WARM_KEYS=%s ROWS=%s VERIFY_EVERY=%s\n' \
    "$LAYERS" "$DECODE_TOKENS" "$WORLD_SIZE" "$DEVICES" "$ARMS" "$VERIFY" "$WARMUP" "$WARM_KEYS" "$ROWS" "$VERIFY_EVERY"
  printf 'T2_PD_PREFILL_ONLY_KEYS=%s T2_PD_DROP_HOST_KEYS=%s\n' \
    "${T2_PD_PREFILL_ONLY_KEYS-0}" "$T2_PD_DROP_HOST_KEYS"
  sha256sum bench/t2_prefill_decode.py llama3fhe/parallel/prefill/kv_emit.py
  nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
} > "$OUT/manifest.txt"

common=(
  --preset rtx6000pro-high --fixtures "$FIXTURES"
  --world-size "$WORLD_SIZE" --devices "${device_array[@]}"
  --layers "$LAYERS" --decode-tokens "$DECODE_TOKENS"
  --input-level 35 --decode-input-level 35 --kv-limbs 12
  --qk-score-level-drop 3 --encode-depth 0 --verify-every "$VERIFY_EVERY"
)
if [ "$DECODE_TOKENS" = 0 ]; then
  common+=(--retain-weights none --no-prefetch-weights --no-stage-weights)
else
  common+=(--retain-weights host --evict-prefill-keys)
fi
if [ "$WARM_KEYS" = 1 ]; then
  common+=(--warm-keys)
fi
if [ -n "$ROWS" ]; then
  common+=(--max-plaintext-rows "$ROWS")
fi
if [ -f "$MODEL_DIR/config.json" ]; then
  common+=(--model-dir "$MODEL_DIR")
fi
if [ "$VERIFY" = 1 ]; then
  common+=(--poly-kv "$POLY_KV" --verify)
fi
if [ "$WARMUP" = 1 ]; then
  common+=(--warmup-layer0)
fi

for arm in $ARMS; do
  direct=1 owner=0 overlap=0 stream=0
  case "$arm" in
    legacy)  direct=0 ;;
    direct)  ;;
    owner)   owner=1 ;;
    overlap) overlap=1 ;;
    stream)  overlap=1 stream=1 ;;
    *) echo "unknown arm: $arm" >&2; exit 2 ;;
  esac
  if [ "$arm" = owner ] && [ "$WORLD_SIZE" = 1 ]; then
    echo "owner routing needs world_size > 1" >&2
    exit 2
  fi
  echo "start $arm $(date -u +'%Y-%m-%dT%H:%M:%SZ')" | tee -a "$OUT/progress.log"
  LLAMA_FHE_PREFILL_DIRECT_PAIRFOLD_K="$direct" \
  LLAMA_FHE_PREFILL_OWNER_ROUTING="$owner" \
  T2_PREFILL_KV_EMIT_OVERLAP="$overlap" \
  T2_PREFILL_KV_EMIT_STREAM="$stream" \
  "$PY" -u bench/t2_prefill_decode.py "${common[@]}" \
    --output "$OUT/$arm" > "$OUT/$arm.log" 2>&1
  echo "done $arm $(date -u +'%Y-%m-%dT%H:%M:%SZ')" | tee -a "$OUT/progress.log"
done
