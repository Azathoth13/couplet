# Benchmarks and measurements

The public entry point is `run_benchmark.py` with `configs/models/`. It uses
one production linked runner for 1/2/4/8 physical GPUs and all three models.

| Location | Purpose |
| --- | --- |
| `t2_prefill_decode.py` | Prefill emits ciphertext KV, then Decode consumes it in the same worker state |
| `t2_model.py`, `t2_layer.py`, `t2_decode.py` | Shared Prefill, single-layer and standalone Decode drivers used by the linked runner |
| `make_model_fixtures.py`, `make_layer_fixture.py` | Prepared weights and reference trajectories |
| `ablations/` | Packing, Encode, GQA, pairing, Prefix and KV-emission controls |
| `breakdown/` | Selected-layer recording over the production runner |
| `profiling/opcount.py` | Atomic-operation instrumentation |
| `adapters/llama32_3b_p3/` | Optimized 3B single-GPU adapter, separate from the generic multi-GPU profile |
| `records/llama/` | Curated final tables and validation notes |

Application acceleration is runtime code under `llama3fhe/acceleration/`,
including the shared EasyFHE extensions in `fhe_ext/` and their CUDA sources.
The SC providers contain the model-specific extension variants. These sources
are required by the supplied execution profiles.

## Layer breakdown

Use the normal frozen-profile launcher with an optional target-layer recorder:

```bash
python bench/breakdown/profile_layer.py --layer 16 -- \
  --model 3b --scheme couplet --phase linked --world-size 1 \
  --data-root /path/to/data --output outputs/3b-layer16 --dry-run
```

Remove `--dry-run` to execute. The additional record fields serialize existing
Decode attention/MLP stage timings; no alternate inference graph is used.
Prefill station/substage fields are already present in the normal reports.
The 16-layer 1B model uses layer indices 0-15. `COUPLET_PROFILE_LAYERS` also
accepts a comma-separated list or ranges for direct low-level runs. By default
no extra nested-stage fields are recorded.

## Ablations

- `ablations/exp1_encode_reuse.py`: isolated QKV/FFN Encode reuse.
- `ablations/exp1_encode_reuse_l16.py`: reuse/re-encode control in the actual
  linked station chain. It repeats GPU packing too; the complete projection
  delta is not pure Encode time.
- `ablations/exp2_joint_complex_gateup.py`: joint versus separate gate/up.
- `ablations/exp4_gqa_recompute.py`: shared versus repeated K/V preparation.
- `ablations/exp5_pairing_ab.py`: QK/PV complex pairing.
- `ablations/run_exp6_kv_emit.sh`: KV format/refresh controls.
- `ablations/prefix_fused_ab.py`: Prefix Softmax branch comparison.

A standalone Decode reference-KV run is an operator diagnostic, distinct from
the linked Prefill→ciphertext-KV→Decode measurement. Fixed-state hot replay is
not continuous generation. Curated records retain their exact timing scopes
and quality limitations.
