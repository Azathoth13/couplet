# Reproducing the Couplet Llama profiles

The documented execution schemes are Couplet and SC.

The model profiles in `configs/models/{1b,3b,8b}.json` select frozen
coefficients, level budgets, physical model geometry, and runtime switches.
`run_benchmark.py --dry-run` prints the command and profile variables without
running FHE. Generate each model's matching assets with `tools/models/prepare.py`.

## Workloads

`--phase prefill` executes all layers and prepares the ciphertext KV cache.
`--phase linked` runs Prefill and Decode in one execution. `--phase decode`
also runs the required Prefill; it does not substitute an encrypted plaintext
reference KV cache.

Default Couplet Decode count is one. With `--hot-replay --decode-tokens 3`, later
invocations replay the same query, position, and original ciphertext KV.
Cache cloning is replay preparation, separate from forward time. A replay
matching the first call is self-consistency, not an independent quality oracle.
The frozen benchmark uses a small client embedding-row asset for its measured
queries. For continuous generation, use the low-level runner with the checkpoint
embedding table instead of the limited `--client-embedding-rows` fixture.

Low-level `--verify` compares decrypted diagnostics with the polynomial
reference. Reference fixtures are client/test artifacts; the linked execution
still consumes ciphertext KV emitted by Prefill. Standalone `bench/t2_decode.py`
reference-cache mode has a different initialization protocol.

## GPU count and packing

One worker runs on each selected physical GPU. Couplet accepts explicit GPU IDs
with `--devices`; SC uses `CUDA_VISIBLE_DEVICES`. Single-GPU emulation is a
debugging mode and cannot establish a real multi-GPU speedup.

1B uses 2048 features and dedicated ownership when query streams are fewer
than ranks. Generic 3B execution uses a padded 4096-feature / 32-head layout,
with normalization and RoPE adjusted by the checkpoint transformation. Its
single-GPU adapter under `bench/adapters/llama32_3b_p3/` filters inactive
queries earlier, uses the P1 Decode route, and rejects world sizes above one.
Keep its results separate from the generic matrix.

SC providers run in isolated processes because packing/head geometries differ.
Prefill shards attention queries and SiLU feature groups. RMSNorm
reduces local feature-square sums, computes two token groups on two roots,
then distributes inverse normalization factors. See `baselines/sc/README.md`.

## Timing

Retain raw JSON and distinguish online per-layer maximum-rank forward sums,
KV snapshot/layout/bootstrap preparation, weight/cache uploads, first-call
Decode, fixed-state hot replay, and complete chain/process wall time.

The benchmark summary reports online forward sums. SC8B Decode uses CPU table
staging on 1/2/4 GPUs and on-device tables on 8 GPUs; its forward-only values
must not be relabeled as upload-inclusive latency.

## Validation

`tools/verify.sh` performs CPU checks. GPU tests are opt-in and are required
to validate CUDA execution on the target platform. The benchmark records
state their correctness thresholds and quality exceptions.

## Reproduction boundary

This artifact loads the supplied frozen nonlinear coefficients and settings.
It includes baseline Alg2, S-chain, their runtime selection logic, the SC
comparison, and the reference calculations needed for the selected profiles.
Standalone coefficient-fitting and depth-retuning tools and the full joint
error-budget/PPL search workflow are outside the release. Reproducing the
selected inference and timing results does not rerun that search.

SC polynomial plan initialization from frozen calibration is a runtime
dependency and remains available. Generic polynomial evaluation and PS
scheduling likewise remain part of the implementation.
