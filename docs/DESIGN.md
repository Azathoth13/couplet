# Runtime design

Couplet and the SC comparison share the model preparation and backend setup,
while retaining their distinct packing and execution paths. The three frozen
model profiles are the authoritative geometry, polynomial and level settings.

## Couplet execution

`llama3fhe/` contains the model geometry, layouts, graph, operators and parallel
runtime. One linked worker program handles both single- and multi-GPU runs.
The per-layer computation is:

```text
feature-major hidden → RMSNorm → QKV / RoPE adapter
  → QK → Softmax → score × V → output projection → residual
  → RMSNorm → gate/up → SiLU × up → down projection → residual
```

Linear projections use BSGS evaluation and reuse encoded weight chunks across
local token shards. Operation-specific complex pairing connects the linear,
attention and cache layouts. Bootstrap and modulus schedules are selected by
the model profile and runtime plan rather than inferred from this diagram.

The `layouts/` definitions own the slot contracts. `operators/` implements
linear, attention, nonlinear and normalization operations. `graph/` composes
these into Prefill and Decode layers. `parallel/` assigns work and exchanges
ciphertext packets between ranks. `acceleration/` contains the shared
application-side backend patches and CUDA sources used by the fast profiles.

1B uses a 2048-feature geometry. The generic 3B profile uses a semantic-preserving
4096-feature / 32-physical-query-head checkpoint adapted from native 3072 / 24.
The separate 3B P3 single-GPU adapter filters inactive queries earlier and does
not support multiple ranks. 8B uses its native 4096 / 32 geometry.

## Softmax and nonlinear evaluation

Both baseline Alg2 and S-chain schedules remain in the Prefill and Decode
implementations. The profiles enable S-chain only where its round count,
layout and modulus conditions allow it; baseline Alg2 remains the fallback.
SC retains its own normalization path.

The repository supplies the selected nonlinear coefficients and configuration.
Runtime polynomial evaluation, PS scheduling and SC initialization from saved
calibration are included. Offline model-wide error-budget/PPL search and
standalone fitting or degree-retuning tools are outside the artifact scope.

## Ciphertext KV continuity

Prefill's encrypted K/V intermediates are converted and refreshed into the
PairFold cache used by Decode. Cache preparation is part of the linked path;
it does not initialize the server cache by encrypting plaintext reference KV.
The cache uses fixed-coordinate token insertion and 128-token pages, retaining
completed pages and replacing only the live tail transactionally.

Plaintext polynomial trajectories are diagnostic/reference assets. Independent
Decode microbenchmarks that initialize from those reference assets have a
different protocol from linked Prefill → ciphertext KV → Decode.

## Parallel execution

Physical GPU runs use one worker per rank, with model-aware token/feature/head
ownership. The measured backend transports ciphertexts through host staging;
NCCL and CUDA IPC are not part of this execution path. Emulation is not used
to claim physical multi-GPU speedups.

SC lives under `baselines/sc/`. Its providers retain model-specific geometry
and source contracts. Prefill shards attention queries and SiLU feature groups;
RMSNorm combines local feature statistics and distributes normalization factors.
Decode uses per-layer input synchronization and table-residency policies.
The saved per-layer calibration files are required by that runtime.

## Entry points and compatibility

`run_benchmark.py` is the public launcher. Shared lower-level runners in `bench/`
serve breakdown and ablation tools. `bench/single_gpu/` retains the production
single-process audit/runner entry points. Current linked profiles use PairFold;
C-form compatibility code in the operator modules supports layout and ownership
regression checks.

See [reproduction](REPRODUCING.md) for workload and timing definitions,
[packing](PACKING.md) for slot examples, and [backend](BACKEND.md) for the native
library and extension contract.
