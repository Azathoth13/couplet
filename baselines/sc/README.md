# SC packing baseline

This directory releases our SC implementation on EasyFHE for Llama-3.2-1B,
Llama-3.2-3B and Llama-3-8B. It includes encrypted Prefill, the encrypted KV
captured by that Prefill, and Decode with 1/2/4/8 physical GPU ranks. It is our
reimplementation/adaptation; it is not the Sylph authors' original benchmark
artifact.

Use the repository launcher with `--scheme sc`. It delegates to `run.py`,
the sole SC-specific command-line entry. EasyFHE is imported only when execution
begins. `--dry-run` prints the execution plan without loading model weights or
touching GPUs; the SC-specific `--preflight` checks prepared assets only.

```bash
# From the repository root, prepare model/SC assets using tools/models/prepare.py.
python run_benchmark.py --scheme sc --model 3b --phase linked --world-size 4 \
  --data-root /path/to/data --output artifacts/sc-3b-w4 --dry-run

# Once the declared EasyFHE backend and model assets are available:
python run_benchmark.py --scheme sc --model 3b --phase linked --world-size 4 \
  --data-root /path/to/data --output artifacts/sc-3b-w4
```

`--phase prefill` stops immediately after complete encrypted Prefill and KV
capture. `--phase decode` and `--phase linked` both run Prefill first: they never
initialize the FHE Decode cache by encrypting a plaintext KV reference. The
Decode reports contain the first call and hot replay. A hot replay repeats the
same query, position 128 and original KV; it is not continuous token generation.
The independent plaintext reference runs before the timed Decode passes, while
the FHE operators consume the actual encrypted Prefill cache.

## Directory map

```text
baselines/sc/
  run.py                 SC launcher called by run_benchmark.py
  install_assets.py      install frozen calibration and generate SC assets
  rotation.py            8B residual/value basis conversion
  worker.py              compose the selected parallel execution policy
  assets/{1b,3b,8b}/     frozen per-layer calibration and manifest templates
  providers/             model-specific runtime implementations
    llama1b/             2048-feature, head-dimension-64 implementation
    llama3b/             padded model with 24 active attention heads
    llama8b/             4096-feature, 32-head implementation
  drivers/               Prefill-to-encrypted-KV-to-Decode execution bridge
  parallel/              partitioning, transport and runtime policies
  tests/                 CPU contracts
  LICENSES/              upstream license notices
```

`providers/` is an internal dependency of the launcher, not a set of alternative
public entry points. The model-specific copies preserve the geometry adaptations
used by the measured configurations. Their `chain/run_chain.py` files are invoked
by the linked drivers to capture the actual encrypted Prefill cache; they cannot
be removed as duplicate standalone scripts.

The 76 per-layer calibration files in `assets/` are required runtime inputs for
16 + 28 + 32 layers. Their feature-group bounds determine refresh and polynomial
plans. They are retained byte for byte rather than collapsed into one global
approximation range. Generated weights, trajectories and tables belong under
the selected data root, outside the source tree.

## Models and assets

Each GPU worker selects one provider under `providers/`, so the 1B, 3B and 8B
variants of `sylph_easy` do not collide in one process. 1B uses 2048-wide
residuals/head dimension 64. The 3B physical residual width is padded to 4096,
with its 24 active Q heads compacted for SC attention. 8B uses 4096/32 Q heads.

Prepared assets live under:

```text
DATA/sc/{1b,3b,8b}/
  chain/chain-manifest.json
  chain/layer-XX/{calibration.json,weights.npz}
  chain/plain_inputs.npz
  chain/lm_head.npy                 # folded terminal head for 1B/3B
  input/query_embedding.npz         # predicted-token embedding for 1B/3B
```

Frozen calibration tables and manifest templates are in `assets/`. They belong
to the fixed P0/128-token benchmark input and specified checkpoint revisions.
They are not an assertion of calibrated inference for arbitrary prompts.
`install_assets.py --model ID --data-root DATA --checkpoint CHECKPOINT` installs
these tables and generates the exact NumPy SC reference trajectory. Run it
through `tools/models/prepare.py` after the model and Couplet fixtures are built.

8B requires an additional residual and per-head value basis conversion beyond
QuaRot. `rotation.py` and `install_assets.py` implement that conversion with the
frozen seed. Copying ordinary QuaRot Q/K/V weights into the 8B SC chain is not
valid. P0 assets contain empty public-prefix arrays and have their own hashes,
recorded in the manifest. Model weights, LM heads, large NPZ files and generated
trajectories are not distributed in Git.

## Parallel and timing protocol

The default `--nonlinear-mode combined` uses output-block linear sharding,
Q-head attention sharding, feature-group SiLU sharding with state-batched
packets, and feature reduction followed by two-root RMSNorm inverse evaluation.
At W1 it uses the single-GPU implementation. `legacy` selects replicated
nonlinear work for comparison. Decode uses distributed coefficient MV groups
and per-layer encrypted-input synchronization.

At W8, 3B/8B keep Decode projection tables on GPU. 8B W1/W2/W4 cache the exact
tables in CPU RAM and upload one layer at a time; the 8B host-table
cache is about 448 GiB. This profile needs substantial host memory. The measured
platform was 8 x RTX PRO 6000 Blackwell 96 GB with approximately 1.5 TiB host
RAM. These are measured configurations, not minimum hardware guarantees.

`summary.json` separates the sum of per-layer maximum-rank forward times from
per-rank chain wall time. Online table uploads remain in the chain wall, and
one-time setup/cache preparation is recorded separately. Every-layer timers
contain the communication they wait for. Do not label the forward sum as total
service latency or compare a hot replay directly against another scheme's first
Decode call. CPU checks do not establish GPU execution or timing acceptance.

## Source and backend

`SOURCE_MANIFEST.json` records source provenance and file hashes.
`parallel/` contains the partition, wire/runtime and Decode
reference/table-cache adapters. Its `run_linked.py` is an internal execution
bridge and does not define a second command-line interface. `drivers/` executes
the linked computation. `worker.py` composes the Prefill and Decode policies.
The integration requires GPU
acceptance on a configured machine before claiming new benchmark results.

The backend remains the separately supplied EasyFHE version documented in the
main repository. SC's CUDA extensions also need a CUDA Toolkit/nvcc and a writable
compile cache. See the main backend instructions and [NOTICE](NOTICE).

CPU-only checks:

```bash
python -m pytest baselines/sc/tests -q
```
