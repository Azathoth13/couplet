# Couplet

Research artifact for the paper:

> **Couplet: Co-Designing Ciphertext Layouts and Computation for Fully Homomorphic LLM Inference**

GPU CKKS inference with Couplet feature-major packing and our SC comparison
implementation. Includes model preparation, Prefill, encrypted KV-cache
preparation, Decode, and real 1/2/4/8-GPU execution.

| Model | Layers | Couplet / SC | Physical GPU counts |
|---|---:|---|---|
| Llama-3.2-1B | 16 | Prefill → ciphertext KV → Decode | 1, 2, 4, 8 |
| Llama-3.2-3B | 28 | Prefill → ciphertext KV → Decode | 1, 2, 4, 8 |
| Llama-3-8B | 32 | Prefill → ciphertext KV → Decode | 1, 2, 4, 8 |

Repository: [Azathoth13/couplet](https://github.com/Azathoth13/couplet).
Model-specific execution profiles are supplied in `configs/models/`.

## Artifact scope

This artifact reproduces inference and performance for the supplied **frozen
model profiles**. The public implementation name and documented CLI scheme are
**Couplet** (`--scheme couplet`); `--scheme sc` selects our SC comparison.

| Included | Boundary |
|---|---|
| Couplet and SC execution | Llama 1B/3B/8B layouts, linear projections, attention, nonlinear evaluation, bootstrapping, KV preparation and linked Decode; single GPU and real 1/2/4/8-GPU profiles |
| Both Softmax schedules | Baseline Alg2 and S-chain implementations, their switches, eligibility checks and fallbacks; the SC baseline retains its own normalization path |
| Frozen nonlinear artifacts | Selected coefficients, approximation intervals, degrees, prescales, normalization rounds and runtime level/refresh settings |
| Preparation and measurements | Official-weight download instructions, QuaRot/model adaptation, fixture generation, breakdown/packing ablations, and benchmark results with configuration and validation details |
| Backend and validation | EasyFHE package/source/build instructions, shared acceleration sources, CPU contracts and optional GPU checks; third-party license notices are retained |

The artifact does **not** provide the full offline joint error-budget search,
model-quality/PPL search driver, or standalone coefficient-fitting and
PS-depth retuning tools. It evaluates the selected polynomials; it does not
rerun the paper's search to obtain them. Licensed model weights are obtained
separately.

Generic polynomial evaluation and runtime initialization remain included.
In particular, SC constructs polynomial plans from the supplied frozen
calibration settings during initialization. That dependency is retained;
it is not the model-wide error-budget/PPL search.

The published calibration tables cover the documented workloads. The artifact
reproduces the selected configurations, rather than claiming validated
accuracy for arbitrary prompts, context lengths, or a rerun of the full search.

## Install and prepare

Use Linux, Python 3.12, a compatible EasyFHE build, and CUDA Toolkit with nvcc.
EasyFHE remains a separate dependency; see [backend requirements](docs/BACKEND.md)
for the reference wheel/hash and shared application-side kernels. The wheel,
corresponding source bundle, and checksums are available in
the [v0.2.0 Release](https://github.com/Azathoth13/couplet/releases/tag/v0.2.0).

```bash
git clone https://github.com/Azathoth13/couplet.git
cd couplet
python3 tools/backend/download.py
export EASYFHE_WHEEL="$PWD/tools/third_party/wheels/easyfhe-0.1.1+cu128.git949c3bd.u2204-cp312-cp312-linux_x86_64.whl"
./tools/setup_env.sh
source ./tools/activate_env.sh
python tools/backend/check.py

# Use a separate conversion environment when ordinary PyTorch is required.
python tools/models/download.py --model 3b --data-root /path/to/data
python tools/models/prepare.py --model 3b --data-root /path/to/data --stage all --dry-run
```

Follow [model preparation](tools/models/README.md) to transform weights and
generate fixtures and SC assets. Remove `--dry-run` to prepare the data.
Checkpoint files and large generated matrices are not stored in Git.
Users must accept the corresponding Meta model license.

## Run

Inspect a plan without loading weights or creating an FHE context:

```bash
python run_benchmark.py --model 3b --scheme couplet --phase linked \
  --world-size 4 --data-root /path/to/data --output outputs/couplet-3b-w4 --dry-run
python run_benchmark.py --model 3b --scheme sc --phase linked \
  --world-size 4 --data-root /path/to/data --output outputs/sc-3b-w4 --dry-run
```

Remove `--dry-run` to execute. Choose `--model 1b`, `3b`, or `8b`,
`--world-size 1`, `2`, `4`, or `8`, and `--phase prefill`, `linked`, or
`decode`. Decode also runs Prefill so that its KV input is produced by the
encrypted Prefill computation. Standalone reference-KV microbenchmarks in
low-level runners have a different initialization protocol.

Couplet's default linked run measures the first Decode. For fixed-state warm
replay, use `--hot-replay --decode-tokens 3`. Replay fixes the query, position,
and original ciphertext KV; it does not generate successive tokens.
See [reproduction and timing definitions](docs/REPRODUCING.md).

## Prefill performance

The table reports Prefill times in seconds for **128 input tokens with no
prefix** on RTX PRO 6000 Blackwell Server Edition 96 GB GPUs. Each value sums
the maximum forward time across ranks at each layer. Initialization, client
work, and separately measured transfers are excluded.

| Model / packing | 1 GPU | 2 GPUs | 4 GPUs | 8 GPUs |
|---|---:|---:|---:|---:|
| 1B / Couplet | 34.646 | 21.495 | 15.595 | 15.606 |
| 1B / SC | 45.940 | 30.050 | 23.780 | 23.440 |
| 3B / Couplet | 96.311 | 57.125 | 37.840 | 31.944 |
| 3B / SC | 100.710 | 62.470 | 47.110 | 40.220 |
| 8B / Couplet | 143.133 | 83.493 | 54.301 | 46.895 |
| 8B / SC | 167.220 | 105.790 | 71.800 | 58.490 |

Values use the supplied model- and scheme-specific profiles. Generic 3B
execution pads 3072 hidden features and 24 query heads to a physical layout
of 4096 features and 32 heads. See the [benchmark records](bench/records/README.md)
for configuration, timing, and validation details, and the
[reproduction guide](docs/REPRODUCING.md) for Decode measurements.

## Repository layout

```text
run_benchmark.py              public Couplet / SC launcher
configs/models/              frozen 1B / 3B / 8B execution profiles
assets/
  polynomials/{1b,3b,8b}/     required tables, grouped by reference/runtime role
  prefill128_input_ids.json   canonical input fixture
llama3fhe/                   layouts, operators, graph and parallel runtime
  acceleration/              backend patches and CUDA sources
    fhe_ext/                 shared EasyFHE extension modules and kernels
baselines/sc/                SC providers and frozen per-layer calibration
bench/
  ablations/                 packing and KV controls
  breakdown/                 layer recording over the shared runner
  profiling/                 atomic-operation counts
  single_gpu/                production single-process audit/run tools
  records/                   curated measured results
  t2_*.py                    shared lower-level execution drivers
tools/
  models/                    download and asset preparation
  convert/                   QuaRot and model adaptation
  backend/                   wheel download, checks and build recipe
  release/                   manifests and provenance
docs/                        reproduction, backend, design and packing guides
tests/                       CPU contracts and optional GPU checks
```

Coefficient tables are grouped by model and execution role. Separate reference
and runtime packs reproduce the fixture trajectories and selected operator
settings; [the table index](assets/polynomials/README.md) explains each role.
The canonical 128-token input fixture is included.

For selected-layer breakdown use `bench/breakdown/profile_layer.py`; for packing
controls see [bench/README.md](bench/README.md). The documentation entry point is
[docs/README.md](docs/README.md).

Run `tools/verify.sh` for CPU checks. GPU checks are optional; see the
[validation guide](docs/REPRODUCING.md#validation). A
[CI template](tools/ci/README.md) is also provided.

Project license: [GPL-3.0-or-later](LICENSE). Third-party/model terms:
[NOTICE](NOTICE) and [SC notices](baselines/sc/NOTICE).
