# Backend and shared acceleration

The model, packing, parallel runtime, SC comparison, and application-side
acceleration sources are published in this repository. The exact EasyFHE wheel
used by the experiments is distributed as a [v0.2.0 Release attachment](https://github.com/Azathoth13/couplet/releases/tag/v0.2.0),
alongside its corresponding source bundle, build recipe, licenses and checksums.
Binary files are Release assets rather than files in the Git source tree.

## Measured environment

| Component | Version |
|---|---|
| Python | 3.12.3 |
| EasyFHE source | `949c3bd` |
| EasyFHE wheel | `easyfhe-0.1.1+cu128.git949c3bd.u2204-cp312-cp312-linux_x86_64.whl` |
| Wheel SHA-256 | `c0bc34862cdbceb614556ac196849e5844e492f938448d9fdd8d8549a3bc3653` |
| CUDA | 12.8; Toolkit and nvcc required for extension compilation |
| Triton | 3.3.1 |
| GPU | RTX PRO 6000 Blackwell Server Edition, 96 GB |

The wheel is 56,863,015 bytes, for CPython 3.12 / Linux x86_64. It embeds
`sm_90` and `sm_120` native kernels. It does not contain the Ampere `sm_80`
variant; Ampere requires a separate compatible build. The build targets glibc
2.35 or newer and requires the CUDA 12 runtime libraries on the host. The
included compatibility recipe describes the CUDA 12.8 toolchain used.

The measured Python package versions are in `requirements-runtime.txt`.
The backend API contract is summarized below. Other architectures
need a compatible EasyFHE build and appropriate `COUPLET_CUDA_ARCH`; the Blackwell
measurements do not establish compatibility or memory requirements elsewhere.

Additional optimizations are installed at runtime from
`llama3fhe/acceleration/`, including the shared extensions in `fhe_ext/`.
Pinning EasyFHE alone does not enable the measured application configuration.

## Installation

```bash
python3 tools/backend/download.py
export EASYFHE_WHEEL="$PWD/tools/third_party/wheels/easyfhe-0.1.1+cu128.git949c3bd.u2204-cp312-cp312-linux_x86_64.whl"
./tools/setup_env.sh
source ./tools/activate_env.sh
python tools/backend/check.py
```

The downloader verifies SHA-256 and byte count before publishing the local
file. A mirror may be supplied with `--url`, with the same mandatory checks.

## Corresponding source and rebuild

The Release's `easyfhe-949c3bd-source.tar.gz` provides the pinned upstream source,
the pinned third-party sources needed by the build, license notices and the
Couplet compatibility recipe. `BACKEND-MANIFEST.json` and `SHA256SUMS` record
the exact artifacts. The portable recipe is also in `tools/backend/build/`.
EasyFHE's GPL-3.0-only terms and bundled component notices are retained.

The wheel is identified by the checksum above. The source bundle provides
rebuild inputs and instructions; bit-identical output across different
toolchains is not claimed.
Alternatively, a compatible compiled source checkout can be selected with
`EASYFHE_SOURCE_DIR`.

For the exact measured Python stack, install `requirements-runtime.txt` before
running. Model conversion has separate dependencies, documented under
`tools/models/` and `tools/convert/`.

Do not install regular PyTorch into the FHE environment to replace EasyFHE's
torch compatibility layer. Use a separate environment when the checkpoint
conversion requires ordinary PyTorch/Transformers.

## Multi-GPU transport

The measured EasyFHE compatibility layer has no NCCL, gloo, or CUDA IPC.
The distributed runtime uses host staging. CPU memory, pinned buffers, and
PCIe topology matter; GPU count alone does not determine scaling. The measured
host had approximately 1.5 TiB RAM. This is a measured configuration, not a
claim that every model needs that much memory.

SC 8B Decode on 1/2/4 GPUs uses CPU projection tables with per-layer uploads;
8 GPUs can retain the tables on-device. Forward-only timings and upload-inclusive
latency must be reported separately.

## Backend API contract

The runtime uses the pinned backend's public ciphertext/state interfaces,
`PackedRaw`, state-aware `ConstantBundle.plaintext`, complex multiplication
(`homo_mul_i`), unrelinearized multiplication and relinearization, and the
OpenFHE bootstrap planning/execution API. Hoisted BSGS supports reusable baby
rotations and bounded plaintext chunks. `Context.clear_cuda_rotation_cache()`
releases reloadable GPU key copies while preserving CPU master keys.

The measured application installs the project acceleration modules from
`llama3fhe/acceleration/`, including the shared CUDA extensions in `fhe_ext/`.
These are part of the fast profile's runtime dependency closure. Changing the
backend build requires checking these interfaces and representative GPU work
before adopting new performance measurements.
