# EasyFHE cu128 Ubuntu 22.04 compatibility build

This is the build recipe for Couplet's measured EasyFHE wheel:

```text
easyfhe-0.1.1+cu128.git949c3bd.u2204-cp312-cp312-linux_x86_64.whl
SHA256 c0bc34862cdbceb614556ac196849e5844e492f938448d9fdd8d8549a3bc3653
```

The Release attachment preserves the exact 56,863,015-byte measured wheel.
It was built from clean EasyFHE commit
`949c3bdebacdaeb885c45d61ba28436e7f959cd3`, with Ubuntu/glibc compatibility
linking and CUDA architectures `sm_90` and `sm_120`. It requires CPython 3.12,
Linux x86_64, glibc 2.35 or newer, and a compatible CUDA 12.8 runtime/driver.
It does not contain an Ampere `sm_80` cubin. Application-side extensions in
Couplet additionally compile with nvcc and the appropriate architecture.

## Preferred installation

Download the wheel, manifest, checksums, and corresponding source bundle from
the same [Couplet release](https://github.com/Azathoth13/couplet/releases).
Verify `SHA256SUMS`, then install the wheel with `pip install --no-deps` in the
FHE environment after installing Couplet's runtime dependencies. The backend
does not bundle CUDA Toolkit or the NVIDIA driver.

## Rebuild from the corresponding source

The companion `easyfhe-949c3bd-source.tar.gz` contains the complete tracked
EasyFHE source, its seven pinned submodules, the NVTX headers used for this
build, licenses, source provenance, and this recipe under `couplet-build/`.
No model weights, installed environment, native wheel, or Git history is in
the source bundle. The separate wheel is the measured binary.

On Ubuntu 24.04 with GCC/G++ 13, CUDA Toolkit 12.8, Python 3.12, development
headers, protobuf development files, and standard build tools:

```bash
tar -xzf easyfhe-949c3bd-source.tar.gz
python3.12 -m venv build-env
source build-env/bin/activate
python -m pip install -r easyfhe-949c3bd/requirements-build.txt
python -m pip install wheel 'setuptools<82'
export EASYFHE_U22_SRC="$PWD/easyfhe-949c3bd"
export CUDA_HOME=/usr/local/cuda-12.8
export MAX_JOBS=16
bash "$EASYFHE_U22_SRC/couplet-build/run_build.sh"
bash "$EASYFHE_U22_SRC/couplet-build/patch_runpath.sh" "$EASYFHE_U22_SRC"/dist/easyfhe-*.whl
bash "$EASYFHE_U22_SRC/couplet-build/verify_wheel.sh" "$EASYFHE_U22_SRC"/dist/patched/easyfhe-*.whl
```

Full native compilation can take roughly an hour. These commands are not
executed by installation or by the source-release checks. A source archive
has no `.git`; its generated `git_version` may be `Unknown`, while the source
provenance JSON and fixed build-version tag identify the pin. Rebuilding with
this recipe is not a promise of a byte-identical ZIP/native binary.

Alternatively, clone the [EasyFHE upstream](https://github.com/jizhuoran/EasyFHE),
checkout the full pin above and initialize submodules, then provide NVTX
`e170594ac7cf1dac584da473d4ca9301087090c1` under `third_party/NVTX` or set
`EASYFHE_U22_NVTX_INC` to its `c/include` directory.

## Build adaptations

- `compat/features.h` disables glibc 2.38 `__isoc23_strto*` redirection.
- `compat/glibc235_compat.c` supplies compatibility wrappers for symbols used
  by statically linked libstdc++.
- `env.sh` configures hidden/static libstdc++, the response-file linker flags,
  dnum-independent upstream fast-inference build, and `9.0;12.0` CUDA targets.
- `patch_runpath.sh` replaces the build-host CUDA directory with
  `/usr/local/cuda/lib64` and regenerates the wheel RECORD.
- `verify_wheel.sh` inspects ELF imports/versions and embedded CUDA images
  without executing an FHE computation or GPU kernel.

`run_build_sm80.sh` is a separate, optional rebuild profile that adds Ampere
and a `.sm80` package suffix. It is not the measured Release attachment.
Select architecture before a clean native build; do not relabel one binary
as another architecture.

Source archive license terms are preserved. EasyFHE declares GPL-3.0-only
with BSD-3-Clause PyTorch-derived components. Bundled submodules and NVTX
retain their own licenses; see `COUPLET-LICENSES/` and source notices.
