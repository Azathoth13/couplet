# Three-model preparation

The public model profiles are `configs/models/{1b,3b,8b}.json`. All paths below
are generated under a caller-selected data directory; weights and NPZ files
are not part of Git.

```bash
# Activate the FHE environment for the complete preparation sequence below.
# HF access must already be authorized through `hf auth login` or HF_TOKEN.
python -m pip install 'huggingface_hub>=0.35,<1'
python tools/models/prepare.py --model 3b --data-root /path/to/data --stage all --dry-run

# Run stages explicitly, or use --stage all for this same sequence.
python tools/models/prepare.py --model 3b --data-root /path/to/data --stage download
python tools/models/prepare.py --model 3b --data-root /path/to/data --stage convert
python tools/models/prepare.py --model 3b --data-root /path/to/data --stage fixtures
python tools/models/prepare.py --model 3b --data-root /path/to/data --stage client-rows
python tools/models/prepare.py --model 3b --data-root /path/to/data --stage linear
python tools/models/prepare.py --model 3b --data-root /path/to/data --stage sc
```

Replace `3b` with `1b` or `8b`. `--raw DIR` accepts an already downloaded
checkpoint; `--checkpoint DIR` sets the converted checkpoint location. These
overrides apply to preparation only: pass the corresponding path override to
the benchmark runner too. `--python` selects the preparation interpreter.

Download and the standalone 3B converter run in a normal NumPy environment.
For that separate CPU environment, `tools/convert/llama32/requirements_3b_cpu.txt`
provides its requirements; do not use it to replace the locked FHE runtime.
The 1B converter's enabled equivalence verifier, fixture generation, linear
asset extraction and SC preparation import the EasyFHE-backed runtime package.
They therefore need an environment with EasyFHE installed, even when their
arithmetic runs on CPU. Use the activated FHE Python for `--stage all` and
select a separate standard PyTorch Python with `--rotate-python` for 8B.
Install the backend using the [backend instructions](../../docs/BACKEND.md)
before running those stages.

The downloader defaults to a pinned official Meta base model, verifies every
weight file against its published SHA-256 and byte count, and writes a local
`download_manifest.json`. Pinned mirror revisions are recorded in
the profiles. `--source frozen-mirror` is accepted only when their weight
blobs match the pinned official files; it never silently changes the model.
Config metadata can differ between sources even when the weight blobs match.

1B uses the NumPy QuaRot converter, seed 0, float32, with embedding mean
centring disabled. 3B starts from the official 3072-feature/24-query-head
checkpoint and exports the equivalent 4096-feature/32-physical-head layout;
epsilon scaling and head remapping are part of the converter. Its small
`--self-test` checks padding and logit equivalence without model downloads.

8B uses a two-stage QuaRot conversion. Before its `convert`
stage, obtain the pinned QuaRot checkout:

```bash
git clone --no-checkout https://github.com/spcl/QuaRot.git tools/third_party/QuaRot
git -C tools/third_party/QuaRot checkout 5008669b08c1f11f9b64d52d16fddd47ca754c5a
```

This upstream revision's conversion-source hashes are recorded in the 8B
profile. Follow
`tools/third_party/README.md`
and install `tools/convert/requirements-rotate.txt` in a **standard CUDA
PyTorch** environment, including `fast-hadamard-transform`. Select that Python
with `--rotate-python /path/to/python`; EasyFHE's torch shim cannot perform
this conversion. This stage uses a GPU. No encrypted benchmark is launched by
the preparation tool.

`fixtures` creates Couplet's layer weights, polynomial reference trajectory and
reference KV. The benchmark's linked path still obtains its Decode cache
from encrypted Prefill; the reference KV is only for checks. `client-rows`
extracts the first generated token's embedding as a small, provenance-tagged
NPZ and is independent of SC. This selected row supports the fixed-input
first-Decode / hot-replay benchmark. For continuous generation, use the full
checkpoint embedding lookup or explicitly extract all required token IDs.

`linear` creates SC1B/3B linear matrices as well. **SC8B additionally changes
the residual and value basis**, so this stage deliberately skips its SC
matrices. `sc` installs the frozen calibration, prepares the model-specific
SC weights and computes SC's own reference trajectory. It must not substitute
the Couplet reference trajectory for SC's different polynomial computation.

The artifact supplies the selected polynomial coefficients, intervals, degrees,
pre-scales, round counts and model/layer mappings used by the frozen profiles.
Model preparation loads these results; it does not rerun offline coefficient
fitting, calibration, degree retuning or joint error-budget/PPL search. Those
research tools are outside this artifact's scope.

Generic polynomial evaluation and the SC runtime's initialization of plans from
frozen calibration data remain included because inference uses them. These
runtime helpers are distinct from the offline search procedure. The tables are
calibrated for the documented 128-token input; arbitrary prompts or context
lengths have not been established by this release.
