# EasyFHE acceleration extensions

This package contains project-maintained CUDA key-switch and EvalMod kernels,
rotation/BSGS helpers, and host caches. Couplet imports the extensions through
`llama3fhe.acceleration.fhe_ext`. Model-specific SC adapters are provided in
[`baselines/sc/`](../../../baselines/sc/README.md).

`SOURCE_MANIFEST.json` records the source hashes. The EasyFHE backend wheel and
corresponding source are distributed separately as Couplet Release attachments.

The supplied profiles enable these helpers through
[`patches.py`](../patches.py).
[`kernel_sources.py`](../kernel_sources.py) prepares CUDA sources compatible
with the selected decomposition parameters before loading them.

CUDA Toolkit with `nvcc` is required on first use. Set `NVCC` and
`COUPLET_CUDA_ARCH` for the installed compiler and GPU target, such as `sm_120`.
Compilation is cached by source content. Generated shared libraries must not
be committed.

These project sources are covered by the repository GPL-3.0-or-later license.
Their arithmetic follows EasyFHE; see the repository [NOTICE](../../../NOTICE)
and source comments for attribution. The native EasyFHE library is a separate
dependency.
