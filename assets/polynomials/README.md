# Frozen nonlinear tables

Only the tables needed by the supplied model profiles and retained runtime
paths are included. There are **20 coefficient files**, grouped by model and
role. `manifest.json` records SHA-256 and byte counts.

| Model / role | Tables | Use |
|---|---:|---|
| `1b/reference/` | 3 | Original polynomial reference/fixture trajectory |
| `1b/runtime/` | 3 | Final RMSNorm and SiLU; Softmax `exp,j1,k` overlay for layers 0–1 |
| `3b/reference/` | 1 | Original registered RMSNorm reference |
| `3b/runtime/` | 3 | Final RMSNorm and shared SiLU/Softmax for the padded model |
| `8b/reference/` | 3 | Registered reference/fixture and production defaults |
| `8b/runtime/` | 3 | Preset RMSNorm/SiLU and exponential-only Softmax overlays |
| `8b/single_gpu/` | 2 | Reduced-degree single-process RMSNorm/Softmax pack; SiLU is shared with `8b/runtime/` |
| `8b/prefix2/` | 1 | Prefix-2 Softmax path |
| `8b/prefill256/` | 1 | Retained chunked-Prefill Softmax path |

Reference and runtime tables are intentionally distinct. In particular, 1B's
fixture trajectory uses the reference pack, and runtime overlays apply only
the named entries. Merging those artifacts would change the reference or
operator configuration. The manifest identifies each selected table by its
current SHA-256.

The exact model paths and environment overlays are selected by
`configs/models/` and the runtime presets. Prefix and 256-token tables support
existing optional code paths; they are not silently selected by the frozen
P0/128-token benchmarks.

Standalone fitting and degree-retuning tools are outside this artifact.
A new fit, input distribution, or context length requires separate numerical
validation.
