# Weights

Model files are not distributed. The supported base checkpoints are
`meta-llama/Llama-3.2-1B`, `meta-llama/Llama-3.2-3B`, and
`meta-llama/Meta-Llama-3-8B`. Accept the corresponding Meta model license and
use policy before downloading. Exact revisions and weight hashes are in
`configs/models/`. See `tools/models/README.md` for the complete preparation
flow and configurable data paths.

For 8B, `tools/convert/prepare_quarot_weights.sh` writes:

- `Meta-Llama-3-8B-QuaRot-fp16/`: rotated intermediate checkpoint;
- `Meta-Llama-3-8B-QuaRot/`: standard HF safetensors, full-fused for FHE.

The final matrices retain ordinary `[out,in]` shapes. Feature-major/complex
diagonal plaintext packing is generated online by `LinearWeight` and is not
serialized into the checkpoint.

1B uses a native 2048-feature converted checkpoint. 3B uses the official
3072-feature model converted to an equivalent 4096-feature physical layout;
that padding is part of the published implementation. SC8B applies an
additional residual/value rotation when constructing its own packing assets.
