#!/usr/bin/env python3
"""Export the exact plaintext Prefix KV pages consumed by T2 Prefill/Decode.

Example::

    python tools/build_prefix_kv_cache.py \\
        --model-dir weights/Meta-Llama-3-8b-quarot-safetensors-v2 \\
        --output fixtures/model/prefix2_kv.npz --layers 32
    python tools/build_prefix_kv_cache.py --model llama3.2-1b \\
        --model-dir assets/weights/Llama-3.2-1B-quarot \\
        --output fixtures/llama32_1b_model/prefix2_kv.npz

The k sink tokens run through every layer in plaintext. Only their RoPE'd K
and unrotated V are saved; content tokens never enter this computation.
``--model`` picks the :mod:`llama3fhe.model_spec` geometry (head_dim, heads,
RoPE scaling); the 8B default also requires the prepared runtime export.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llama3fhe.application.checkpoint import (  # noqa: E402
    CheckpointMetadata,
    WeightLoader,
    load_layer_weights,
    validate_runtime_checkpoint,
)
from llama3fhe.approx.prefix_cache_io import (  # noqa: E402
    model_cache_metadata,
    save_prefix_kv_cache,
)
from llama3fhe.approx.prefix_stack import (  # noqa: E402
    prefix_token_ids as repeated_prefix_token_ids,
    run_exact_prefix_layer,
)
from llama3fhe.approx.attention import set_default_rope_scaling  # noqa: E402
from llama3fhe.model_spec import model_spec_by_name  # noqa: E402


def export_prefix_kv_cache(
    model_dir: str | Path,
    output: str | Path,
    *,
    num_layers: int | None = None,
    prefix_k: int = 2,
    prefix_token_id: int = 198,
    prefix_token_ids: Sequence[int] | None = None,
    model: str = "llama3-8b",
) -> None:
    spec = model_spec_by_name(model)
    model_dir = Path(model_dir).resolve()
    metadata = CheckpointMetadata.from_json(model_dir / "config.json")
    num_layers = int(spec.num_layers if num_layers is None else num_layers)
    if spec.name == "llama3-8b":
        validate_runtime_checkpoint(model_dir)
        metadata.validate_llama3_8b(num_layers=num_layers)
    elif (int(metadata.hidden_size), int(metadata.intermediate_size)) != (int(spec.hidden_dim), int(spec.intermediate_dim)):
        raise ValueError(f"{model_dir} is not a {spec.name} checkpoint (hidden {metadata.hidden_size}, "
                         f"intermediate {metadata.intermediate_size}).")
    set_default_rope_scaling(spec.rope_scaling)
    _, layout = spec.prefill_layouts()
    if not 0 < prefix_k <= layout.seq_len:
        raise ValueError(f"prefix_k must be in 1..{layout.seq_len}.")

    with WeightLoader(model_dir, tensor_dtype=np.float32) as loader:
        embedding = loader.get("model.embed_tokens.weight", out_dtype=np.float32)
        token_ids = (repeated_prefix_token_ids(prefix_k, token_id=prefix_token_id)
                     if prefix_token_ids is None else np.asarray(prefix_token_ids, dtype=np.int64))
        if token_ids.shape != (prefix_k,):
            raise ValueError(f"Expected {prefix_k} prefix token IDs, got shape {token_ids.shape}.")
        if np.any(token_ids < 0) or np.any(token_ids >= embedding.shape[0]):
            raise ValueError(f"prefix_token_ids={token_ids.tolist()} are outside the vocabulary.")
        hidden = np.asarray(embedding[token_ids], dtype=np.float32).copy()
        del embedding
        keys: list[np.ndarray] = []
        values: list[np.ndarray] = []
        for layer_idx in range(int(num_layers)):
            weights = load_layer_weights(loader, layer_idx, metadata=metadata)
            hidden, key, value = run_exact_prefix_layer(
                hidden,
                weights=weights,
                layout=layout,
                rope_theta=float(metadata.rope_theta),
                eps=float(metadata.rms_norm_eps),
            )
            keys.append(np.asarray(key, dtype=np.float32))
            values.append(np.asarray(value, dtype=np.float32))
            del weights
            print(json.dumps({"layer": layer_idx, "prefix_k": prefix_k}), flush=True)

    key_array = np.stack(keys, axis=0)
    value_array = np.stack(values, axis=0)
    save_prefix_kv_cache(
        output,
        key_array,
        value_array,
        layers=range(int(num_layers)),
        prefix_k=prefix_k,
        prefix_token_id=prefix_token_id,
        prefix_token_ids=token_ids,
        rope_theta=metadata.rope_theta,
        model_metadata=model_cache_metadata(model_dir),
    )
    print(json.dumps({"output": str(Path(output).resolve()), "keys_shape": list(key_array.shape)}), flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="llama3-8b", help="model spec name (llama3-8b | llama3.2-1b)")
    parser.add_argument("--layers", type=int, default=None, help="default: every layer of --model")
    parser.add_argument("--prefix-k", type=int, default=2)
    token_group = parser.add_mutually_exclusive_group()
    token_group.add_argument("--prefix-token-id", type=int, default=198)
    token_group.add_argument("--prefix-token-ids", type=str, help="comma-separated IDs, one per prefix token")
    args = parser.parse_args(argv)
    export_prefix_kv_cache(
        args.model_dir,
        args.output,
        num_layers=args.layers,
        prefix_k=args.prefix_k,
        prefix_token_id=args.prefix_token_id,
        prefix_token_ids=(None if args.prefix_token_ids is None
                          else [int(part.strip()) for part in args.prefix_token_ids.split(",")]),
        model=args.model,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
