#!/usr/bin/env python3
"""Extract a few client embedding rows without loading the full embedding table."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument("--token-ids", type=int, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.checkpoint / "config.json"
    token_ids = args.token_ids
    if token_ids is None:
        if args.fixtures is None:
            parser.error("provide --fixtures or --token-ids")
        manifest = json.loads((args.fixtures / "manifest.json").read_text())
        token_ids = [int(manifest["poly_top_k"][0]["token_id"])]
    token_ids = list(dict.fromkeys(token_ids))
    index = args.checkpoint / "model.safetensors.index.json"
    tensor_name = "model.embed_tokens.weight"
    weight_file = (json.loads(index.read_text())["weight_map"][tensor_name]
                   if index.exists() else "model.safetensors")
    from safetensors import safe_open

    with safe_open(args.checkpoint / weight_file, framework="np") as archive:
        tensor = archive.get_slice(tensor_name)
        shape, dtype = list(tensor.get_shape()), str(tensor.get_dtype())
        if any(token < 0 or token >= shape[0] for token in token_ids):
            raise ValueError("embedding token IDs are outside the vocabulary")
        rows = np.concatenate([tensor[token:token + 1] for token in token_ids], axis=0)
    if dtype not in {"F16", "F32"}:
        raise ValueError("extract rows from a converted F16/F32 checkpoint")
    provenance = {"source_tensor": tensor_name, "source_shape": shape, "source_dtype": dtype,
                  "source_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                  "source_row_sha256": [hashlib.sha256(row.tobytes()).hexdigest() for row in rows],
                  "token_ids": token_ids}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, token_ids=np.asarray(token_ids, dtype=np.int64),
             embeddings=rows.astype(np.float32), provenance_json=json.dumps(provenance, sort_keys=True))
    print(json.dumps({"output": str(args.output), "token_ids": token_ids,
                      "shape": list(rows.shape)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
