#!/usr/bin/env python3
"""Rebuild large Couplet fixture and SC1B/3B chain weight files.

The hidden trajectory, calibration and manifests are small inputs copied from
the frozen experiment. This extracts the exact linear weights from the rotated
checkpoint using the same loader and NPZ field order as
``bench/make_model_fixtures.py`` and ``make_1b_chain.py``.

SC8B uses an additional residual/value rotation; pass ``--skip-sc`` and use
``baselines/sc/install_assets.py`` for that model's SC weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--odin-root", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--fixture-dir", required=True, type=Path)
    parser.add_argument("--sc-chain", type=Path)
    parser.add_argument("--skip-sc", action="store_true",
                        help="write only Couplet fixtures (SC8B needs its distinct rotation builder)")
    parser.add_argument("--layers", required=True, type=int)
    args = parser.parse_args()
    if not args.skip_sc and args.sc_chain is None:
        parser.error("provide --sc-chain or --skip-sc")

    sys.path.insert(0, str(args.odin_root.resolve()))
    from llama3fhe.application.checkpoint import (  # noqa: PLC0415
        CheckpointMetadata, WeightLoader, load_layer_weights,
    )

    metadata = CheckpointMetadata.from_json(args.model_dir / "config.json")
    if int(metadata.num_hidden_layers) != args.layers:
        raise ValueError(f"checkpoint has {metadata.num_hidden_layers} layers, expected {args.layers}")
    args.fixture_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_sc:
        args.sc_chain.mkdir(parents=True, exist_ok=True)
    loader = WeightLoader(args.model_dir, tensor_dtype=np.float32)
    kv_shape = (0, int(metadata.num_key_value_heads),
                int(metadata.hidden_size) // int(metadata.num_attention_heads))
    try:
        for index in range(args.layers):
            weights = load_layer_weights(loader, index, metadata=metadata)
            fixture = args.fixture_dir / f"layer{index}_weights.npz"
            np.savez(
                fixture,
                input_layernorm=np.asarray(weights.input_layernorm, dtype=np.float32),
                q_proj=np.asarray(weights.q_proj, dtype=np.float32),
                k_proj=np.asarray(weights.k_proj, dtype=np.float32),
                v_proj=np.asarray(weights.v_proj, dtype=np.float32),
                o_proj=np.asarray(weights.o_proj, dtype=np.float32),
                post_attention_layernorm=np.asarray(weights.post_attention_layernorm, dtype=np.float32),
                gate_proj=np.asarray(weights.gate_proj, dtype=np.float32),
                up_proj=np.asarray(weights.up_proj, dtype=np.float32),
                down_proj=np.asarray(weights.down_proj, dtype=np.float32),
            )
            if args.skip_sc:
                print(json.dumps({"layer": index, "fixture_sha256": sha256(fixture)}), flush=True)
                continue
            sc_layer = args.sc_chain / f"layer-{index:02d}"
            sc_layer.mkdir(exist_ok=True)
            np.savez(
                sc_layer / "weights.npz",
                q=np.asarray(weights.q_proj, dtype=np.float32),
                k=np.asarray(weights.k_proj, dtype=np.float32),
                v=np.asarray(weights.v_proj, dtype=np.float32),
                o=np.asarray(weights.o_proj, dtype=np.float32),
                gate=np.asarray(weights.gate_proj, dtype=np.float32),
                up=np.asarray(weights.up_proj, dtype=np.float32),
                down=np.asarray(weights.down_proj, dtype=np.float32),
                attention_norm=np.asarray(weights.input_layernorm, dtype=np.float32),
                ffn_norm=np.asarray(weights.post_attention_layernorm, dtype=np.float32),
                public_keys=np.empty(kv_shape, dtype=np.float32),
                public_values=np.empty(kv_shape, dtype=np.float32),
            )
            print(json.dumps({
                "layer": index,
                "fixture_sha256": sha256(fixture),
                "sc_sha256": sha256(sc_layer / "weights.npz"),
            }), flush=True)
        head = None if args.skip_sc else loader.get("lm_head.weight", out_dtype=np.float32)
    finally:
        loader.close()

    if not args.skip_sc:
        head_path = args.sc_chain / "lm_head.npy"
        np.save(head_path, head)
        print(json.dumps({"lm_head_sha256": sha256(head_path)}), flush=True)


if __name__ == "__main__":
    main()
