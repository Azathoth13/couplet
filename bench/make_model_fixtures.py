#!/usr/bin/env python3
"""Build the 32-layer real-weight fixtures for bench/t2_model.py in one pass.

``bench/make_layer_fixture.py`` re-runs layers ``0..L-1`` for every target ``L``
(O(L^2) over the model); this script walks the plaintext poly model once
(poly RMSNorm, Alg2 softmax, poly SiLU -- the mathematics the encrypted
stations approximate) over all layers of the canonical prefill128 fixture and
writes, per layer, the nine weight tensors and, once, the whole hidden-state
trajectory:

    <OUT>/layer{i}_weights.npz     input_layernorm, q_proj, k_proj, v_proj, o_proj,
                                   post_attention_layernorm, gate_proj, up_proj, down_proj
                                   (float32, ~878 MB each)
    <OUT>/hidden_trajectory.npz    input_ids, hidden_0 (embedding) .. hidden_{L} (layer L-1 output),
                                   poly_logits (client-side exact final RMSNorm + LM head on the
                                   last token of hidden_{L}), poly_top_tokens
    <OUT>/manifest.json            per-layer OOB statistics, timings, checkpoint name
    --poly-kv-out PATH             (optional) keys / values [layers, 128, 1024] (RoPE'd C-form
                                   sources of every layer) + token_id / logit of the poly
                                   prefill's next token: the ``--poly-kv`` input of
                                   ``bench/t2_decode.py`` (``paris128_prefill_kv_poly.npz``)

``bench/t2_model.py`` streams ``layer{i}_weights.npz`` through the encrypted
station chain and compares its decrypted layer outputs with ``hidden_{i+1}``.
``layer{i}_weights.npz`` + ``hidden_{i}`` / ``hidden_{i+1}`` together are the same
data as ``make_layer_fixture.py``'s ``layer{i}_fixture.npz`` (minus key_rope / value).

Usage::

    python bench/make_model_fixtures.py <ROOT> <OUT> [--layers 32] [--model-dir DIR] [--skip-existing]
                                        [--poly-kv-out fixtures/paris128_prefill_kv_poly.npz]
    python bench/make_model_fixtures.py <ROOT> <OUT> --model llama3.2-1b \
                                        --model-dir assets/weights/Llama-3.2-1B-quarot

``--model`` picks the :mod:`llama3fhe.model_spec` shape (layouts, layer count, RoPE
scaling, polynomial pack under ``assets/polynomials/<model>/``).

``--prefix-kv-cache prefix2_kv.npz`` (``tools/build_prefix_kv_cache.py``) walks the
*content* poly stack with the plaintext attention-sink page of every layer (content RoPE
starts at ``k``, every Softmax row gets ``k`` public columns) -- the oracle trajectory of
``bench/t2_model.py --prefix-kv-cache``.  ``--poly-dir`` swaps the polynomial pack
(e.g. a Prefix-calibrated fit), ``--link-weights DIR`` symlinks ``layer{i}_weights.npz``
from an existing fixture set instead of rewriting them.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root")
    parser.add_argument("out")
    parser.add_argument("--layers", type=int, default=None, help="default: every layer of --model")
    parser.add_argument("--model", default="llama3-8b", help="model spec name (llama3-8b | llama3.2-1b)")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--skip-existing", action="store_true",
                        help="do not rewrite layer{i}_weights.npz files that already exist")
    parser.add_argument("--poly-kv-out", default=None,
                        help="also write the decode --poly-kv npz (keys/values of every layer + next token)")
    parser.add_argument("--prefix-kv-cache", default=None,
                        help="plaintext per-layer Prefix KV page (.npz, tools/build_prefix_kv_cache.py)")
    parser.add_argument("--poly-dir", default=None,
                        help="directory with rmsnorm/silu/softmax_poly_coeffs.json (default: the model's registered pack)")
    parser.add_argument("--poly-rms", default=None, help="rmsnorm table file (overrides --poly-dir)")
    parser.add_argument("--poly-silu", default=None, help="silu table file (overrides --poly-dir)")
    parser.add_argument("--poly-softmax", default=None, help="softmax table file (overrides --poly-dir)")
    parser.add_argument("--link-weights", default=None,
                        help="symlink layer{i}_weights.npz from this fixture dir instead of writing them")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out = Path(args.out).resolve()
    sys.path.insert(0, str(root))

    from llama3fhe.application.args import DEFAULT_INPUT_IDS, model_poly_paths
    from llama3fhe.application.approx_config import ApproximationConfig
    from llama3fhe.application.checkpoint import CheckpointMetadata, WeightLoader, load_layer_weights
    from llama3fhe.application.client import final_logits, top_k_logits
    from llama3fhe.approx.attention import set_default_rope_scaling
    from llama3fhe.approx.layer import poly_prefill_layer_numpy
    from llama3fhe.model_spec import model_spec_by_name

    spec = model_spec_by_name(args.model)
    set_default_rope_scaling(spec.rope_scaling)
    model_dir = (Path(args.model_dir).resolve() if args.model_dir
                 else root / "weights" / "Meta-Llama-3-8b-quarot-safetensors-v2")
    metadata = CheckpointMetadata.from_json(model_dir / "config.json")
    if (int(metadata.hidden_size), int(metadata.intermediate_size)) != (int(spec.hidden_dim), int(spec.intermediate_dim)):
        raise SystemExit(f"{model_dir} is not a {spec.name} checkpoint (hidden {metadata.hidden_size}, "
                         f"intermediate {metadata.intermediate_size}).")
    feature_layout, attention_layout = spec.prefill_layouts()
    n_layers = int(args.layers) if args.layers is not None else int(spec.num_layers)
    fixture = json.loads(Path(DEFAULT_INPUT_IDS).read_text())
    input_ids = np.asarray(fixture["input_ids"], dtype=np.int64)
    assert input_ids.shape == (int(feature_layout.seq_len),)
    rms_path, silu_path, softmax_path = model_poly_paths(spec.name)
    if args.poly_dir:
        poly_dir = Path(args.poly_dir).resolve()
        rms_path, silu_path, softmax_path = (poly_dir / "rmsnorm_poly_coeffs.json", poly_dir / "silu_poly_coeffs.json",
                                             poly_dir / "softmax_poly_coeffs.json")
    rms_path = Path(args.poly_rms).resolve() if args.poly_rms else rms_path
    silu_path = Path(args.poly_silu).resolve() if args.poly_silu else silu_path
    softmax_path = Path(args.poly_softmax).resolve() if args.poly_softmax else softmax_path
    approx = ApproximationConfig(rmsnorm_path=rms_path, silu_path=silu_path, softmax_path=softmax_path)

    prefix_bundles = None
    prefix_k = 0
    if args.prefix_kv_cache:
        from llama3fhe.approx.prefix_cache_io import load_prefix_kv_cache

        prefix_bundles = load_prefix_kv_cache(Path(args.prefix_kv_cache).resolve(), layout=attention_layout,
                                              expected_layers=range(n_layers), expected_rope_theta=float(spec.rope_theta))
        prefix_k = int(prefix_bundles[0].prefix_k)
        print(f"prefix page: {args.prefix_kv_cache} prefix_k={prefix_k} layers={n_layers}", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    link_from = Path(args.link_weights).resolve() if args.link_weights else None
    loader = WeightLoader(model_dir, tensor_dtype=np.float32)
    embedding = loader.get("model.embed_tokens.weight", out_dtype=np.float32)
    hidden = np.asarray(embedding[input_ids], dtype=np.float32)
    del embedding
    print(f"embedding hidden: shape {hidden.shape} max|x| {np.max(np.abs(hidden)):.4g} "
          f"rms {np.sqrt(np.mean(hidden**2)):.4g}", flush=True)

    trajectory = {"input_ids": input_ids, "hidden_0": hidden.copy()}
    per_layer = []
    poly_keys, poly_values = [], []
    total_started = time.perf_counter()
    for idx in range(n_layers):
        started = time.perf_counter()
        weights = load_layer_weights(loader, idx, metadata=metadata)
        load_seconds = time.perf_counter() - started
        weight_path = out / f"layer{idx}_weights.npz"
        if link_from is not None:
            source = link_from / f"layer{idx}_weights.npz"
            if not source.exists():
                raise SystemExit(f"--link-weights: {source} is missing")
            if weight_path.is_symlink() or weight_path.exists():
                weight_path.unlink()
            weight_path.symlink_to(os.path.relpath(source, out))  # relative: the fixture tree is rsync'ed as a whole
        elif not (args.skip_existing and weight_path.exists()):
            np.savez(
                weight_path,
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
        save_seconds = time.perf_counter() - started - load_seconds
        result = poly_prefill_layer_numpy(
            hidden, weights=weights, approximations=approx.for_layer(idx), attention_layout=attention_layout,
            layer_idx=idx, rope_theta=float(metadata.rope_theta), use_alg2_softmax=True,
            prefix_bundle=None if prefix_bundles is None else prefix_bundles[idx],
        )
        hidden = np.asarray(result.hidden, dtype=np.float32)
        trajectory[f"hidden_{idx + 1}"] = hidden.copy()
        if args.poly_kv_out:
            poly_keys.append(np.asarray(result.key, dtype=np.float32))
            poly_values.append(np.asarray(result.value, dtype=np.float32))
        row = {
            "layer": idx, "max_abs": float(np.max(np.abs(hidden))), "rms": float(np.sqrt(np.mean(hidden ** 2))),
            "rmsn1_oob_frac": result.rmsn1_oob_frac, "rmsn2_oob_frac": result.rmsn2_oob_frac,
            "silu_oob_frac": result.silu_oob_frac, "softmax_exp_in_range": result.softmax_exp_in_range,
            "softmax_invsqrt_in_range": result.softmax_invsqrt_in_range,
            "gamma_is_unit": {"input": bool(np.all(weights.input_layernorm == 1.0)),
                              "post": bool(np.all(weights.post_attention_layernorm == 1.0))},
            "seconds": round(time.perf_counter() - started, 1),
            "load_s": round(load_seconds, 2), "save_s": round(save_seconds, 2),
        }
        per_layer.append(row)
        print(json.dumps(row), flush=True)
        del weights

    logits = final_logits(hidden, loader=loader, metadata=metadata)
    top = top_k_logits(logits, 10)
    trajectory["poly_logits"] = np.asarray(logits, dtype=np.float32)
    trajectory["poly_top_tokens"] = np.asarray([int(t["token_id"]) for t in top], dtype=np.int64)
    np.savez(out / "hidden_trajectory.npz", **trajectory)
    if args.poly_kv_out:
        # Same format as the saved reference-KV schema.
        kv_path = Path(args.poly_kv_out).resolve()
        kv_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            kv_path,
            keys=np.stack(poly_keys, axis=0),
            values=np.stack(poly_values, axis=0),
            token_id=np.int64(top[0]["token_id"]),
            logit=np.float32(top[0]["logit"]),
        )
        print(json.dumps({"poly_kv_out": str(kv_path), "keys_shape": list(np.stack(poly_keys, axis=0).shape)}), flush=True)
    manifest = {
        "source": str(model_dir.name), "model": spec.name, "input_fixture": str(Path(DEFAULT_INPUT_IDS).name),
        "layers": int(n_layers), "rope_theta": float(metadata.rope_theta),
        "rope_scaling": None if spec.rope_scaling is None else vars(spec.rope_scaling),
        "polynomials": {"rmsnorm": str(rms_path), "silu": str(silu_path), "softmax": str(softmax_path)},
        "prefix_k": int(prefix_k), "prefix_kv_cache": None if not args.prefix_kv_cache else str(Path(args.prefix_kv_cache).resolve()),
        "rms_eps": float(getattr(metadata, "rms_norm_eps", 1e-5)),
        "weights_dtype": "float32",
        "files": {"weights": "layer{i}_weights.npz", "trajectory": "hidden_trajectory.npz"},
        "trajectory_keys": sorted(trajectory.keys()),
        "poly_top_k": top,
        "per_layer": per_layer,
        "total_seconds": round(time.perf_counter() - total_started, 1),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    loader.close()
    print(json.dumps({"poly_top_k": top[:5], "total_seconds": manifest["total_seconds"], "out": str(out)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
