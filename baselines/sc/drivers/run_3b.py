"""One-card Llama-3.2-3B P0 SC Prefill -> encrypted KV -> warm Decode.

This adapts the verified SC linked runner to 28 layers. The query embedding is
read from an exact, provenance-tagged single-row NPZ. The archived checkpoint
has 4096-wide residuals and 32 padded Q heads. SYLPH_HEAD24_COMPACT=1 loads
the exact 24-head Q/O view while retaining the original calibration.
"""
from __future__ import annotations

import json
import hashlib
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("SYLPH_PUBLIC_PREFIX", "0")
BASE = Path(os.environ.get(
    "SC_LINKED_BASE_DIR", str(Path(__file__).resolve().parent)
))
sys.path.insert(0, str(BASE))
import run_prefill_decode_mv_p0_cipher_kv_gpu as gpu
from query_input import forward_sylph_residual

base = gpu.base


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def query_from_exact_row(manifest, w0, token_id, body_ids_path, row_path):
    body = json.loads(Path(body_ids_path).read_text())["input_ids"]
    if list(body) != list(manifest["input_ids"]) or len(body) != 128:
        raise ValueError("body token IDs differ from the SC Prefill chain")
    archive = np.load(row_path, allow_pickle=False)
    ids = archive["token_ids"].tolist()
    if ids != [int(token_id)]:
        raise ValueError(f"embedding row token IDs {ids} != {[token_id]}")
    embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
    if embeddings.shape != (1, 4096) or not np.isfinite(embeddings).all():
        raise ValueError("query row must contain one finite 4096-wide embedding")
    raw = np.asarray(embeddings[0], dtype=np.float64)
    if "residual_signs" in w0 and "residual_permutation" in w0:
        query = forward_sylph_residual(raw[None], w0["residual_signs"],
                                        w0["residual_permutation"])[0]
        basis = "SC residual signed Hadamard and permutation"
    else:
        # The padded 3B folded-norm chain has no additional Sylph residual transform.
        query = raw
        basis = "QuaRot residual (3B padded folded-norm chain)"
    provenance = json.loads(str(archive["provenance_json"]))
    if provenance.get("source_tensor") != "model.embed_tokens.weight":
        raise ValueError("query row provenance is not the model embedding")
    if provenance.get("source_config_sha256") != manifest.get("source_config_sha256"):
        raise ValueError("query row checkpoint config does not match the SC chain")
    if list(provenance.get("source_shape") or []) != [int(manifest["model"].get("vocab_size", 128256)), 4096]:
        raise ValueError("query row source does not have the padded 3B embedding shape")
    if provenance.get("token_ids") != [int(token_id)]:
        raise ValueError("query row provenance token ID differs from its payload")
    dtype = manifest.get("expected_next_token_embedding_row_dtype")
    if dtype not in ("F16", "F32") or provenance.get("source_dtype") != dtype:
        raise ValueError("query row source dtype differs from the SC checkpoint")
    expected_sha = manifest.get("expected_next_token_embedding_row_sha256")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise ValueError("SC chain lacks an expected embedding-row SHA256")
    actual_sha = hashlib.sha256(embeddings[0].astype("<f2" if dtype == "F16" else "<f4").tobytes()).hexdigest()
    if (provenance.get("source_row_sha256") != [expected_sha] or actual_sha != expected_sha):
        raise ValueError("query embedding values do not match the SC checkpoint row")
    return query, {
        "token_id": int(token_id),
        "body_tokens": len(body),
        "body_ids_match_chain": True,
        "embedding_row_path": str(Path(row_path).resolve()),
        "embedding_source_row_sha256": expected_sha,
        "query_basis": basis,
    }


def main():
    args = base.args_from_cli()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest, entries = base.layers()
    if (manifest.get("model_name") != "llama3.2-3b-padded-quarot"
            or len(entries) != 28 or [e["layer"] for e in entries] != list(range(28))):
        raise ValueError("expected the 28-layer padded Llama-3.2-3B SC chain")
    model = manifest.get("model") or {}
    if (int(model.get("hidden", -1)), int(model.get("heads", -1)),
            int(model.get("kv_heads", -1)), int(model.get("head_dim", -1))) != (4096, 32, 8, 128):
        raise ValueError("SC chain physical geometry does not match padded Llama-3.2-3B")
    if (manifest.get("prefix_token_ids") not in (None, [])
            or manifest.get("public_prefix_token_ids") not in (None, [])
            or int(manifest.get("public_prefix_tokens", 0)) != 0):
        raise ValueError("the SC chain must not contain public-prefix tokens")
    plain_inputs = np.load(base.PLAIN_INPUTS)
    if any(f"layer{i:02d}" not in plain_inputs for i in range(28)):
        raise ValueError("the P0 chain lacks a complete validation trajectory")
    for entry in entries:
        base.unstage(entry)
    w0, cal0 = base.load(entries[0], staged=False)
    head_dim, heads, kv_heads = base.geometry(manifest)
    geo = base.MVGeometry(int(w0["q"].shape[1]), int(w0["gate"].shape[0]),
                          heads, kv_heads, head_dim)
    first, last = (0, 0) if args.preflight_only else (0, 27)
    prefill, client, eng, saved, cache_meta = base.capture_prefill_caches(
        geo, first=first, last=last, out=out,
        diagnose=os.environ.get("SC_PREFILL_DIAGNOSE", "0") == "1")
    (out / "prefill_cipher_kv_meta.json").write_text(json.dumps(cache_meta, indent=1))
    preflight = base.cache_preflight(client, eng, geo, saved[0], w0, cal0,
                                     plain_inputs["layer00"], atol=args.cache_atol)
    (out / "cache_layer0_preflight.json").write_text(json.dumps(preflight, indent=1))
    if not preflight["passed"]:
        raise RuntimeError(f"encrypted Prefill KV preflight failed: {preflight}")
    if args.preflight_only:
        print(json.dumps({"preflight_only": True, "cache": preflight,
                          "prefill_layer_seconds": prefill["layers"][0]["seconds"]}), flush=True)
        return
    comparison = prefill["terminal"]["comparison"]
    expected_query_token = int(manifest["expected_next_token_id"])
    if any(arg.startswith("--query-token-id") for arg in sys.argv[1:]) and args.query_token_id != expected_query_token:
        raise ValueError("--query-token-id does not match the chain's plaintext terminal prediction")
    if (len(prefill["layers"]) != 28 or prefill.get("next_token_id") != expected_query_token
            or not comparison["passed"] or not comparison["top_k_match"]):
        raise RuntimeError("SC 3B encrypted Prefill did not produce the expected query token")
    if not os.environ.get("SC_QUERY_EMBEDDING_ROW"):
        raise ValueError("set SC_QUERY_EMBEDDING_ROW to the provenance-tagged embedding row NPZ for the predicted token")
    row_path = Path(os.environ["SC_QUERY_EMBEDDING_ROW"])
    if not row_path.is_file():
        raise FileNotFoundError(row_path)
    query, query_report = query_from_exact_row(
        manifest, w0, prefill["next_token_id"], args.body_ids, row_path)
    del w0, cal0
    expected_decode_token = (args.expected_next_token_id
                             if any(arg.startswith("--expected-next-token-id") for arg in sys.argv[1:])
                             else None)
    results = []
    for index in range(args.warmup_passes + args.measure_passes):
        result = base.decode_pass(
            client, eng, geo, manifest, entries, plain_inputs, saved, query,
            index=index, measured=index >= args.warmup_passes,
            query_token_id=prefill["next_token_id"],
            expected_token=expected_decode_token, out=out)
        if index == 0:
            # The upstream linked runner saves only measured replay passes.
            # Preserve the actual first-token report for a matched comparison.
            (out / "decode_cipher_kv_pass0.json").write_text(json.dumps(result, indent=1))
        results.append(result)
    offload_total = sum(sum(m["cache_cpu_offload_seconds_in_prefill_forward"].values())
                        for m in cache_meta)
    summary = dict(
        model=manifest.get("model_name"), prefix_tokens=0,
        source_manifest_sha256=sha256_file(Path(base.CHAIN) / 'chain-manifest.json'),
        source_plain_inputs_sha256=sha256_file(base.PLAIN_INPUTS),
        physical_padding={"hidden": 4096, "query_heads": 32, "logical_hidden": 3072,
                          "logical_query_heads": 24},
        operator_query_heads=int(geo.heads),
        packing_arm="compact24" if os.environ.get("SYLPH_HEAD24_COMPACT") == "1" else "padded32",
        prefill_next_token_id=prefill["next_token_id"],
        prefill_terminal_passed=bool(comparison["passed"] and comparison["top_k_match"]),
        prefill_layer_seconds_total=round(sum(r["seconds"] for r in prefill["layers"]), 3),
        prefill_chain_seconds=prefill["chain_seconds"],
        prefill_cache_cpu_offload_seconds_total=round(offload_total, 3),
        prefill_cache_layers=cache_meta,
        kv_host_bytes=sum(m["k_host_bytes"] + m["v_host_bytes"] for m in cache_meta),
        query=query_report, cache_source="FHE Prefill emitted ciphertext",
        kv_from_fhe_prefill=True,
        session_mode="same_state_replay_from_one_encrypted_prefill",
        cache_layer0_preflight=preflight,
        warmup_layer_seconds=[r["layer_seconds_total"] for r in results if not r["measured"]],
        warmup_next_token_ids=[r["next_token_id"] for r in results if not r["measured"]],
        measured_layer_seconds=[r["layer_seconds_total"] for r in results if r["measured"]],
        measured_restore_seconds=[r["restore_seconds_total"] for r in results if r["measured"]],
        measured_next_token_ids=[r["next_token_id"] for r in results if r["measured"]],
    )
    (out / "prefill_decode_cipher_kv_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({"summary": summary}), flush=True)


if __name__ == "__main__":
    main()
