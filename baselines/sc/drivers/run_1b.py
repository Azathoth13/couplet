"""One-card Llama-3.2-1B P0 SC Prefill -> encrypted KV -> warm Decode.

This adapts the verified 8B linked runner to 16 layers. The SC chain manifest's
original full checkpoint path is unavailable on the GPU host, so the query
embedding is read from an exact, provenance-tagged single-row NPZ instead.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

BASE = Path(os.environ.get(
    "SC_LINKED_BASE_DIR", str(Path(__file__).resolve().parent)
))
sys.path.insert(0, str(BASE))
import run_prefill_decode_mv_p0_cipher_kv_gpu as gpu
from query_input import forward_sylph_residual

base = gpu.base


def query_from_exact_row(manifest, w0, token_id, body_ids_path, row_path):
    body = json.loads(Path(body_ids_path).read_text())["input_ids"]
    if list(body) != list(manifest["input_ids"]) or len(body) != 128:
        raise ValueError("body token IDs differ from the SC Prefill chain")
    archive = np.load(row_path, allow_pickle=False)
    ids = archive["token_ids"].tolist()
    if ids != [int(token_id)]:
        raise ValueError(f"embedding row token IDs {ids} != {[token_id]}")
    raw = np.asarray(archive["embeddings"][0], dtype=np.float64)
    if "residual_signs" in w0 and "residual_permutation" in w0:
        query = forward_sylph_residual(raw[None], w0["residual_signs"],
                                        w0["residual_permutation"])[0]
        basis = "SC residual signed Hadamard and permutation"
    else:
        # The 1B folded-norm chain has no additional Sylph residual transform.
        query = raw
        basis = "QuaRot residual (1B folded-norm chain)"
    provenance = json.loads(str(archive["provenance_json"]))
    if provenance.get("source_tensor") != "model.embed_tokens.weight":
        raise ValueError("query row provenance is not the model embedding")
    return query, {
        "token_id": int(token_id),
        "body_tokens": len(body),
        "body_ids_match_chain": True,
        "embedding_row_path": str(Path(row_path).resolve()),
        "embedding_source_row_sha256": provenance["source_row_sha256"][0],
        "query_basis": basis,
    }


def main():
    args = base.args_from_cli()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest, entries = base.layers()
    if len(entries) != 16 or [e["layer"] for e in entries] != list(range(16)):
        raise ValueError("expected the 16-layer Llama-3.2-1B SC chain")
    if manifest.get("prefix_token_ids") not in (None, []):
        raise ValueError("the SC chain must not contain public-prefix tokens")
    plain_inputs = np.load(base.PLAIN_INPUTS)
    if any(f"layer{i:02d}" not in plain_inputs for i in range(16)):
        raise ValueError("the P0 chain lacks a complete validation trajectory")
    for entry in entries:
        base.unstage(entry)
    w0, cal0 = base.load(entries[0], staged=False)
    head_dim, heads, kv_heads = base.geometry(manifest)
    geo = base.MVGeometry(int(w0["q"].shape[1]), int(w0["gate"].shape[0]),
                          heads, kv_heads, head_dim)
    first, last = (0, 0) if args.preflight_only else (0, 15)
    prefill, client, eng, saved, cache_meta = base.capture_prefill_caches(
        geo, first=first, last=last, out=out)
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
    if (len(prefill["layers"]) != 16 or prefill.get("next_token_id") != args.query_token_id
            or not comparison["passed"] or not comparison["top_k_match"]):
        raise RuntimeError("SC 1B encrypted Prefill did not produce the expected query token")
    row_path = Path(os.environ.get(
        "SC_QUERY_EMBEDDING_ROW",
        str(Path(base.CHAIN).parent / "input/query_embedding.npz")))
    query, query_report = query_from_exact_row(
        manifest, w0, prefill["next_token_id"], args.body_ids, row_path)
    del w0, cal0
    results = []
    for index in range(args.warmup_passes + args.measure_passes):
        result = base.decode_pass(
            client, eng, geo, manifest, entries, plain_inputs, saved, query,
            index=index, measured=index >= args.warmup_passes,
            query_token_id=prefill["next_token_id"],
            expected_token=args.expected_next_token_id, out=out)
        results.append(result)
    offload_total = sum(sum(m["cache_cpu_offload_seconds_in_prefill_forward"].values())
                        for m in cache_meta)
    summary = dict(
        model=manifest.get("model_name"), prefix_tokens=0,
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
        measured_layer_seconds=[r["layer_seconds_total"] for r in results if r["measured"]],
        measured_restore_seconds=[r["restore_seconds_total"] for r in results if r["measured"]],
        measured_next_token_ids=[r["next_token_id"] for r in results if r["measured"]],
    )
    (out / "prefill_decode_cipher_kv_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({"summary": summary}), flush=True)


if __name__ == "__main__":
    main()
