"""SC Prefix=0 encrypted Prefill -> encrypted KV cache -> same-token Decode.

This experiment keeps one EasyFHE client/context for both phases. It captures the
already computed K-after-RoPE and V GQA expansions during encrypted Prefill;
there is no plaintext KV on the ciphertext Decode path. The independent plain
trajectory is used only for error checks. The first complete Decode pass warms
the GPU and is excluded from the measured median.

Example on the SC GPU host::

    SYLPH_ROOT=/path/to/sc/provider \
    SYLPH_CHAIN=/path/to/assets/sc/8b/chain \
    SYLPH_PLAIN_INPUTS=/path/to/assets/sc/8b/chain/plain_inputs.npz \
    SYLPH_PUBLIC_PREFIX=0 \
    python run_prefill_decode_mv_p0_cipher_kv.py \
      --body-ids=assets/prefill128_input_ids.json \
      --out=artifacts/sc

Use --preflight-only to run just Prefill layer 0 and compare its encrypted KV
cache against a plaintext reference before attempting the full 32-layer run.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import gc
import json
import os
import runpy
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ["SYLPH_WARMUP"] = "1"
if os.environ.get("SYLPH_PUBLIC_PREFIX") != "0":
    raise ValueError("SYLPH_PUBLIC_PREFIX=0 is required for this P0 experiment")
sys.path.insert(0, os.environ.get("SYLPH_ROOT", str(Path(__file__).resolve().parents[1] / "providers/llama8b")))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import easyfhe as torch
import numpy as np

from sylph_easy.chain.manifest import CHAIN, PLAIN_INPUTS, geometry, layers, load, unstage
from sylph_easy.easyfhe_ext.engine import Engine
from sylph_easy.layer import plain, setup
from sylph_easy.layer.decoder import DecoderLayer
from sylph_easy.layer.decode import expanded_cache
from sylph_easy.layer.decode_mv import DecodeMVLayer, DeviceProjector, MVGeometry
from sylph_easy.layer.layout import HeadGeometry
from sylph_easy.layer.setup import SLOTS
from query_input import prepare_query


def args_from_cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--body-ids", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--query-token-id", type=int, default=12366)
    p.add_argument("--expected-next-token-id", type=int, default=13)
    p.add_argument("--warmup-passes", type=int, default=1)
    p.add_argument("--measure-passes", type=int, default=2)
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--cache-atol", type=float, default=0.05)
    a = p.parse_args()
    if a.warmup_passes < 1 or a.measure_passes < 1:
        p.error("at least one full-token warmup and one measured pass are required")
    if not np.isfinite(a.cache_atol) or a.cache_atol < 0:
        p.error("--cache-atol must be finite and nonnegative")
    return a


def cpu_cipher(ct):
    return ct.cipher_like([part.detach().to("cpu") for part in ct.cv], batch_size=ct.batch_size)


def gpu_cipher(ct):
    return ct.cipher_like([part.to("cuda") for part in ct.cv], batch_size=ct.batch_size)


def capture_prefill_caches(geo, *, first, last, out, diagnose=False):
    """Run the unchanged Prefill chain, intercepting its two GQA _expand results.

    The K/V copies are parked on the CPU during the timed attention call to
    avoid retaining their GPU storage through Softmax or later layer stages.
    Their transfer time is recorded separately in cache_meta.
    """
    original_create = setup.create
    original_forward = DecoderLayer.forward
    original_clear = Engine.clear_cache
    original_pool = futures.ThreadPoolExecutor
    pools = []
    pending = []
    parked = []
    cache_meta = []
    forward_calls = 0

    def create_with_decode_keys(*a, **kw):
        kw["extra_rotations"] = set(kw.get("extra_rotations", ())) | geo.rotations()
        return original_create(*a, **kw)

    def tracked_pool(*a, **kw):
        pool = original_pool(*a, **kw)
        pools.append(pool)
        return pool

    def forward_capture(layer, x, *a, **kw):
        nonlocal forward_calls
        # run_chain.py makes one real first-layer warmup call before its timed loop.
        active = forward_calls > 0
        forward_calls += 1
        if not active:
            return original_forward(layer, x, *a, **kw)
        calls = []
        att = layer.attention
        original_expand = att._expand
        original_softmax = att.softmax
        original_sv = att._sv
        offload_seconds = {"k": 0.0, "v": 0.0}
        cache_recompute_seconds = {"k": 0.0, "v": 0.0}
        shard = getattr(layer, "_attention_shard", None)
        full_att = getattr(layer, "_full_attention_for_cache", None)
        captured_inputs = {}

        def expand_capture(kv):
            expanded = original_expand(kv)
            if shard is not None:
                t_cache = time.time()
                cached = full_att._expand(kv)
                cache_recompute_seconds[("k", "v")[len(calls)]] = time.time() - t_cache
                calls.append(cached)
            else:
                calls.append(expanded)
            return expanded

        def park_call(which, slot):
            if len(calls) != slot + 1:
                raise RuntimeError(f"Prefill {which} cache expansion order changed")
            t0 = time.time()
            calls[slot] = tuple(cpu_cipher(ct) for ct in calls[slot])
            offload_seconds[which] = time.time() - t0

        def softmax_capture(*sm_args, **sm_kwargs):
            # QK has finished; keeping expanded K on the GPU through Softmax
            # defeats Attention.forward's deliberate early K release.
            park_call("k", 0)
            return original_softmax(*sm_args, **sm_kwargs)

        def sv_capture(*sv_args, **sv_kwargs):
            value = original_sv(*sv_args, **sv_kwargs)
            park_call("v", 1)
            return value

        att._expand = expand_capture
        att.softmax = softmax_capture
        att._sv = sv_capture
        if shard is not None and shard[0] == shard[1]:
            if a or "probe" in kw:
                raise RuntimeError("idle attention cache capture needs an unoccupied probe")

            def capture_probe(name, cipher):
                if name in ("k", "v"):
                    captured_inputs[name] = cipher

            kw["probe"] = capture_probe
        try:
            result = original_forward(layer, x, *a, **kw)
        finally:
            att._expand = original_expand
            att.softmax = original_softmax
            att._sv = original_sv
        if shard is not None and shard[0] == shard[1]:
            # W8 compact24 has two idle Q-shard ranks.  Their QKV projection
            # still receives the complete K/V batch; construct the same full
            # cache locally so no cache ciphertext collective is needed.
            if set(captured_inputs) != {"k", "v"}:
                raise RuntimeError("idle attention rank did not capture full K/V inputs")
            b = layer.b
            t_cache = time.time()
            k_lo, k_hi = full_att._halves(b.trim(captured_inputs["k"], 6))
            kr = full_att._interleave(full_att._rope(k_lo, 0), full_att._rope(k_hi, 1))
            calls.append(tuple(cpu_cipher(ct) for ct in full_att._expand(kr)))
            cache_recompute_seconds["k"] = time.time() - t_cache
            t_cache = time.time()
            v_lo, v_hi = full_att._halves(b.trim(captured_inputs["v"], 3))
            vr = full_att._interleave(v_lo, v_hi)
            calls.append(tuple(cpu_cipher(ct) for ct in full_att._expand(vr)))
            cache_recompute_seconds["v"] = time.time() - t_cache
        if len(calls) != 2 or pending:
            raise RuntimeError(f"Prefill attention emitted {len(calls)} cache pairs; expected K then V")
        pending.append((len(parked) + first, {"k": calls[0], "v": calls[1]},
                        offload_seconds, cache_recompute_seconds))
        return result

    def clear_and_park(eng):
        if pending:
            layer_idx, live, offload_seconds, cache_recompute_seconds = pending.pop()
            info = {"layer": layer_idx, "k_limbs": [], "v_limbs": [], "q_shards": geo.n_q,
                    "cache_cpu_offload_seconds_in_prefill_forward": offload_seconds,
                    "cache_recompute_seconds_in_prefill_forward": cache_recompute_seconds}
            saved = {}
            for name in ("k", "v"):
                pair = live[name]
                if len(pair) != 2 or any(ct.batch_size != geo.n_q for ct in pair):
                    raise RuntimeError(f"Prefill layer {layer_idx} {name} has wrong GQA cache shape")
                info[f"{name}_limbs"] = [eng.limbs(ct) for ct in pair]
                saved[name] = tuple(cpu_cipher(ct) for ct in pair)
                info[f"{name}_host_bytes"] = sum(part.numel() * part.element_size()
                                                   for ct in saved[name] for part in ct.cv)
            if min(info["k_limbs"]) < eng.floor + 4 or min(info["v_limbs"]) < eng.floor + 2:
                raise RuntimeError(f"Prefill layer {layer_idx} KV levels insufficient: {info}")
            parked.append(saved)
            cache_meta.append(info)
            del live
        return original_clear(eng)

    setup.create = create_with_decode_keys
    DecoderLayer.forward = forward_capture
    Engine.clear_cache = clear_and_park
    futures.ThreadPoolExecutor = tracked_pool
    old_argv = sys.argv
    os.environ["SYLPH_LOGS"] = str(out)
    sys.argv = ["run_chain.py", str(first), str(last)] + (["--diagnose"] if diagnose else [])
    try:
        state = runpy.run_path(str(Path(os.environ.get("SYLPH_ROOT", str(Path(__file__).resolve().parents[1] / "providers/llama8b"))) /
                                   "sylph_easy/chain/run_chain.py"), run_name="__main__")
    finally:
        sys.argv = old_argv
        setup.create = original_create
        DecoderLayer.forward = original_forward
        Engine.clear_cache = original_clear
        futures.ThreadPoolExecutor = original_pool
        for pool in pools:
            pool.shutdown(wait=True, cancel_futures=True)
    if pending or len(parked) != last - first + 1 or forward_calls != last - first + 2:
        raise RuntimeError("the Prefill cache capture did not cover every requested layer")
    report, client, eng = state["report"], state["client"], state["b"]
    state.clear()  # Release the Prefill final hidden and last staged weights.
    gc.collect()
    torch.cuda.empty_cache()
    return report, client, eng, parked, cache_meta


def cache_preflight(client, eng, geo, saved, w0, cal0, context, *, atol):
    g = plain.decode_geometry(cal0, w0)
    kc, vc = plain.kv_cache(context.astype(np.float64), w0,
                            **{k: v for k, v in g.items() if k != "heads"})
    report = {}
    geom = HeadGeometry(geo.head_dim, geo.heads, geo.kv_heads)
    for name, reference in (("k", kc), ("v", vc)):
        expected = expanded_cache(reference, geom)
        errors = []
        for half, host in enumerate(saved[name]):
            dev = gpu_cipher(host)
            actual = np.stack([client.decrypt(eng.norm(c)).cpu().numpy()[:SLOTS].real
                               for c in eng.unstack(dev)])
            want = np.stack(expected[half])
            errors.append(float(np.max(np.abs(actual - want))))
            del dev
        report[name] = {"max_abs_by_half": errors, "limbs": [eng.limbs(c) for c in saved[name]]}
    report["passed"] = all(np.isfinite(e) and e <= atol for family in ("k", "v")
                           for e in report[family]["max_abs_by_half"])
    report["cache_atol"] = atol
    return report


def terminal_report(manifest, hidden, reference, w):
    from sylph_easy.tools import terminal_logits as tl
    terminal = manifest.get("terminal", {})
    if terminal.get("kind") == "quarot_folded_norm":
        lm_head = np.load(os.path.join(CHAIN, terminal["lm_head"]))
        eps = float(terminal["epsilon"])
        logits = lambda h: ((h / np.sqrt(np.mean(h ** 2) + eps)) @
                            lm_head.T.astype(np.float64)).reshape(-1)
        actual, want = logits(hidden), logits(reference)
        return dict(actual=tl.summarize_logits(actual), reference=tl.summarize_logits(want),
                    comparison=tl.compare_last_token_logits(actual, want))
    h = tl.inverse_sylph_residual(hidden[None], w["residual_signs"], w["residual_permutation"])
    ref = tl.inverse_sylph_residual(reference[None], w["residual_signs"], w["residual_permutation"])
    _, result = tl.evaluate_terminal_logits(manifest["source_checkpoint"], h,
                                            reference_hidden=ref, top_k=5)
    return dict(actual=result["actual"], reference=result["reference"],
                comparison=result["comparison"])


def decode_pass(client, eng, geo, manifest, entries, plain_inputs, saved, query, *, index, measured,
                query_token_id, expected_token, out):
    ct = client.encrypt(geo.encode_x(query), slots=SLOTS)
    x_ref = query.copy()
    lookahead = max(1, int(os.environ.get("SYLPH_LOOKAHEAD", "2")))
    host_pool = ThreadPoolExecutor(max_workers=lookahead)
    gpu_pool = ThreadPoolExecutor(max_workers=1)

    def prepare(entry):
        return load(entry, staged=True)

    def to_gpu(host):
        w, cal = host.result()
        layer = DecodeMVLayer(eng, w, cal, geo, DeviceProjector(eng, geo), 128, upload=True)
        return w, cal, layer

    host = {i: host_pool.submit(prepare, entries[i])
            for i in range(min(lookahead, len(entries)))}
    pending = gpu_pool.submit(to_gpu, host.pop(0))
    layers_out = []
    restore_total = 0.0
    t_chain = time.time()
    last_w = None
    try:
        for idx, entry in enumerate(entries):
            t_load = time.time()
            w, cal, layer = pending.result()
            load_seconds = time.time() - t_load
            if idx + lookahead < len(entries):
                host[idx + lookahead] = host_pool.submit(prepare, entries[idx + lookahead])
            if layer.position != 128 or layer.attention.public != 0:
                raise RuntimeError("P0 Decode position or public prefix changed")
            t_restore = time.time()
            cache = {name: tuple(gpu_cipher(c) for c in saved[idx][name]) for name in ("k", "v")}
            torch.cuda.synchronize()
            restore_seconds = time.time() - t_restore
            restore_total += restore_seconds
            layer.times.clear()
            torch.cuda.synchronize()
            t_forward = time.time()
            ct = layer.forward(ct, cache)
            torch.cuda.synchronize()
            seconds = time.time() - t_forward
            # Independent oracle only; the reference KV never enter the FHE Decode cache.
            opts = plain.decode_geometry(cal, w)
            kc, vc = plain.kv_cache(plain_inputs[f"layer{idx:02d}"].astype(np.float64), w,
                                    **{k: v for k, v in opts.items() if k != "heads"})
            x_ref, _ = plain.decode_layer(x_ref, (kc, vc), w, position=128, **opts)
            row = dict(layer=idx, seconds=round(seconds, 3), restore_seconds=round(restore_seconds, 3),
                       load_seconds=round(load_seconds, 3), limbs=eng.limbs(ct),
                       stages={k: round(v, 3) for k, v in layer.times.items()})
            layers_out.append(row)
            print(json.dumps({"decode_pass": index, "measured": measured, **row}), flush=True)
            last_w = w
            layer.P.release()
            del layer, cache, w, kc, vc
            unstage(entry)
            eng.clear_cache()
            gc.collect()
            if idx + 1 < len(entries):
                pending = gpu_pool.submit(to_gpu, host.pop(idx + 1))
    finally:
        host_pool.shutdown(wait=True)
        gpu_pool.shutdown(wait=True)
    chain_seconds = time.time() - t_chain
    hidden = geo.decode_x(client.decrypt(eng.norm(ct)).cpu().numpy()[:SLOTS].real)
    terminal = terminal_report(manifest, hidden, x_ref, last_w)
    comparison = terminal["comparison"]
    token = terminal["actual"]["argmax_token_id"]
    report = dict(mode="sc-p0-prefill-cipher-kv-decode-replay", pass_index=index, measured=measured,
                  prefix_tokens=0, position=128, cache_source="FHE Prefill emitted ciphertext",
                  kv_from_fhe_prefill=True, same_state_replay=index > 0,
                  query_token_id=int(query_token_id),
                  layer_seconds_total=round(sum(r["seconds"] for r in layers_out), 3),
                  restore_seconds_total=round(restore_total, 3), chain_seconds=round(chain_seconds, 2),
                  next_token_id=token, reference_next_token_id=terminal["reference"]["argmax_token_id"],
                  hidden_rel_error=float(np.max(np.abs(hidden - x_ref)) / np.max(np.abs(x_ref))),
                  terminal=terminal, layers=layers_out)
    if measured:
        (out / f"decode_cipher_kv_pass{index}.json").write_text(json.dumps(report, indent=1))
    eng.counts = {}
    eng.clear_cache()
    gc.collect()
    torch.cuda.synchronize()
    if ((expected_token is not None and token != expected_token)
            or not comparison["passed"] or not comparison["top_k_match"]):
        raise RuntimeError(f"Decode pass {index} failed token/top-5 correctness: {comparison}")
    print(json.dumps({"decode_pass_summary": {k: report[k] for k in
                      ("pass_index", "measured", "layer_seconds_total", "restore_seconds_total",
                       "chain_seconds", "next_token_id", "hidden_rel_error")}}), flush=True)
    return report


def main():
    args = args_from_cli()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest, entries = layers()
    if len(entries) != 32 or [e["layer"] for e in entries] != list(range(32)):
        raise ValueError("expected the 32-layer Llama-3-8B SC chain")
    if manifest.get("prefix_token_ids") != []:
        raise ValueError("the SC chain must declare no public-prefix tokens")
    plain_inputs = np.load(PLAIN_INPUTS)
    if any(f"layer{i:02d}" not in plain_inputs for i in range(32)):
        raise ValueError("the P0 chain lacks a complete plaintext validation trajectory")
    for entry in entries:
        unstage(entry)
    w0, cal0 = load(entries[0], staged=False)
    head_dim, heads, kv_heads = geometry(manifest)
    geo = MVGeometry(int(w0["q"].shape[1]), int(w0["gate"].shape[0]),
                     heads, kv_heads, head_dim)
    first, last = (0, 0) if args.preflight_only else (0, 31)
    prefill, client, eng, saved, cache_meta = capture_prefill_caches(geo, first=first, last=last, out=out)
    (out / "prefill_cipher_kv_meta.json").write_text(json.dumps(cache_meta, indent=1))
    preflight = cache_preflight(client, eng, geo, saved[0], w0, cal0, plain_inputs["layer00"],
                                atol=args.cache_atol)
    (out / "cache_layer0_preflight.json").write_text(json.dumps(preflight, indent=1))
    if not preflight["passed"]:
        raise RuntimeError(f"encrypted Prefill KV preflight failed: {preflight}")
    if args.preflight_only:
        print(json.dumps({"preflight_only": True, "cache": preflight,
                          "prefill_layer_seconds": prefill["layers"][0]["seconds"]}), flush=True)
        return
    comparison = prefill["terminal"]["comparison"]
    if (len(prefill["layers"]) != 32 or prefill.get("next_token_id") != args.query_token_id
            or not comparison["passed"] or not comparison["top_k_match"]):
        raise RuntimeError("FHE Prefill did not produce the verified query token")
    query, query_report = prepare_query(manifest, w0, plain_inputs["layer00"],
                                         token_id=prefill["next_token_id"],
                                         body_ids_path=args.body_ids, expected_prefix_ids=())
    del w0, cal0
    results = []
    for index in range(args.warmup_passes + args.measure_passes):
        measured = index >= args.warmup_passes
        result = decode_pass(client, eng, geo, manifest, entries, plain_inputs, saved, query,
                             index=index, measured=measured,
                             query_token_id=prefill["next_token_id"],
                             expected_token=args.expected_next_token_id, out=out)
        results.append(result)
    offload_total = sum(sum(m["cache_cpu_offload_seconds_in_prefill_forward"].values())
                          for m in cache_meta)
    summary = dict(model=manifest.get("model_name"), prefix_tokens=0,
                   prefill_next_token_id=prefill["next_token_id"],
                   prefill_terminal_passed=bool(comparison["passed"] and comparison["top_k_match"]),
                   prefill_layer_seconds_total=round(sum(r["seconds"] for r in prefill["layers"]), 3),
                   prefill_chain_seconds=prefill["chain_seconds"],
                   prefill_cache_cpu_offload_seconds_total=round(offload_total, 3),
                   prefill_layer_seconds_without_cpu_offload_estimate=round(
                       sum(r["seconds"] for r in prefill["layers"]) - offload_total, 3),
                   prefill_cache_layers=cache_meta,
                   kv_host_bytes=sum(m["k_host_bytes"] + m["v_host_bytes"] for m in cache_meta),
                   query=query_report, cache_source="FHE Prefill emitted ciphertext",
                   kv_from_fhe_prefill=True,
                   session_mode="same_state_replay_from_one_encrypted_prefill",
                   cache_layer0_preflight=preflight,
                   warmup_layer_seconds=[r["layer_seconds_total"] for r in results if not r["measured"]],
                   measured_layer_seconds=[r["layer_seconds_total"] for r in results if r["measured"]],
                   measured_restore_seconds=[r["restore_seconds_total"] for r in results if r["measured"]],
                   measured_next_token_ids=[r["next_token_id"] for r in results if r["measured"]])
    (out / "prefill_decode_cipher_kv_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({"summary": summary}), flush=True)


if __name__ == "__main__":
    main()
