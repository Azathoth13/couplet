#!/usr/bin/env python3
"""Experiment 1: BSGS weight Encode reuse in fused QKV, FFN gate/up and down.

The control re-encodes a chunk before every ciphertext MAC, preserving the
production chunk-major schedule and all homomorphic arithmetic. CUDA Event
Encode timing is collected in separate diagnostic runs from wall timing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("ODIN_ROOT", Path(__file__).resolve().parents[2])).resolve()
sys.path.insert(0, str(ROOT))
os.environ["T2_STAGE_SYNC"] = "1"
os.environ["LLAMA_FHE_LINEAR_PLAIN_RESIDENT"] = "0"
os.environ["ODIN_WEIGHT_TRACE"] = "0"

import easyfhe  # noqa: E402
from easyfhe import fhe  # noqa: E402
import easyfhe.fhe.constants as constants_mod  # noqa: E402
from llama3fhe.backend import create_sim_context, release_if_supported  # noqa: E402
from llama3fhe.config import SimulatorConfig  # noqa: E402
from llama3fhe.layouts.attention import AttentionPairLayout  # noqa: E402
from llama3fhe.operators.linear import (  # noqa: E402
    LinearOperator, LinearOperatorConfig, LinearWeight, linear_rotations,
)
from llama3fhe.operators.linear.prefill.qkv_kernel import (  # noqa: E402
    ComplexTokenDiagonalLayout, _projection_weights,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cases", default="qkv,gate_up,down")
    p.add_argument("--pages", default="0", help="FFN pages, e.g. 0,1,2,3")
    p.add_argument("--depth", type=int, default=37)
    p.add_argument("--limbs", type=int, default=13)
    p.add_argument("--rows", type=int, default=512)
    p.add_argument("--hoist", choices=("normal", "ext_normal"), default="normal")
    p.add_argument("--chunk-count", type=int, default=0, help="0=all eight chunks")
    p.add_argument("--pairs", type=int, choices=(1, 2), default=1)
    return p.parse_args()


def release_all(ciphers):
    for cipher in ciphers:
        release_if_supported(cipher)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class EncodeTrace:
    def __init__(self):
        self.events = {"stage1": [], "stage2": []}
        self.originals = {}

    def __enter__(self):
        for name, label in (("encode_stage1_packed", "stage1"),
                            ("encode_stage2", "stage2")):
            original = getattr(constants_mod, name)
            self.originals[name] = original

            def wrapper(*args, _original=original, _label=label, **kwargs):
                value = args[0] if _label == "stage1" else args[0].encoded_values
                rows = int(value.shape[0]) if len(value.shape) > 1 else 1
                start = easyfhe.cuda.Event(enable_timing=True)
                end = easyfhe.cuda.Event(enable_timing=True)
                start.record()
                try:
                    return _original(*args, **kwargs)
                finally:
                    end.record()
                    self.events[_label].append((start, end, rows))

            setattr(constants_mod, name, wrapper)
        return self

    def __exit__(self, *_):
        for name, original in self.originals.items():
            setattr(constants_mod, name, original)

    def summary(self):
        easyfhe.cuda.synchronize()
        return {
            f"{label}_{field}": result
            for label, events in self.events.items()
            for field, result in (
                ("calls", len(events)),
                ("rows", sum(rows for _, _, rows in events)),
                ("gpu_s", sum(start.elapsed_time(end) / 1000.0
                              for start, end, _ in events)),
            )
        }


def decrypt_complex(context, ciphers):
    result = []
    for cipher in ciphers:
        negative_i = fhe.homo_mul_i(cipher, context.context, negative=True)
        try:
            result.append(np.asarray(context.decrypt(cipher), dtype=np.float64)
                          + 1j * np.asarray(context.decrypt(negative_i), dtype=np.float64))
        finally:
            release_if_supported(negative_i)
    return np.stack(result)


def page_weight(case, page, data, hidden):
    if case == "qkv":
        _, _, weight = _projection_weights(
            hidden, data["q_proj"], data["k_proj"], data["v_proj"],
            attention_layout=AttentionPairLayout())
        return np.asarray(weight, dtype=np.complex128)
    if case == "gate_up":
        gate, up = data["gate_proj"], data["up_proj"]
        active = max(0, min(4096, gate.shape[0] - page * 4096))
        weight = np.zeros((4096, 4096), dtype=np.complex128)
        span = slice(page * 4096, page * 4096 + active)
        weight[:active] = 0.25 * (gate[span] + 1j * up[span])
        return weight
    down = data["down_proj"]
    active = max(0, min(4096, down.shape[1] - page * 4096))
    weight = np.zeros((4096, 4096), dtype=np.float64)
    weight[:, :active] = 0.5 * down[:, page * 4096 : page * 4096 + active]
    return weight


def benchmark(case, page, data, hidden, packed_hidden, packed_down,
              context, config, chunks, options):
    weight_array = page_weight(case, page, data, hidden)
    packed = packed_down if case == "down" else packed_hidden
    inputs = tuple(context.encrypt(row, level=int(context.context.L) - options.limbs,
                                   slots=32768) for row in packed)
    conjugates = () if case == "down" else tuple(
        fhe.homo_rotate(cipher, int(context.context.M) - 1, context.context)
        for cipher in inputs)
    evaluated = inputs + conjugates
    count = len(evaluated)
    assert count == (8 if case == "down" else 16)
    operator = LinearOperator(crypto_context=context, token_lanes=8, config=config)
    weight = LinearWeight(weight_array, dtype=weight_array.dtype, crypto_context=context)
    original_mac = operator._mac_chunk
    original_plaintext = weight.plaintext
    calls = Counter()
    selected = tuple(chunk.index for chunk in chunks)
    row_count = sum(chunk.giant_count * 64 for chunk in chunks)
    active_level = int(context.context.L) - options.limbs
    try:
        weight.prepare(config)
        easyfhe.cuda.synchronize()

        def counted_plaintext(*args, **kwargs):
            calls["batches"] += 1
            calls["rows"] += int(args[1].giant_count) * 64
            return original_plaintext(*args, **kwargs)

        weight.plaintext = counted_plaintext

        def no_reuse_mac(chunk, plaintexts, input_ciphers, accumulators,
                         cached_babies, *, plan, strategy, baby_anchor_step):
            # The first plaintext comes from apply; refresh it for each later CT.
            for index, cipher in enumerate(input_ciphers):
                current = plaintexts if index == 0 else weight.plaintext(
                    plan, chunk, level=active_level, operator_config=config)
                one_acc = [accumulators[index]]
                original_mac(chunk, current, (cipher,), one_acc,
                             [cached_babies[index]], plan=plan, strategy=strategy,
                             baby_anchor_step=baby_anchor_step)
                accumulators[index] = one_acc[0]

        def run(mode, diagnostic=False, decrypt=False):
            calls.clear()
            operator.clear_baby_cache()
            operator._mac_chunk = original_mac if mode == "reuse" else no_reuse_mac
            easyfhe.cuda.synchronize()
            started = time.perf_counter()
            trace = EncodeTrace() if diagnostic else None
            if trace is None:
                outputs, profile = operator.apply(evaluated, weight,
                                                  chunk_indices=selected)
            else:
                with trace:
                    outputs, profile = operator.apply(evaluated, weight,
                                                      chunk_indices=selected)
            easyfhe.cuda.synchronize()
            row = {"mode": mode, "diagnostic": diagnostic,
                   "wall_s": time.perf_counter() - started,
                   "encode_batch_calls": calls["batches"],
                   "encode_rows": calls["rows"],
                   "first_chunk_pack_encode_s": float(profile.get("pack_encode", 0.0)),
                   "baby_precompute_s": float(profile.get("baby_precompute", 0.0)),
                   "rescale_s": float(profile.get("rescale", 0.0)),
                   "output_limbs": [int(x.state.cur_limbs) for x in outputs]}
            expected = len(chunks) * (1 if mode == "reuse" else count)
            if calls["batches"] != expected or calls["rows"] != row_count * (1 if mode == "reuse" else count):
                release_all(outputs)
                raise RuntimeError(f"unexpected encode count: {row}")
            if trace is not None:
                row.update(trace.summary())
                for label in ("stage1", "stage2"):
                    if row[f"{label}_calls"] != expected or row[f"{label}_rows"] != calls["rows"]:
                        release_all(outputs)
                        raise RuntimeError(f"unexpected {label} count: {row}")
            print(json.dumps({"case": case, "page": page, **row}), flush=True)
            decoded = decrypt_complex(context, outputs) if decrypt else None
            release_all(outputs)
            return row, decoded

        for mode in ("reuse", "no_reuse"):
            run(mode)
        order = (("reuse", "no_reuse", "no_reuse", "reuse") if options.pairs == 2
                 else ("reuse", "no_reuse"))
        samples = [run(mode)[0] for mode in order]
        diagnostic = {}
        decoded = {}
        for mode in ("reuse", "no_reuse"):
            diagnostic[mode], decoded[mode] = run(mode, diagnostic=True, decrypt=True)
        diff = np.abs(decoded["reuse"] - decoded["no_reuse"])
        max_abs = float(diff.max())
        relative = max_abs / max(1e-12, float(np.abs(decoded["reuse"]).max()))
        if relative > 1e-6:
            raise RuntimeError(f"A/B output disagreement: {case} page {page}: {relative}")
        median = {mode: statistics.median(row["wall_s"] for row in samples
                                          if row["mode"] == mode)
                  for mode in ("reuse", "no_reuse")}
        result = {"case": case, "page": page, "selected_chunks": list(selected),
                  "evaluated_ciphertexts": count, "diagonals": row_count,
                  "logical_ct_pt_entries": row_count * count,
                  "reuse_encode_rows": row_count,
                  "no_reuse_encode_rows": row_count * count,
                  "saved_encode_rows": row_count * (count - 1),
                  "reuse_batch_calls": len(chunks),
                  "no_reuse_batch_calls": len(chunks) * count,
                  "wall_median_s": median,
                  "saved_wall_s": median["no_reuse"] - median["reuse"],
                  "diagnostic": diagnostic, "samples": samples,
                  "verification": {"max_abs": max_abs, "relative": relative}}
        for label in ("stage1", "stage2"):
            result[f"saved_{label}_gpu_s"] = (
                diagnostic["no_reuse"][f"{label}_gpu_s"]
                - diagnostic["reuse"][f"{label}_gpu_s"])
        return result
    finally:
        operator._mac_chunk = original_mac
        operator.close()
        weight.close()
        release_all(inputs)
        release_all(conjugates)


def main():
    a = parse_args()
    cases = tuple(item.strip() for item in a.cases.split(",") if item.strip())
    pages = tuple(int(item) for item in a.pages.split(","))
    if any(case not in {"qkv", "gate_up", "down"} for case in cases):
        raise ValueError("invalid --cases")
    if any(page not in range(4) for page in pages) or not pages:
        raise ValueError("invalid --pages")
    config = LinearOperatorConfig(
        baby_steps=64, baby_anchor_step=64, max_plaintext_rows=a.rows,
        reuse_baby_rotations=True, hoist_strategy=a.hoist,
        constant_cache_mode="none", raw_weight_policy="device", encode_depth=0)
    layout = ComplexTokenDiagonalLayout(seq_len=128, dimension=4096, slots=32768)
    plan = config.plan(dimension=4096, token_lanes=8)
    chunks = plan.chunks if a.chunk_count == 0 else plan.chunks[:a.chunk_count]
    if not chunks or a.chunk_count < 0 or a.chunk_count > len(plan.chunks):
        raise ValueError("invalid --chunk-count")
    rotations = linear_rotations(dimension=4096, token_lanes=8, slots=32768,
                                 include_conjugation=True, operator_config=config)
    setup_start = time.perf_counter()
    context = create_sim_context(
        SimulatorConfig(logN=16, maxLevelsRemaining=a.depth, dnum=5,
                        dcrtBits=59, firstMod=60, device="cuda"),
        input_dim=4096, block_out=8, rotations=rotations,
        include_default_rotations=False)
    print(json.dumps({"phase": "context_ready", "L": int(context.context.L),
                      "setup_s": time.perf_counter() - setup_start,
                      "rotation_keys": len(rotations)}), flush=True)
    with np.load(a.fixture, allow_pickle=False) as data:
        hidden = np.asarray(data["hidden_in"], dtype=np.float64)
        if hidden.shape != (128, 4096):
            raise ValueError(f"invalid hidden_in shape {hidden.shape}")
        packed_hidden = layout.pack(hidden)
        # Down operand values are fixed; they do not affect Encode scheduling.
        packed_down = layout.pack(np.random.default_rng(20261110).normal(
            0.0, 0.1, (128, 4096)))
        reports = []
        for case in cases:
            for page in ((0,) if case == "qkv" else pages):
                print(json.dumps({"phase": "start", "case": case, "page": page}), flush=True)
                report = benchmark(case, page, data, hidden, packed_hidden,
                                   packed_down, context, config, chunks, a)
                reports.append(report)
                print(json.dumps({"phase": "complete", "case": case,
                                  "page": page,
                                  "saved_encode_rows": report["saved_encode_rows"],
                                  "saved_wall_s": report["saved_wall_s"]}), flush=True)
    output = {"experiment": "BSGS weight Encode reuse across token ciphertexts",
              "source": str(ROOT), "script_sha256": sha256(Path(__file__)),
              "fixture": str(a.fixture), "fixture_sha256": sha256(a.fixture),
              "settings": {"L": int(context.context.L), "depth": a.depth,
                           "limbs": a.limbs, "rows": a.rows, "hoist": a.hoist,
                           "chunks": len(chunks), "pairs": a.pairs},
              "reports": reports}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"result": str(a.out), "reports": len(reports)}), flush=True)


if __name__ == "__main__":
    main()
