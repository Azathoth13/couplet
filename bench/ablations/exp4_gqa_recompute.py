#!/usr/bin/env python3
"""Experiment 4: GQA K/V preparation sharing versus actual repeated work.

Both arms have the same 32 logical Q heads and 8 distinct logical K/V heads.
The repeat arm physically expands K/V to 32 heads (four identical copies of
each logical head), encrypts each pair as a distinct ciphertext, and runs the
unchanged QK or PV operator with GQA ratio one.  The shared arm uses the
production 32/8 layout and GQA ratio four.  Inputs, keys, and outputs are
prepared outside the timed operator region; input and operator memory are
reported separately. This is an operator ablation, not a full model run.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("ODIN_ROOT", Path(__file__).resolve().parents[2])).resolve()
sys.path.insert(0, str(ROOT))

import easyfhe  # noqa: E402
from llama3fhe.backend import create_sim_context, release_if_supported  # noqa: E402
from llama3fhe.config import SimulatorConfig  # noqa: E402
from llama3fhe.layouts.attention import AttentionPairLayout  # noqa: E402
from llama3fhe.operators.attention.audit import (  # noqa: E402
    delta_qk_rotations, delta_pv_rotations,
    delta_qk_operation_counts, delta_pv_operation_counts,
)
from llama3fhe.operators.attention.config import AttentionOperatorConfig  # noqa: E402
from llama3fhe.operators.attention.prefill.qk import qk_delta_fhe  # noqa: E402
from llama3fhe.operators.attention.prefill.pv import pv_delta_fhe  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--operation", choices=("qk", "pv", "both"), default="both")
    parser.add_argument("--depth", type=int, default=37)
    parser.add_argument("--limbs", type=int, default=13)
    parser.add_argument("--shifts", type=int, default=64,
                        help="1..64; use 8 for a smoke, 64 for full operator")
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--max-relative-error", type=float, default=0.02)
    return parser.parse_args()


def sync():
    easyfhe.cuda.synchronize()


def release_many(ciphers):
    for cipher in tuple(ciphers):
        release_if_supported(cipher)


def layouts():
    shared = AttentionPairLayout(seq_len=128, head_dim=128,
                                 query_heads=32, key_value_heads=8,
                                 slots=32768)
    repeated = AttentionPairLayout(seq_len=128, head_dim=128,
                                   query_heads=32, key_value_heads=32,
                                   slots=32768)
    return {"shared": shared, "repeated": repeated}


def static_counts(operation, layout):
    if operation == "qk":
        return delta_qk_operation_counts(
            layout, complex_feature_folding=True,
            key_features_pre_folded=True,
            query_features_pre_scaled=True,
            query_features_pre_folded=True)
    return delta_pv_operation_counts(
        layout, complex_delta_folding=True,
        probability_pre_folded=False,
        value_pre_folded=False)


def logical_data(seed):
    rng = np.random.default_rng(seed)
    query = rng.normal(0.0, 0.02, size=(128, 32, 128)).astype(np.float32)
    key = rng.normal(0.0, 0.02, size=(128, 8, 128)).astype(np.float32)
    value = rng.normal(0.0, 0.02, size=(128, 8, 128)).astype(np.float32)
    probability = rng.uniform(0.1, 1.0, size=(32, 128, 128)).astype(np.float32)
    probability /= probability.sum(axis=2, keepdims=True)
    return query, key, value, probability


def packed_inputs(operation, mode, layout, data):
    query, key, value, probability = data
    if mode == "repeated":
        # Distinct K/V payloads are encrypted below. This is a real work and
        # memory control, not 16 Python references to four ciphertexts.
        key = np.repeat(key, 4, axis=1)
        value = np.repeat(value, 4, axis=1)
    if operation == "qk":
        return (layout.pack_query_c_feature_folded(query, scale=0.5),
                layout.pack_key_delta_feature_folded(key))
    return layout.pack_score_delta(probability), layout.pack_value_delta(value)


def operator_config():
    return AttentionOperatorConfig(
        rotation_mode="fast", rotation_chunk_size=8,
        qk_output_level=None,
        qk_complex_feature_folding=True,
        qk_key_features_pre_folded=True,
        qk_query_features_pre_scaled=True,
        qk_query_features_pre_folded=True,
        pv_complex_delta_folding=True,
    )


def evaluate(operation, left, right, *, layout, context, config, shifts):
    kwargs = dict(layout=layout, crypto_context=context,
                  operator_config=config, shift_range=(0, shifts))
    if operation == "qk":
        return qk_delta_fhe(left, right, **kwargs)
    return pv_delta_fhe(left, right, **kwargs)


def unpack(operation, outputs, layout, context):
    plain = np.stack([np.asarray(context.decrypt(cipher), dtype=np.float64)
                      for cipher in outputs])
    return (layout.unpack_score_delta(plain) if operation == "qk"
            else layout.unpack_output_c(plain))


def measure_mode(operation, mode, layout, data, context, config, options):
    left_pack, right_pack = packed_inputs(operation, mode, layout, data)
    if left_pack.shape[0] != 16 or right_pack.shape[0] != (4 if mode == "shared" else 16):
        raise RuntimeError("unexpected GQA ciphertext packing")
    sync()
    before_input = int(easyfhe.cuda.memory_allocated())
    level = int(context.context.L) - options.limbs
    left = tuple(context.encrypt(row, level=level, slots=32768) for row in left_pack)
    right = tuple(context.encrypt(row, level=level, slots=32768) for row in right_pack)
    if len({id(cipher) for cipher in right}) != len(right):
        raise RuntimeError("repeated K/V ciphertexts are aliases")
    sync()
    after_input = int(easyfhe.cuda.memory_allocated())
    rows = []
    logical_output = None
    try:
        # Warm one call after the inputs exist: CUDA kernels and mask bundle
        # allocation should not be charged to one mode's timed samples.
        warm = evaluate(operation, left, right, layout=layout, context=context,
                        config=config, shifts=options.shifts)
        sync()
        release_many(warm)
        sync()
        after_warm = int(easyfhe.cuda.memory_allocated())
        for _ in range(options.reps):
            easyfhe.cuda.reset_peak_memory_stats()
            sync()
            started = time.perf_counter()
            outputs = evaluate(operation, left, right, layout=layout,
                               context=context, config=config,
                               shifts=options.shifts)
            sync()
            wall = time.perf_counter() - started
            peak = int(easyfhe.cuda.max_memory_allocated())
            after_output = int(easyfhe.cuda.memory_allocated())
            try:
                if logical_output is None:
                    logical_output = unpack(operation, outputs, layout, context)
                output_limbs = [int(cipher.state.cur_limbs) for cipher in outputs]
            finally:
                release_many(outputs)
            rows.append({
                "wall_s": wall,
                "peak_allocated_bytes": peak,
                "peak_over_pre_input_bytes": peak - before_input,
                "peak_over_post_input_bytes": peak - after_input,
                "peak_over_warm_input_bytes": peak - after_warm,
                "post_output_allocated_bytes": after_output,
                "output_limbs": output_limbs,
            })
        return {
            "mode": mode,
            "gqa_ratio": layout.gqa_ratio,
            "query_ciphertexts": len(left),
            "kv_ciphertexts": len(right),
            "distinct_kv_ciphertext_handles": len({id(cipher) for cipher in right}),
            "allocated_before_inputs_bytes": before_input,
            "allocated_after_inputs_bytes": after_input,
            "allocated_after_warm_bytes": after_warm,
            "input_increment_bytes": after_input - before_input,
            "median_wall_s": statistics.median(row["wall_s"] for row in rows),
            "median_peak_allocated_bytes": statistics.median(
                row["peak_allocated_bytes"] for row in rows),
            "median_peak_over_pre_input_bytes": statistics.median(
                row["peak_over_pre_input_bytes"] for row in rows),
            "median_peak_over_post_input_bytes": statistics.median(
                row["peak_over_post_input_bytes"] for row in rows),
            "median_peak_over_warm_input_bytes": statistics.median(
                row["peak_over_warm_input_bytes"] for row in rows),
            "selected_counts": {
                "ct_ct_multiplications": int(layout.query_pair_count) * options.shifts,
                "kv_nonzero_shift_preparations": (
                    int(layout.key_value_pair_count) * (options.shifts - 1)),
                "kv_initial_half_folds": (
                    int(layout.key_value_pair_count) if operation == "pv" else 0),
            },
            "full_64_shift_static_counts": static_counts(operation, layout),
            "samples": rows,
        }, logical_output
    finally:
        release_many(left)
        release_many(right)
        sync()
        easyfhe.cuda.empty_cache()


def main():
    options = parse_args()
    if not 1 <= options.shifts <= 64 or options.reps < 1 or options.depth <= options.limbs:
        raise ValueError("need 1<=shifts<=64, reps>=1, depth>limbs")
    by_mode = layouts()
    selected = ("qk", "pv") if options.operation == "both" else (options.operation,)
    rotations = set()
    for layout in by_mode.values():
        if "qk" in selected:
            rotations.update(delta_qk_rotations(
                layout, complex_feature_folding=True,
                key_features_pre_folded=True,
                query_features_pre_folded=True, log_n=16))
        if "pv" in selected:
            rotations.update(delta_pv_rotations(
                layout, complex_delta_folding=True,
                probability_pre_folded=False,
                value_pre_folded=False, log_n=16))
    started = time.perf_counter()
    context = create_sim_context(
        SimulatorConfig(logN=16, maxLevelsRemaining=options.depth,
                        dnum=5, dcrtBits=59, firstMod=60, device="cuda"),
        input_dim=4096, block_out=8,
        rotations=tuple(sorted(rotations)), include_default_rotations=False)
    sync()
    print(json.dumps({"phase": "context_ready",
                      "setup_s": time.perf_counter() - started,
                      "rotation_keys": len(rotations),
                      "context_allocated_bytes": int(easyfhe.cuda.memory_allocated())}),
          flush=True)
    data = logical_data(options.seed)
    config = operator_config()
    results = []
    for operation in selected:
        measured = {}
        logical = {}
        for mode, layout in by_mode.items():
            print(json.dumps({"phase": "start", "operation": operation,
                              "mode": mode}), flush=True)
            measured[mode], logical[mode] = measure_mode(
                operation, mode, layout, data, context, config, options)
            print(json.dumps({"phase": "complete", "operation": operation,
                              "mode": mode,
                              "wall_s": measured[mode]["median_wall_s"],
                              "input_increment_bytes": measured[mode]["input_increment_bytes"],
                              "peak_bytes": measured[mode]["median_peak_allocated_bytes"]}),
                  flush=True)
        difference = np.abs(logical["shared"] - logical["repeated"])
        max_abs = float(difference.max())
        relative = max_abs / max(1e-12, float(np.abs(logical["shared"]).max()))
        if relative > options.max_relative_error:
            raise RuntimeError(f"{operation} logical output mismatch: {relative}")
        results.append({
            "operation": operation,
            "modes": measured,
            "verification": {"max_abs": max_abs, "relative": relative},
            "extra_wall_s": (measured["repeated"]["median_wall_s"]
                             - measured["shared"]["median_wall_s"]),
            "extra_input_allocated_bytes": (
                measured["repeated"]["input_increment_bytes"]
                - measured["shared"]["input_increment_bytes"]),
            "extra_peak_allocated_bytes": (
                measured["repeated"]["median_peak_allocated_bytes"]
                - measured["shared"]["median_peak_allocated_bytes"]),
            "extra_peak_over_pre_input_bytes": (
                measured["repeated"]["median_peak_over_pre_input_bytes"]
                - measured["shared"]["median_peak_over_pre_input_bytes"]),
            "extra_peak_over_warm_input_bytes": (
                measured["repeated"]["median_peak_over_warm_input_bytes"]
                - measured["shared"]["median_peak_over_warm_input_bytes"]),
        })
    report = {
        "experiment": "GQA K/V sharing vs distinct per-query-pair recomputation",
        "settings": {"depth": options.depth, "L": int(context.context.L),
                     "limbs": options.limbs, "shifts": options.shifts,
                     "reps": options.reps, "seed": options.seed,
                     "rotation_keys": len(rotations)},
        "results": results,
    }
    options.out.parent.mkdir(parents=True, exist_ok=True)
    options.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"result": str(options.out), "operations": len(results)}),
          flush=True)


if __name__ == "__main__":
    main()
