#!/usr/bin/env python3
"""Experiment 5: QK feature pairing and PV key pairing, each against no fold.

QK has a same-ciphertext A/B (runtime folding vs canonical) and a separate
production-style pre-folded-input arm.  PV uses exactly the same encrypted
post-Softmax probability and V inputs in both arms.  One Q/KV head pair keeps
this an operator-level experiment without changing the GQA model.  The raw
output limb count is reported: runtime PV folding spends an extra level.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("ODIN_ROOT", Path(__file__).resolve().parents[2])).resolve()
sys.path.insert(0, str(ROOT))
os.environ["T2_STAGE_SYNC"] = "1"

import easyfhe  # noqa: E402
from easyfhe import fhe  # noqa: E402
from llama3fhe.backend import create_sim_context, release_if_supported  # noqa: E402
from llama3fhe.config import SimulatorConfig  # noqa: E402
from llama3fhe.layouts.attention import AttentionPairLayout  # noqa: E402
from llama3fhe.operators.attention.audit import (  # noqa: E402
    delta_qk_operation_counts, delta_qk_rotations,
    delta_pv_operation_counts, delta_pv_rotations,
)
from llama3fhe.operators.attention.config import AttentionOperatorConfig  # noqa: E402
from llama3fhe.operators.attention.prefill.qk import qk_delta_fhe  # noqa: E402
from llama3fhe.operators.attention.prefill.pv import pv_delta_fhe  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--operation", choices=("qk", "pv", "both"), default="both")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--log-n", type=int, default=16)
    p.add_argument("--depth", type=int, default=16)
    p.add_argument("--limbs", type=int, default=9)
    p.add_argument("--pairs", type=int, choices=(1, 2), default=2)
    p.add_argument("--seed", type=int, default=20261110)
    return p.parse_args()


def sha256(path):
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            result.update(block)
    return result.hexdigest()


def release_all(ciphers):
    for cipher in ciphers:
        release_if_supported(cipher)


def encrypt_many(context, values, level, slots):
    return tuple(context.encrypt(row, level=level, slots=slots) for row in values)


def sync():
    easyfhe.cuda.synchronize()


def main():
    a = parse_args()
    if a.seq_len % a.head_dim or a.seq_len % 2 or a.head_dim % 2:
        raise ValueError("need even head_dim dividing even seq_len")
    if a.depth <= a.limbs or a.limbs < 5:
        raise ValueError("need 5<=limbs<depth")
    slots = 1 << (a.log_n - 1)
    layout = AttentionPairLayout(
        seq_len=a.seq_len, head_dim=a.head_dim,
        query_heads=2, key_value_heads=2, slots=slots)
    base = AttentionOperatorConfig(
        rotation_mode="fast", rotation_chunk_size=8, qk_output_level=None,
        qk_lazy_relinearization=True, pv_lazy_relinearization=True,
        qk_complex_feature_folding=False, qk_key_features_pre_folded=False,
        qk_query_features_pre_scaled=False, qk_query_features_pre_folded=False,
        pv_complex_delta_folding=False)
    qk_modes = {
        "canonical": base,
        "fold_runtime": replace(base, qk_complex_feature_folding=True),
        "fold_prepacked": replace(
            base, qk_complex_feature_folding=True,
            qk_key_features_pre_folded=True,
            qk_query_features_pre_scaled=True,
            qk_query_features_pre_folded=True),
    }
    pv_modes = {
        "canonical": base,
        "fold_runtime": replace(base, pv_complex_delta_folding=True),
    }
    rotation_sets = []
    if a.operation in ("qk", "both"):
        rotation_sets.extend((
            delta_qk_rotations(layout, complex_feature_folding=False, log_n=a.log_n),
            delta_qk_rotations(layout, complex_feature_folding=True,
                               key_features_pre_folded=False,
                               query_features_pre_folded=False, log_n=a.log_n),
            delta_qk_rotations(layout, complex_feature_folding=True,
                               key_features_pre_folded=True,
                               query_features_pre_folded=True, log_n=a.log_n),
        ))
    if a.operation in ("pv", "both"):
        rotation_sets.extend((
            delta_pv_rotations(layout, complex_delta_folding=False, log_n=a.log_n),
            delta_pv_rotations(layout, complex_delta_folding=True, log_n=a.log_n),
        ))
    rotations = tuple(sorted(set().union(*map(set, rotation_sets))))
    started = time.perf_counter()
    context = create_sim_context(
        SimulatorConfig(logN=a.log_n, maxLevelsRemaining=a.depth,
                        dnum=5, dcrtBits=59, firstMod=60, device="cuda"),
        input_dim=a.head_dim, block_out=a.seq_len, rotations=rotations,
        include_default_rotations=False)
    level = int(context.context.L) - a.limbs
    print(json.dumps({"phase": "context_ready", "L": int(context.context.L),
                      "rotation_keys": len(rotations),
                      "setup_s": time.perf_counter() - started}), flush=True)

    rng = np.random.default_rng(a.seed)
    shape = (a.seq_len, 2, a.head_dim)
    query = rng.normal(0.0, 0.05, shape).astype(np.float32)
    key = rng.normal(0.0, 0.05, shape).astype(np.float32)
    value = rng.normal(0.0, 0.1, shape).astype(np.float32)
    probability = rng.uniform(0.1, 1.0, (2, a.seq_len, a.seq_len)).astype(np.float32)
    probability /= probability.sum(axis=-1, keepdims=True)
    expected_qk = np.einsum("thd,shd->hts", query, key, optimize=True)
    expected_pv = np.einsum("hts,shd->thd", probability, value, optimize=True)
    outputs = {}

    def run_modes(operation, modes, inputs_by_mode, kernel, unpack, expected):
        samples = []
        decoded = {}
        observed_counts = {}
        # The audit includes preparation masks/rotations.  Count actual Ct-Ct
        # calls in a separate untimed diagnostic run to catch path drift.
        for mode, cfg in modes.items():
            lhs, rhs = inputs_by_mode[mode]
            if operation == "qk":
                audit = delta_qk_operation_counts(
                    layout,
                    complex_feature_folding=cfg.qk_complex_feature_folding,
                    key_features_pre_folded=cfg.qk_key_features_pre_folded,
                    query_features_pre_scaled=cfg.qk_query_features_pre_scaled,
                    query_features_pre_folded=cfg.qk_query_features_pre_folded)
            else:
                audit = delta_pv_operation_counts(
                    layout, complex_delta_folding=cfg.pv_complex_delta_folding)
            actual_ct_ct = 0
            original_mul = fhe.homo_mul_no_relin

            def counted_mul(*args, **kwargs):
                nonlocal actual_ct_ct
                actual_ct_ct += 1
                return original_mul(*args, **kwargs)

            sync()
            fhe.homo_mul_no_relin = counted_mul
            prepared = ()
            try:
                prepared = kernel(lhs, rhs, layout=layout, crypto_context=context,
                                  operator_config=cfg)
                sync()
            finally:
                fhe.homo_mul_no_relin = original_mul
                release_all(prepared)
            if actual_ct_ct != audit["ct_ct_multiplications"]:
                raise RuntimeError(
                    f"{operation}/{mode}: actual {actual_ct_ct} Ct-Ct calls "
                    f"but static audit says {audit['ct_ct_multiplications']}")
            audit["actual_ct_ct_calls"] = actual_ct_ct
            observed_counts[mode] = audit
        order = (list(modes) + list(reversed(modes))) * a.pairs
        for mode in order:
            lhs, rhs = inputs_by_mode[mode]
            cfg = modes[mode]
            sync()
            baseline = int(easyfhe.cuda.memory_allocated())
            easyfhe.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            produced = kernel(lhs, rhs, layout=layout, crypto_context=context,
                              operator_config=cfg)
            sync()
            wall = time.perf_counter() - started
            peak = int(easyfhe.cuda.max_memory_allocated())
            row = {"mode": mode, "wall_s": wall, "peak_allocated_bytes": peak,
                   "baseline_allocated_bytes": baseline,
                   "peak_extra_bytes": max(0, peak - baseline),
                   "output_limbs": [int(cipher.state.cur_limbs) for cipher in produced]}
            samples.append(row)
            print(json.dumps({"operation": operation, **row}), flush=True)
            if mode not in decoded:
                plaintext = np.stack([np.asarray(context.decrypt(cipher)).real
                                      for cipher in produced])
                decoded[mode] = unpack(plaintext)
            release_all(produced)

        check = {}
        reference = decoded["canonical"]
        limb_sets = {mode: tuple(next(item["output_limbs"] for item in samples
                                       if item["mode"] == mode))
                     for mode in modes}
        if any(len(set(limbs)) != 1 for limbs in limb_sets.values()):
            raise RuntimeError(f"{operation} has mixed output limbs within a mode: {limb_sets}")
        if any(tuple(item["output_limbs"]) != limb_sets[item["mode"]]
               for item in samples):
            raise RuntimeError(f"{operation} output limbs varied across repeats")
        canonical_limbs = limb_sets["canonical"][0]
        depth_key = f"multiplicative_depth_from_{operation}_inputs"
        canonical_depth = observed_counts["canonical"][depth_key]
        for mode, limbs in limb_sets.items():
            actual_extra = canonical_limbs - limbs[0]
            expected_extra = observed_counts[mode][depth_key] - canonical_depth
            if actual_extra != expected_extra:
                raise RuntimeError(
                    f"{operation}/{mode} output depth differs from audit: "
                    f"actual {actual_extra}, expected {expected_extra}")
        for mode, result in decoded.items():
            abs_error = float(np.max(np.abs(result - expected)))
            rel_error = abs_error / max(1e-12, float(np.max(np.abs(expected))))
            delta = float(np.max(np.abs(result - reference)))
            check[mode] = {"max_abs_vs_oracle": abs_error,
                           "relative_vs_oracle": rel_error,
                           "max_abs_vs_canonical": delta}
            if rel_error > 0.1 or delta > 0.01:
                raise RuntimeError(f"{operation}/{mode} output mismatch: {check[mode]}")
        return {
            "operation": operation,
            "comparison_contract": "raw operator outputs; no output modulus alignment",
            "modes": {mode: {"audit_counts": observed_counts[mode],
                             "output_limbs": list(limb_sets[mode]),
                             "extra_levels_spent_vs_canonical": (
                                 canonical_limbs - limb_sets[mode][0]),
                             "wall_median_s": statistics.median(
                                 item["wall_s"] for item in samples if item["mode"] == mode),
                             "peak_allocated_bytes_median": int(statistics.median(
                                 item["peak_allocated_bytes"] for item in samples
                                 if item["mode"] == mode)),
                             "peak_extra_bytes_median": int(statistics.median(
                                 item["peak_extra_bytes"] for item in samples
                                 if item["mode"] == mode)),
                             "verification": check[mode]}
                      for mode in modes},
            "samples": samples,
        }

    if a.operation in ("qk", "both"):
        q_can = encrypt_many(context, layout.pack_query_c(query), level, slots)
        k_can = encrypt_many(context, layout.pack_key_delta(key), level, slots)
        q_fold = encrypt_many(context, layout.pack_query_c_feature_folded(query),
                              level, slots)
        k_fold = encrypt_many(context, layout.pack_key_delta_feature_folded(key),
                              level, slots)
        try:
            qk_inputs = {"canonical": (q_can, k_can),
                         "fold_runtime": (q_can, k_can),
                         "fold_prepacked": (q_fold, k_fold)}
            outputs["qk"] = run_modes(
                "qk", qk_modes, qk_inputs, qk_delta_fhe,
                layout.unpack_score_delta, expected_qk)
        finally:
            release_all(q_can + k_can + q_fold + k_fold)
    if a.operation in ("pv", "both"):
        p_can = encrypt_many(context, layout.pack_score_delta(probability),
                             level, slots)
        v_can = encrypt_many(context, layout.pack_value_delta(value), level, slots)
        try:
            pv_inputs = {"canonical": (p_can, v_can),
                         "fold_runtime": (p_can, v_can)}
            outputs["pv"] = run_modes(
                "pv", pv_modes, pv_inputs, pv_delta_fhe,
                layout.unpack_output_c, expected_pv.reshape(a.seq_len, -1))
        finally:
            release_all(p_can + v_can)
    result = {
        "experiment": "operation-specific complex pairing",
        "source": str(ROOT), "script_sha256": sha256(Path(__file__)),
        "config": {"seq_len": a.seq_len, "head_dim": a.head_dim,
                   "query_heads": 2, "key_value_heads": 2,
                   "log_n": a.log_n, "depth": a.depth,
                   "L": int(context.context.L), "limbs": a.limbs,
                   "pairs": a.pairs, "rotation_keys": len(rotations),
                   "seed": a.seed},
        "results": outputs,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"result": str(a.out), "operations": list(outputs)}), flush=True)


if __name__ == "__main__":
    main()
