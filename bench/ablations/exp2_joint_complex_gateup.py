#!/usr/bin/env python3
"""Experiment 2: joint complex Gate/Up plaintext vs two real plaintexts.

Joint: P=(G+iU)/4; project P on z and conj(z), then recover
    gate_half = Pz + conj(Pconj(z)) = Gz/2
    up_half   = -i(Pz - conj(Pconj(z))) = Uz/2.
Separate: project G/2 and U/2 on z directly. Both paths have 16 input
ciphertext evaluations per feature page and the same final 16 ciphertexts.
Only the number of encoded weight rows doubles (4096 -> 8192). The separate
path saves input and recovery conjugations and can reuse z's baby rotations.
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
from llama3fhe.operators.linear import (  # noqa: E402
    LinearOperator, LinearOperatorConfig, LinearWeight, linear_rotations,
)
from llama3fhe.operators.linear.prefill.qkv_kernel import (  # noqa: E402
    ComplexTokenDiagonalLayout,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--pages", default="0,1,2,3")
    p.add_argument("--depth", type=int, default=37)
    p.add_argument("--limbs", type=int, default=13)
    p.add_argument("--rows", type=int, default=2048)
    p.add_argument("--chunk-count", type=int, default=0)
    p.add_argument("--pairs", type=int, choices=(1, 2), default=2)
    p.add_argument("--gate-alpha", type=float, default=1.0,
                   help="same public Gate scale in both arms; 1 isolates complex encoding")
    return p.parse_args()


def sync():
    easyfhe.cuda.synchronize()


def release_all(objects):
    for item in objects:
        release_if_supported(item)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class EncodeTrace:
    """Timing the two actual EasyFHE Encode stages outside main wall samples."""

    def __init__(self):
        self.events = {"stage1": [], "stage2": []}
        self.originals = {}

    def __enter__(self):
        for name, label in (("encode_stage1_packed", "stage1"),
                            ("encode_stage2", "stage2")):
            original = getattr(constants_mod, name)
            self.originals[name] = original

            def wrapped(*args, _original=original, _label=label, **kwargs):
                values = args[0] if _label == "stage1" else args[0].encoded_values
                rows = int(values.shape[0]) if len(values.shape) > 1 else 1
                start = easyfhe.cuda.Event(enable_timing=True)
                end = easyfhe.cuda.Event(enable_timing=True)
                start.record()
                try:
                    return _original(*args, **kwargs)
                finally:
                    end.record()
                    self.events[_label].append((start, end, rows))

            setattr(constants_mod, name, wrapped)
        return self

    def __exit__(self, *_):
        for name, original in self.originals.items():
            setattr(constants_mod, name, original)

    def summary(self):
        sync()
        return {
            f"{label}_{field}": value
            for label, events in self.events.items()
            for field, value in (
                ("calls", len(events)),
                ("rows", sum(rows for _, _, rows in events)),
                ("gpu_s", sum(start.elapsed_time(end) / 1000.0
                              for start, end, _ in events)),
            )
        }


def decrypt_complex(context, ciphers):
    decoded = []
    for cipher in ciphers:
        negative_i = fhe.homo_mul_i(cipher, context.context, negative=True)
        try:
            decoded.append(np.asarray(context.decrypt(cipher), dtype=np.float64)
                           + 1j * np.asarray(context.decrypt(negative_i), dtype=np.float64))
        finally:
            release_if_supported(negative_i)
    return np.stack(decoded)


def benchmark_page(page, *, gate, up, packed, layout, context, config,
                   chunks, options):
    active = min(4096, gate.shape[0] - page * 4096)
    if active <= 0:
        raise ValueError(f"page {page} has no Gate/Up features")
    span = slice(page * 4096, page * 4096 + active)
    joint = np.zeros((4096, 4096), dtype=np.complex128)
    real_gate = np.zeros((4096, 4096), dtype=np.float64)
    real_up = np.zeros((4096, 4096), dtype=np.float64)
    real_gate[:active] = 0.5 * options.gate_alpha * gate[span]
    real_up[:active] = 0.5 * up[span]
    joint[:active] = 0.5 * (real_gate[:active] + 1j * real_up[:active])
    source = tuple(context.encrypt(row, level=int(context.context.L) - options.limbs,
                                   slots=32768) for row in packed)
    assert len(source) == 8
    operator = LinearOperator(crypto_context=context, token_lanes=8, config=config)
    weights = {
        "joint": LinearWeight(joint, dtype=np.complex128, crypto_context=context),
        "gate": LinearWeight(real_gate, dtype=np.float64, crypto_context=context),
        "up": LinearWeight(real_up, dtype=np.float64, crypto_context=context),
    }
    counts = Counter()
    chunk_indices = tuple(chunk.index for chunk in chunks)
    encoded_rows = sum(chunk.giant_count * 64 for chunk in chunks)
    originals = {}
    try:
        for name, weight in weights.items():
            weight.prepare(config)
            original = weight.plaintext
            originals[name] = original

            def counted(*args, _original=original, _name=name, **kwargs):
                counts[f"{_name}_calls"] += 1
                counts[f"{_name}_rows"] += int(args[1].giant_count) * 64
                return _original(*args, **kwargs)

            weight.plaintext = counted
        sync()

        def run(mode, *, diagnostic=False, decrypt=False):
            counts.clear()
            operator.clear_baby_cache()
            sync()
            easyfhe.cuda.reset_peak_memory_stats()
            baseline_bytes = int(easyfhe.cuda.memory_allocated())
            started = time.perf_counter()
            trace = EncodeTrace() if diagnostic else None
            input_conjugates = ()
            projections = ()
            gate_outputs = ()
            up_outputs = ()
            profiles = []
            timing = Counter()
            try:
                if trace is not None:
                    trace.__enter__()
                if mode == "joint":
                    step = time.perf_counter()
                    input_conjugates = tuple(
                        fhe.homo_rotate(cipher, int(context.context.M) - 1,
                                        context.context) for cipher in source)
                    sync()
                    timing["input_conjugation_s"] = time.perf_counter() - step
                    projections, profile = operator.apply(
                        source + input_conjugates, weights["joint"],
                        chunk_indices=chunk_indices)
                    profiles.append(profile)
                    step = time.perf_counter()
                    gate_values = []
                    up_values = []
                    for direct, conjugate_eval in zip(projections[:8], projections[8:]):
                        conjugated = fhe.homo_rotate(
                            conjugate_eval, int(context.context.M) - 1,
                            context.context)
                        try:
                            gate_values.append(fhe.homo_add(direct, conjugated,
                                                            context.context))
                            delta = fhe.homo_sub(direct, conjugated, context.context)
                            try:
                                up_values.append(fhe.homo_mul_i(
                                    delta, context.context, negative=True))
                            finally:
                                release_if_supported(delta)
                        finally:
                            release_if_supported(conjugated)
                    gate_outputs, up_outputs = tuple(gate_values), tuple(up_values)
                    sync()
                    timing["recovery_s"] = time.perf_counter() - step
                else:
                    gate_outputs, gate_profile = operator.apply(
                        source, weights["gate"], chunk_indices=chunk_indices)
                    up_outputs, up_profile = operator.apply(
                        source, weights["up"], chunk_indices=chunk_indices)
                    profiles.extend((gate_profile, up_profile))
                sync()
                timing["wall_s"] = time.perf_counter() - started
                row = {"mode": mode, "diagnostic": diagnostic,
                       "wall_s": timing["wall_s"],
                       "input_conjugation_s": timing["input_conjugation_s"],
                       "recovery_s": timing["recovery_s"],
                       "pack_encode_s": sum(float(p.get("pack_encode", 0)) for p in profiles),
                       "online_s": sum(float(p.get("online", 0)) for p in profiles),
                       "baby_precompute_s": sum(float(p.get("baby_precompute", 0)) for p in profiles),
                       "rescale_s": sum(float(p.get("rescale", 0)) for p in profiles),
                       "encoded_rows": sum(counts[f"{name}_rows"] for name in weights),
                       "encode_batch_calls": sum(counts[f"{name}_calls"] for name in weights),
                       "plaintext_counts": dict(counts),
                       "peak_allocated_gib": easyfhe.cuda.max_memory_allocated() / 2**30,
                       "incremental_peak_gib": (easyfhe.cuda.max_memory_allocated() - baseline_bytes) / 2**30,
                       "output_limbs": sorted({int(x.state.cur_limbs) for x in gate_outputs + up_outputs})}
                if trace is not None:
                    event_trace = trace
                    trace.__exit__(None, None, None)
                    trace = None
                    row.update(event_trace.summary())
                expected_calls = len(chunks) * (1 if mode == "joint" else 2)
                expected_rows = encoded_rows * (1 if mode == "joint" else 2)
                if row["encode_batch_calls"] != expected_calls or row["encoded_rows"] != expected_rows:
                    raise RuntimeError(f"unexpected Encode count: {row}")
                if diagnostic:
                    for stage in ("stage1", "stage2"):
                        if row[f"{stage}_calls"] != expected_calls or row[f"{stage}_rows"] != expected_rows:
                            raise RuntimeError(f"unexpected {stage} count: {row}")
                print(json.dumps({"page": page, **row}), flush=True)
                decoded = (decrypt_complex(context, gate_outputs),
                           decrypt_complex(context, up_outputs)) if decrypt else None
                return row, decoded
            finally:
                if trace is not None:
                    trace.__exit__(None, None, None)
                release_all(input_conjugates)
                release_all(projections)
                release_all(gate_outputs)
                release_all(up_outputs)

        for mode in ("joint", "separate"):
            run(mode)
        order = (("joint", "separate", "separate", "joint") if options.pairs == 2
                 else ("joint", "separate"))
        samples = [run(mode)[0] for mode in order]
        diagnostic = {}
        decoded = {}
        for mode in ("joint", "separate"):
            diagnostic[mode], decoded[mode] = run(mode, diagnostic=True, decrypt=True)
        verification = {}
        for index, name in enumerate(("gate", "up")):
            error = np.abs(decoded["joint"][index] - decoded["separate"][index])
            max_abs = float(error.max())
            relative = max_abs / max(1e-12, float(np.abs(decoded["joint"][index]).max()))
            verification[f"{name}_max_abs"] = max_abs
            verification[f"{name}_relative"] = relative
            if relative > 5e-3:
                raise RuntimeError(f"joint/separate {name} mismatch on page {page}: {relative}")
        # Full BSGS only: a small plaintext oracle validates the two halves.
        if encoded_rows == 4096:
            for index, (name, matrix) in enumerate((("gate", gate), ("up", up))):
                decoded_rows = layout.unpack(decoded["joint"][index])
                expected = 0.5 * (options.gate_alpha if name == "gate" else 1.0) * (
                    layout.unpack(packed).astype(np.float64) @ matrix[span.start:span.start + min(16, active)].T)
                actual = decoded_rows[:, : min(16, active)]
                verification[f"{name}_sample_plain_relative"] = (
                    float(np.max(np.abs(actual - expected)))
                    / max(1e-12, float(np.max(np.abs(expected)))))
        median = {mode: {key: statistics.median(s[key] for s in samples
                                                 if s["mode"] == mode)
                         for key in ("wall_s", "pack_encode_s", "online_s",
                                     "baby_precompute_s", "recovery_s",
                                     "peak_allocated_gib", "incremental_peak_gib")}
                  for mode in ("joint", "separate")}
        result = {"page": page, "active_features": active,
                  "selected_chunks": list(chunk_indices),
                  "ct_pt_entries_per_arm": 16 * encoded_rows,
                  "joint_encode_rows": encoded_rows,
                  "separate_encode_rows": 2 * encoded_rows,
                  "joint_projection_ciphertexts": 16,
                  "separate_projection_ciphertexts": 16,
                  "joint_input_conjugate_ciphertexts": 8,
                  "separate_input_conjugate_ciphertexts": 0,
                  "final_ciphertexts_per_arm": 16,
                  "median": median, "diagnostic": diagnostic,
                  "samples": samples, "verification": verification}
        return result
    finally:
        operator.close()
        for weight in weights.values():
            weight.close()
        release_all(source)


def main():
    a = parse_args()
    pages = tuple(int(item) for item in a.pages.split(","))
    if not pages or any(page not in range(4) for page in pages):
        raise ValueError("--pages must use 0,1,2,3")
    if a.depth <= a.limbs or a.rows % 64:
        raise ValueError("need depth>limbs and rows divisible by 64")
    config = LinearOperatorConfig(
        baby_steps=64, baby_anchor_step=64, max_plaintext_rows=a.rows,
        reuse_baby_rotations=True, hoist_strategy="normal",
        constant_cache_mode="none", raw_weight_policy="device", encode_depth=0)
    plan = config.plan(dimension=4096, token_lanes=8)
    chunks = plan.chunks if a.chunk_count == 0 else plan.chunks[:a.chunk_count]
    if not chunks or a.chunk_count < 0 or a.chunk_count > len(plan.chunks):
        raise ValueError("invalid --chunk-count")
    rotations = linear_rotations(
        dimension=4096, token_lanes=8, slots=32768,
        include_conjugation=True, operator_config=config)
    started = time.perf_counter()
    context = create_sim_context(
        SimulatorConfig(logN=16, maxLevelsRemaining=a.depth, dnum=5,
                        dcrtBits=59, firstMod=60, device="cuda"),
        input_dim=4096, block_out=8, rotations=rotations,
        include_default_rotations=False)
    print(json.dumps({"phase": "context_ready", "L": int(context.context.L),
                      "setup_s": time.perf_counter() - started,
                      "rotation_keys": len(rotations)}), flush=True)
    layout = ComplexTokenDiagonalLayout(seq_len=128, dimension=4096, slots=32768)
    with np.load(a.fixture, allow_pickle=False) as data:
        hidden = np.asarray(data["hidden_in"], dtype=np.float64)
        gate = np.asarray(data["gate_proj"], dtype=np.float64)
        up = np.asarray(data["up_proj"], dtype=np.float64)
        if hidden.shape != (128, 4096) or gate.shape != (14336, 4096) or up.shape != gate.shape:
            raise ValueError(f"invalid fixture shapes: {hidden.shape}, {gate.shape}, {up.shape}")
        packed = layout.pack(hidden)
        reports = []
        for page in pages:
            print(json.dumps({"phase": "start", "page": page}), flush=True)
            report = benchmark_page(page, gate=gate, up=up, packed=packed,
                                    layout=layout, context=context, config=config,
                                    chunks=chunks, options=a)
            reports.append(report)
            print(json.dumps({"phase": "complete", "page": page,
                              "verification": report["verification"]}), flush=True)
    result = {"experiment": "joint complex vs separate real Gate/Up Encode",
              "script_sha256": sha256(Path(__file__)),
              "fixture": str(a.fixture), "fixture_sha256": sha256(a.fixture),
              "settings": {"depth": a.depth, "L": int(context.context.L),
                           "limbs": a.limbs, "rows": a.rows,
                           "chunks": len(chunks), "pairs": a.pairs,
                           "gate_alpha": a.gate_alpha},
              "reports": reports}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"result": str(a.out), "reports": len(reports)}), flush=True)


if __name__ == "__main__":
    main()
