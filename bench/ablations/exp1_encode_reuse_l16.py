#!/usr/bin/env python3
"""Measure Encode reuse inside the actual single-card L16 Prefill station chain.

Run twice with ODIN_EXP1_MODE=reuse and ODIN_EXP1_MODE=reencode. Both runs use
the normal t2_prefill_decode CLI, key plan, structured weight descriptors,
512-row chunks, L16 fixture input, and an untimed L16 warmup. The control
re-encodes a weight chunk for every ciphertext MAC after the first; all MACs
and their order stay identical. This control repeats GPU packing as well as
Encode, so its projection delta must not be called pure Encode time.

ODIN_EXP1_RECORDS names a JSON file for per-projection GPU events, input limb
counts, and Encode row counts. T2_STAGE_SYNC=0 and --encode-depth 0 are required
to match the current linked production path.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench import t2_prefill_decode as runner  # noqa: E402

_PRODUCTION_WORKER = runner.worker_entry


def worker_entry(rank, preset, endpoints, **kwargs):
    """Install an experiment-only hook after CUDA has been pinned in the child."""

    import easyfhe
    import easyfhe.fhe.constants as constants_mod
    import numpy as np

    from llama3fhe.operators.linear.operator import LinearOperator
    from llama3fhe.operators.linear.resident_plaintexts import RESIDENT
    from llama3fhe.operators.linear.weight import LinearWeight

    mode = os.environ.get("ODIN_EXP1_MODE", "").strip()
    trace_encode = os.environ.get("ODIN_EXP1_TRACE_ENCODE", "0") == "1"
    if mode not in {"reuse", "reencode"}:
        raise ValueError("ODIN_EXP1_MODE must be reuse or reencode")
    if RESIDENT.enabled():
        raise RuntimeError("cross-token resident plaintext cache must be disabled for this control")

    original_project = LinearOperator.project
    original_apply = LinearOperator.apply
    # ODIN_MULTI_MAC installs its fused method *inside* worker_entry. Capture
    # that production method after installation, then wrap it for the control.
    original_mac = LinearOperator._mac_chunk
    original_plaintext = LinearWeight.plaintext
    original_stage1 = constants_mod.encode_stage1_packed
    original_stage2 = constants_mod.encode_stage2
    original_patches_install = runner.acceleration_patches.install_from_env
    records = []
    active = None
    per_case_calls = {"qkv": 0, "gate_up": 0, "down": 0}

    def case_for_project(dtype):
        caller = sys._getframe(2).f_code.co_filename.replace("\\", "/")
        if caller.endswith("/operators/linear/prefill/qkv_kernel.py"):
            return "qkv"
        if caller.endswith("/operators/linear/prefill/mlp.py"):
            return "gate_up" if np.dtype(dtype).kind == "c" else "down"
        return None

    def project(self, input_ciphers, weight, *, dtype, **project_kwargs):
        nonlocal active
        case = case_for_project(dtype)
        if case is None:
            return original_project(self, input_ciphers, weight, dtype=dtype, **project_kwargs)
        if active is not None:
            raise RuntimeError("nested tracked linear projection")
        if int(self.config.encode_depth) != 0:
            raise RuntimeError("L16 Encode A/B requires encode_depth=0")
        if str(self.config.constant_cache_mode) != "none":
            raise RuntimeError("L16 Encode A/B requires constant_cache_mode=none")
        ordinal = per_case_calls[case]
        per_case_calls[case] += 1
        warmup_count = 1 if case == "qkv" else 4
        rec = {
            "case": case,
            "ordinal": ordinal,
            "phase": "warmup" if ordinal < warmup_count else "timed",
            "mode": mode,
            "input_limbs": sorted({int(c.state.cur_limbs) for c in input_ciphers}),
            "input_ciphers": len(input_ciphers),
            "max_plaintext_rows": int(self.config.max_plaintext_rows),
            "encode_batches": 0,
            "encode_rows": 0,
        }
        if trace_encode:
            rec["_encode_events"] = {"stage1": [], "stage2": []}
        start_event = easyfhe.cuda.Event(enable_timing=True)
        end_event = easyfhe.cuda.Event(enable_timing=True)
        rec["_events"] = (start_event, end_event)
        active = rec
        start_event.record()
        started = time.perf_counter()
        try:
            outputs, profile = original_project(
                self, input_ciphers, weight, dtype=dtype, **project_kwargs
            )
            rec["host_project_s"] = time.perf_counter() - started
            rec["operator_profile"] = {
                str(key): float(value) for key, value in profile.items()
                if ".chunk_" not in str(key)
            }
            return outputs, profile
        finally:
            end_event.record()
            active = None
            records.append(rec)

    def apply(self, input_ciphers, weight, **apply_kwargs):
        rec = active
        if rec is None:
            return original_apply(self, input_ciphers, weight, **apply_kwargs)
        if len(rec["input_limbs"]) != 1:
            raise RuntimeError(f"mixed input limbs in {rec['case']}: {rec['input_limbs']}")
        plan = self.plan(weight.dimension)
        chunks = (
            plan.chunks if apply_kwargs.get("chunk_indices") is None else
            tuple(plan.chunks[int(index)] for index in apply_kwargs["chunk_indices"])
        )
        rec["weight_kind"] = "structured" if weight._structured_weight is not None else "dense"
        rec["baby_steps"] = int(plan.baby_steps)
        rec["chunk_count"] = len(chunks)
        rec["diagonals"] = sum(int(chunk.giant_count) * int(plan.baby_steps) for chunk in chunks)
        rec["logical_ct_pt_entries"] = rec["diagonals"] * rec["input_ciphers"]
        rec["_weight"] = weight
        rec["_plan"] = plan
        rec["_level"] = int(self.crypto_context.level_for_cipher(input_ciphers[0]))
        try:
            return original_apply(self, input_ciphers, weight, **apply_kwargs)
        finally:
            rec.pop("_weight", None)
            rec.pop("_plan", None)
            rec.pop("_level", None)

    def plaintext(self, plan, chunk, *, level, operator_config):
        rec = active
        if rec is not None and rec.get("_weight") is self:
            rec["encode_batches"] += 1
            rec["encode_rows"] += int(chunk.giant_count) * int(plan.baby_steps)
        return original_plaintext(
            self, plan, chunk, level=level, operator_config=operator_config
        )

    def mac(self, chunk, plaintexts, input_ciphers, accumulators,
            cached_babies, *, plan, strategy, baby_anchor_step):
        rec = active
        if rec is None or mode == "reuse":
            return original_mac(
                self, chunk, plaintexts, input_ciphers, accumulators,
                cached_babies, plan=plan, strategy=strategy,
                baby_anchor_step=baby_anchor_step,
            )
        weight = rec.get("_weight")
        if weight is None:
            raise RuntimeError("re-Encode control lost its active weight")
        for index, cipher in enumerate(input_ciphers):
            current = plaintexts if index == 0 else weight.plaintext(
                plan, chunk, level=rec["_level"], operator_config=self.config
            )
            one_accumulator = [accumulators[index]]
            original_mac(
                self, chunk, current, (cipher,), one_accumulator,
                [cached_babies[index]], plan=plan, strategy=strategy,
                baby_anchor_step=baby_anchor_step,
            )
            accumulators[index] = one_accumulator[0]

    def trace_stage(label, original):
        def traced(*args, **kwargs):
            rec = active
            if rec is None:
                return original(*args, **kwargs)
            start = easyfhe.cuda.Event(enable_timing=True)
            end = easyfhe.cuda.Event(enable_timing=True)
            start.record()
            try:
                return original(*args, **kwargs)
            finally:
                end.record()
                rec["_encode_events"][label].append((start, end))
        return traced

    def install_patches_then_hook(*args, **kwargs):
        nonlocal original_mac
        installed = original_patches_install(*args, **kwargs)
        original_mac = LinearOperator._mac_chunk
        LinearOperator._mac_chunk = mac
        return installed

    LinearOperator.project = project
    LinearOperator.apply = apply
    LinearWeight.plaintext = plaintext
    runner.acceleration_patches.install_from_env = install_patches_then_hook
    if trace_encode:
        constants_mod.encode_stage1_packed = trace_stage("stage1", original_stage1)
        constants_mod.encode_stage2 = trace_stage("stage2", original_stage2)
    try:
        result = _PRODUCTION_WORKER(rank, preset, endpoints, **kwargs)
        easyfhe.cuda.synchronize()
        for rec in records:
            start_event, end_event = rec.pop("_events")
            rec["gpu_projection_s"] = float(start_event.elapsed_time(end_event) / 1000.0)
            expected = rec["chunk_count"] * (1 if mode == "reuse" else rec["input_ciphers"])
            if rec["encode_batches"] != expected:
                raise RuntimeError(f"Encode batch count mismatch: {rec}")
            expected_rows = rec["diagonals"] * (1 if mode == "reuse" else rec["input_ciphers"])
            if rec["encode_rows"] != expected_rows:
                raise RuntimeError(f"Encode row count mismatch: {rec}")
            if rec["weight_kind"] != "structured":
                raise RuntimeError(f"expected production structured weights: {rec}")
            if trace_encode:
                encode_events = rec.pop("_encode_events")
                for label in ("stage1", "stage2"):
                    pairs = encode_events[label]
                    if len(pairs) != expected:
                        raise RuntimeError(f"{label} call count mismatch: {rec}")
                    rec[f"{label}_cuda_s"] = sum(
                        float(start.elapsed_time(end) / 1000.0)
                        for start, end in pairs
                    )
        if {case: per_case_calls[case] for case in per_case_calls} != {"qkv": 2, "gate_up": 8, "down": 8}:
            raise RuntimeError(f"expected warmup plus measured L16: {per_case_calls}")
        report = {"mode": mode, "rank": int(rank), "records": records}
        path_raw = os.environ.get("ODIN_EXP1_RECORDS", "").strip()
        if path_raw:
            path = Path(path_raw)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("EXP1_L16_RECORDS " + json.dumps(report, separators=(",", ":")), flush=True)
        return result
    finally:
        LinearOperator.project = original_project
        LinearOperator.apply = original_apply
        LinearOperator._mac_chunk = original_mac
        LinearWeight.plaintext = original_plaintext
        runner.acceleration_patches.install_from_env = original_patches_install
        if trace_encode:
            constants_mod.encode_stage1_packed = original_stage1
            constants_mod.encode_stage2 = original_stage2


def main() -> int:
    args = runner.parse_args()
    mode = os.environ.get("ODIN_EXP1_MODE", "").strip()
    if mode not in {"reuse", "reencode"}:
        raise SystemExit("set ODIN_EXP1_MODE=reuse or reencode")
    if not (
        args.layers == "16" and args.world_size == 1
        and args.decode_tokens == 0 and args.warmup_layer0
        and args.emit_kv_cache and args.encode_depth == 0
        and args.max_plaintext_rows in (None, 512)
        and os.environ.get("T2_STAGE_SYNC") == "0"
    ):
        raise SystemExit(
            "required: --layers 16 --world-size 1 --decode-tokens 0 "
            "--warmup-layer0 --encode-depth 0, KV emit on, rows auto/512, T2_STAGE_SYNC=0"
        )
    runner.worker_entry = worker_entry
    return runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
