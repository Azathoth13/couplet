#!/usr/bin/env python3
"""Isolated 3B P3: remove padded Q pairs before Attention input gather.

The physical H4096 QKV carrier and K/V/cache paths are unchanged. P3 filters
Q pair indices 3, 7, 11, 15 in the Q-only target shard, then runs the P1
12-pair Attention core and restores four zero O_C pairs for physical PairFM.

Example::

  python run_odin_p3_l16.py --source-root . -- \
    --preset rtx6000pro-high --model llama3.2-3b-padded --world-size 1 --devices 0 \
    --fixtures fixtures/llama32_3b_padded_model --layers 16 --input-level 35 \
    --verify-every 1 --warmup-layer0 --output artifacts/odin_p3_l16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


LIVE_PAIR_INDICES = (0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14)
DUMMY_PAIR_INDICES = (3, 7, 11, 15)
PHYSICAL_PAIR_INDICES = tuple(range(16))


def check_physical_3b_layout(layout) -> None:
    geometry = (
        int(layout.query_heads), int(layout.key_value_heads),
        int(layout.head_dim), int(layout.seq_len), int(layout.query_pair_count),
    )
    if geometry != (32, 8, 128, 128, 16):
        raise ValueError(f"P3 requires physical padded 3B Q32/KV8/D128/S128 geometry, got {geometry}")


def adapt_query_target(original, lane_ciphers, **kwargs):
    """Remove exactly four dummy targets before gather/axis2/expand/repeat."""
    requested = tuple(int(i) for i in kwargs["query_pair_indices"])
    key_indices = tuple(kwargs["key_pair_indices"])
    value_indices = tuple(kwargs["value_carrier_indices"])
    if not requested:
        return original(lane_ciphers, **kwargs)
    if key_indices or value_indices:
        raise ValueError("P3 expects a Q-only target shard; K/V must remain separate")
    check_physical_3b_layout(kwargs["attention_layout"])
    if requested != PHYSICAL_PAIR_INDICES:
        raise ValueError(f"P3 expects all 16 physical Q targets, got {requested}")
    kwargs = dict(kwargs)
    kwargs["query_pair_indices"] = LIVE_PAIR_INDICES
    result = original(lane_ciphers, **kwargs)
    actual = tuple(int(i) for i, _ in result.query_items)
    if actual != LIVE_PAIR_INDICES:
        result.release()
        raise AssertionError(f"P3 Q target output indices differ: {actual}")
    result.scheduler_stats["p3_dummy_q_pairs_skipped"] = 4.0
    # The linked benchmark persists attn_input.stage_seconds, but not its
    # scheduler_stats. A zero-time marker makes the P3 path auditable in the
    # summary without contaminating either measured wall time or stage sums.
    result.stage_seconds["p3_early_q_filter"] = 0.0
    return result


def adapt_attention_core(original, query_items, key_items, value_items, **kwargs):
    """Run the P1 compact core, then restore the physical O_C/PairFM ABI."""
    check_physical_3b_layout(kwargs["attention_layout"])
    incoming = tuple((int(i), c) for i, c in query_items)
    if tuple(i for i, _ in incoming) != LIVE_PAIR_INDICES:
        raise ValueError(f"P3 core expects 12 live Q pairs, got {[i for i, _ in incoming]}")
    core_kwargs = dict(kwargs)
    core_kwargs["ownership"] = None
    result = original(incoming, key_items, value_items, **core_kwargs)
    output = tuple((int(i), c) for i, c in result.output_items)
    if tuple(i for i, _ in output) != LIVE_PAIR_INDICES:
        result.release()
        raise AssertionError(f"P3 compact core returned pairs {[i for i, _ in output]}")

    context = kwargs["context"]
    anchor = output[0][1]
    zeros = []
    started = time.perf_counter()
    try:
        for _ in DUMMY_PAIR_INDICES:
            zeros.append(context.fhe.homo_sub(anchor, anchor, context.context))
        restored = tuple(sorted(output + tuple(zip(DUMMY_PAIR_INDICES, zeros, strict=True))))
        if tuple(i for i, _ in restored) != PHYSICAL_PAIR_INDICES:
            raise AssertionError("PairFM would receive incomplete physical pair indices")
        limbs = {int(c.state.cur_limbs) for _, c in restored}
        if len(limbs) != 1:
            raise AssertionError(f"PairFM O_C input limbs differ: {limbs}")
        from llama3fhe.backend import profile_sync_device
        profile_sync_device(context.device)
        result.stage_seconds["head_zero_restore"] = time.perf_counter() - started
        result.level_trace["head_zero_restore"] = tuple(sorted(
            {int(context.level_for_cipher(c)) for c in zeros}
        ))
        result.output_limbs = tuple(int(c.state.cur_limbs) for _, c in restored)
        result.output_items = restored
        return result
    except BaseException:
        from llama3fhe.backend import release_if_supported
        for cipher in zeros:
            release_if_supported(cipher)
        result.release()
        raise


def worker_entry_p3(rank, preset, endpoints, **kwargs):
    if int(preset.world_size) != 1:
        raise ValueError("isolated P3 supports world_size=1 only")
    import llama3fhe.operators.attention.distributed.distributed_input as distributed_input
    import llama3fhe.parallel.prefill.stations as stations

    original_target = distributed_input.attention_input_target_shard_fhe
    original_core = stations.attention_core_station

    def target(lane_ciphers, **target_kwargs):
        return adapt_query_target(original_target, lane_ciphers, **target_kwargs)

    def core(query_items, key_items, value_items, **core_kwargs):
        return adapt_attention_core(original_core, query_items, key_items, value_items, **core_kwargs)

    distributed_input.attention_input_target_shard_fhe = target
    stations.attention_core_station = core
    try:
        from bench.t2_model import worker_entry
        return worker_entry(rank, preset, endpoints, **kwargs)
    finally:
        distributed_input.attention_input_target_shard_fhe = original_target
        stations.attention_core_station = original_core


def main() -> int:
    default_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=default_root)
    args, forwarded = parser.parse_known_args()
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    source_root = args.source_root.resolve()
    if not (source_root / "bench/t2_model.py").is_file():
        raise FileNotFoundError(source_root / "bench/t2_model.py")
    sys.path.insert(0, str(source_root))
    from bench import t2_model
    from llama3fhe.parallel.launch import spawn_workers as launch_spawn_workers

    def patched_spawn(preset, _entry, **spawn_kwargs):
        return launch_spawn_workers(preset, worker_entry_p3, **spawn_kwargs)

    t2_model.spawn_workers = patched_spawn
    print(json.dumps({
        "head_packing_ab": "p3_early_q_filter",
        "scope": "physical QKV/KV emit/PairFM; Q-only target shard filters 4 dummy pairs",
        "source_root": str(source_root), "forwarded": forwarded,
    }), flush=True)
    sys.argv = [str(source_root / "bench/t2_model.py"), *forwarded]
    return int(t2_model.main())


if __name__ == "__main__":
    raise SystemExit(main())
