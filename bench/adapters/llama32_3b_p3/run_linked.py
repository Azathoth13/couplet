#!/usr/bin/env python3
"""P3 Prefill→encrypted KV→Decode experiment with the existing P1 Decode route.

Forward normal ``bench/t2_prefill_decode.py`` options after ``--``. The
Prefill-only change is P3 early Q filtering; Decode uses the established P1
three-query route so P1 and P3 share the same downstream path.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from contextlib import nullcontext

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from bench.adapters.llama32_3b_p3.run_prefill import adapt_attention_core, adapt_query_target


def worker_entry_p3_linked(rank, preset, endpoints, **kwargs):
    if int(preset.world_size) != 1:
        raise ValueError("isolated P3 linked adapter supports world_size=1 only")
    from bench.adapters.llama32_3b_p3.decode_route import adapt_pairfold_attention, _shared_route_qk

    import llama3fhe.operators.attention.distributed.distributed_input as distributed_input
    import llama3fhe.parallel.prefill.stations as stations
    import llama3fhe.operators.attention.pairfold.pairfold_operator as operator
    import llama3fhe.operators.attention.pairfold.pairfold_decode as decode

    original_target = distributed_input.attention_input_target_shard_fhe
    original_core = stations.attention_core_station
    original_attention = operator.pairfold_attention_fhe
    original_qk = operator.qk_pairfold_pages_fhe

    def target(lane_ciphers, **target_kwargs):
        return adapt_query_target(original_target, lane_ciphers, **target_kwargs)

    def core(query_items, key_items, value_items, **core_kwargs):
        return adapt_attention_core(original_core, query_items, key_items, value_items,
                                    **core_kwargs)

    def decode_attention(*args, **attention_kwargs):
        return adapt_pairfold_attention(original_attention, *args, **attention_kwargs)

    distributed_input.attention_input_target_shard_fhe = target
    stations.attention_core_station = core
    operator.pairfold_attention_fhe = decode_attention
    operator.qk_pairfold_pages_fhe = _shared_route_qk(original_qk, decode.__dict__)
    try:
        if os.environ.get("P3_STRUCTURED_METADATA_CACHE", "0") == "1":
            from bench.adapters.llama32_3b_p3.metadata_cache_overlay import structured_metadata_cache, print_summary
            metadata_cache = structured_metadata_cache()
        else:
            metadata_cache = nullcontext(None)
        with metadata_cache as metadata_stats:
            from bench.t2_prefill_decode import worker_entry
            try:
                return worker_entry(rank, preset, endpoints, **kwargs)
            finally:
                if metadata_stats is not None:
                    print_summary(metadata_stats)
    finally:
        distributed_input.attention_input_target_shard_fhe = original_target
        stations.attention_core_station = original_core
        operator.pairfold_attention_fhe = original_attention
        operator.qk_pairfold_pages_fhe = original_qk


def main() -> int:
    default_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=default_root)
    args, forwarded = parser.parse_known_args()
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    source_root = args.source_root.resolve()
    if not (source_root / "bench/t2_prefill_decode.py").is_file():
        raise FileNotFoundError(source_root / "bench/t2_prefill_decode.py")
    sys.path.insert(0, str(source_root))
    from bench import t2_prefill_decode
    from llama3fhe.parallel.launch import spawn_workers as launch_spawn_workers

    def patched_spawn(preset, _entry, **spawn_kwargs):
        return launch_spawn_workers(preset, worker_entry_p3_linked, **spawn_kwargs)

    t2_prefill_decode.spawn_workers = patched_spawn
    print(json.dumps({
        "head_packing_ab": "p3_early_q_filter_plus_p1_decode",
        "scope": "Prefill Q-only target filter; physical K/V/KV emit/PairFM; P1 Decode",
        "source_root": str(source_root), "forwarded": forwarded,
    }), flush=True)
    sys.argv = [str(source_root / "bench/t2_prefill_decode.py"), *forwarded]
    return int(t2_prefill_decode.main())


if __name__ == "__main__":
    raise SystemExit(main())
