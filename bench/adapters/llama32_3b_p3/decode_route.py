#!/usr/bin/env python3
"""Single-card 3B Decode P1 adapter for the existing PairFold attention core.

The physical Q32/KV8 PairFold carrier, KV cache and W_O remain unchanged.
Select the six live `(group, query_offset)` streams (offsets 0,1,2 in each
group); the existing QK, Softmax head_rows mask, and PV kernels then skip the
dummy offset 3 and return one folded O carrier with zeros in those residues.
No ciphertext repack or zero insertion is needed at the W_O boundary.

`p1-naive` uses the current one-query-at-a-time route. `p1-shared` installs
an experiment-local copy of the existing QK function with its group-route
loop changed to use the selected offsets; it preserves the production
shared sparse baby basis without evaluating dummy offset 3. The control
keeps the original function. Forward normal `bench/t2_decode.py` arguments after `--`. `--fixtures-dir`
plus `--poly-kv` supplies a standalone L16 reference cache; that is a
separate scope from Prefill-produced ciphertext KV linkage.
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import sys


LIVE_STREAMS = ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2))
DUMMY_HEADS = (3, 7, 11, 15, 19, 23, 27, 31)


def adapt_pairfold_attention(original, *args, **kwargs):
    layout = kwargs["pairfold_layout"]
    score_layout = kwargs["score_layout"]
    if (int(layout.query_heads), int(layout.key_value_heads), int(layout.gqa_ratio),
            int(layout.head_dim), int(layout.seq_len), int(layout.key_value_groups)) != (32, 8, 4, 128, 128, 2):
        raise ValueError("Decode P1 requires physical Llama-3.2-3B PairFold geometry")
    if int(score_layout.query_heads) != 32 or int(score_layout.key_value_heads) != 8:
        raise ValueError("Decode P1 requires physical Q32 score layout")
    if kwargs.get("query_selection") is not None:
        raise ValueError("this isolated Decode P1 adapter supports one rank only")
    from llama3fhe.operators.attention.pairfold.pairfold_decode import query_selection_head_rows
    active = query_selection_head_rows(LIVE_STREAMS, layout=layout)
    if active != tuple(h for h in range(32) if h not in DUMMY_HEADS):
        raise AssertionError(f"unexpected active Decode head rows: {active}")
    options = dict(kwargs)
    options["query_selection"] = LIVE_STREAMS
    # The current QK kernel's shared sparse query route is enabled only when
    # query_selection=None. Explicit selection uses six individual routes;
    # that cost is part of P1 and must be visible in the stage breakdown.
    result = original(*args, **options)
    result.profile["head_p1_live_streams"] = float(len(LIVE_STREAMS))
    result.profile["head_p1_dummy_heads"] = float(len(DUMMY_HEADS))
    return result


def _shared_route_qk(original, module_globals):
    """Compile a narrow overlay of the production QK function in its globals."""
    source = inspect.getsource(original)
    old_guard = "grouped = query_group_extractor is not None and query_selection is None"
    new_guard = "grouped = query_group_extractor is not None"
    old_groups = """        if grouped:
            grouped_streams = tuple(
                tuple((group, query_offset) for query_offset in range(int(layout.gqa_ratio)))
                for group in range(int(layout.key_value_groups))
            )
        else:
            grouped_streams = tuple((stream,) for stream in streams)"""
    new_groups = """        if grouped:
            grouped_streams = tuple(
                tuple(stream for stream in streams if stream[0] == group)
                for group in range(int(layout.key_value_groups))
                if any(stream[0] == group for stream in streams)
            )
        else:
            grouped_streams = tuple((stream,) for stream in streams)"""
    if source.count(old_guard) != 1 or source.count(old_groups) != 1:
        raise RuntimeError("production QK source changed; inspect the shared-route overlay before running")
    source = source.replace(old_guard, new_guard).replace(old_groups, new_groups)
    namespace = {}
    exec(compile(source, inspect.getsourcefile(original) or "<qk-overlay>", "exec"),
         module_globals, namespace)
    return namespace[original.__name__]


def _worker_entry_p1(rank, preset, endpoints, *, shared_route: bool, **kwargs):
    if int(preset.world_size) != 1:
        raise ValueError("this isolated Decode P1 adapter supports world_size=1 only")
    import llama3fhe.operators.attention.pairfold.pairfold_operator as operator
    original = operator.pairfold_attention_fhe
    original_qk = operator.qk_pairfold_pages_fhe
    if shared_route:
        import llama3fhe.operators.attention.pairfold.pairfold_decode as decode
        operator.qk_pairfold_pages_fhe = _shared_route_qk(
            original_qk, decode.__dict__)

    def adapted(*args, **attention_kwargs):
        return adapt_pairfold_attention(original, *args, **attention_kwargs)

    operator.pairfold_attention_fhe = adapted
    try:
        from bench.t2_decode import worker_entry
        return worker_entry(rank, preset, endpoints, **kwargs)
    finally:
        operator.pairfold_attention_fhe = original
        operator.qk_pairfold_pages_fhe = original_qk


def worker_entry_p1_naive(rank, preset, endpoints, **kwargs):
    return _worker_entry_p1(rank, preset, endpoints, shared_route=False, **kwargs)


def worker_entry_p1_shared(rank, preset, endpoints, **kwargs):
    return _worker_entry_p1(rank, preset, endpoints, shared_route=True, **kwargs)


def main() -> int:
    default_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=default_root)
    parser.add_argument("--mode", choices=("p0", "p1-naive", "p1-shared"), required=True)
    args, forwarded = parser.parse_known_args()
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    source_root = args.source_root.resolve()
    if not (source_root / "bench/t2_decode.py").is_file():
        raise FileNotFoundError(source_root / "bench/t2_decode.py")
    sys.path.insert(0, str(source_root))
    from bench import t2_decode
    if args.mode != "p0":
        from llama3fhe.parallel.launch import spawn_workers as launch_spawn_workers
        worker = worker_entry_p1_shared if args.mode == "p1-shared" else worker_entry_p1_naive

        def patched_spawn(preset, _entry, **spawn_kwargs):
            return launch_spawn_workers(preset, worker, **spawn_kwargs)

        t2_decode.spawn_workers = patched_spawn
    print(json.dumps({"head_packing_ab": args.mode,
                      "scope": "single-token Decode with separately encrypted plaintext-reference KV cache",
                      "source_root": str(source_root), "forwarded": forwarded}), flush=True)
    sys.argv = [str(source_root / "bench/t2_decode.py"), *forwarded]
    return int(t2_decode.main())


if __name__ == "__main__":
    raise SystemExit(main())
