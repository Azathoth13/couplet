"""SC8B oracle replay without GPU-resident Decode weights.

The reference uses the exact linked-driver plaintext functions.  Rank 0 loads
one weight archive at a time without staging it in shared memory; only the
33 small hidden states and per-layer geometry are sent to other ranks.  The
original FHE Decode pass still loads its weights with its existing lookahead.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

from .oracle import _PlainProxy, _ReplayInputs, _empty_rows, _sha256


def precompute_unstaged(base: Any, entries: Any, plain_inputs: Any,
                        query: np.ndarray) -> tuple[np.ndarray, list[dict], list[dict]]:
    """Run the original plaintext equations with one unstaged layer at a time."""

    x_ref = np.array(query, copy=True)
    states, geometries, timings = [x_ref.copy()], [], []
    original_plain = base.plain
    for idx, entry in enumerate(entries):
        t_load = time.perf_counter()
        w, cal = base.load(entry, staged=False)
        load_seconds = time.perf_counter() - t_load
        t_oracle = time.perf_counter()
        opts = original_plain.decode_geometry(cal, w)
        kc, vc = original_plain.kv_cache(
            plain_inputs[f"layer{idx:02d}"].astype(np.float64), w,
            **{key: value for key, value in opts.items() if key != "heads"},
        )
        x_ref, _ = original_plain.decode_layer(
            x_ref, (kc, vc), w, position=128, **opts)
        states.append(np.array(x_ref, copy=True))
        geometries.append(opts)
        timings.append({"layer": idx, "load_seconds": load_seconds,
                        "oracle_seconds": time.perf_counter() - t_oracle})
        del w, cal, kc, vc
        base.gc.collect()
    packed = np.stack(states)
    if not np.isfinite(packed).all():
        raise RuntimeError("SC8B plaintext Decode oracle produced nonfinite hidden state")
    return packed, geometries, timings


def make_decode_pass(base: Any, runtime: Any, original_decode_pass: Callable[..., Any]) -> Callable[..., Any]:
    """Remove only repeated online plaintext reference work from Decode."""

    cached_states: np.ndarray | None = None
    cached_geometries: list[dict] | None = None

    def decode_pass(client: Any, eng: Any, geo: Any, manifest: Any,
                    entries: Any, plain_inputs: Any, saved: Any,
                    query: np.ndarray, **kwargs: Any) -> Any:
        nonlocal cached_states, cached_geometries
        entries = list(entries)
        output = Path(kwargs["out"])
        if cached_states is None:
            started = time.perf_counter()
            if runtime.rank == 0:
                states, geometries, timings = precompute_unstaged(
                    base, entries, plain_inputs, query)
                payload = (states, geometries)
            else:
                payload, timings = None, []
            t_collective = time.perf_counter()
            gathered = runtime.all_gather("sc8b_decode_oracle/no_resident_reference", payload)
            collective_seconds = time.perf_counter() - t_collective
            if len(gathered) != runtime.world_size or gathered[0] is None or any(
                    item is not None for item in gathered[1:]):
                raise RuntimeError("SC8B reference broadcast failed")
            cached_states, cached_geometries = gathered[0]
            cached_states = np.asarray(cached_states)
            if (cached_states.shape != (len(entries) + 1, *np.asarray(query).shape)
                    or len(cached_geometries) != len(entries)
                    or not np.isfinite(cached_states).all()
                    or not np.array_equal(cached_states[0], query)):
                raise RuntimeError("SC8B rank reference trajectory/query mismatch")
            (output / "decode_oracle_precompute.json").write_text(json.dumps({
                "mode": "replay_no_resident", "rank": int(runtime.rank),
                "root_compute_seconds": (t_collective - started) if runtime.rank == 0 else 0.0,
                "collective_seconds": collective_seconds,
                "total_setup_seconds": time.perf_counter() - started,
                "reference_sha256": _sha256(cached_states),
                "reference_shape": list(cached_states.shape),
                "root_layer_oracle_seconds": timings,
            }, indent=2))

        rows = _empty_rows(entries)
        proxy = _PlainProxy(base.plain, base, entries, rows, cached_states, cached_geometries)
        original_plain, original_unstage, original_gc = base.plain, base.unstage, base.gc
        original_clear = eng.clear_cache

        def timed_unstage(entry: Any) -> Any:
            idx = proxy.current
            t0 = time.perf_counter()
            try:
                return original_unstage(entry)
            finally:
                if idx is not None:
                    rows[idx]["unstage_seconds"] += time.perf_counter() - t0

        def timed_clear_cache(*args: Any, **kw: Any) -> Any:
            idx = proxy.current
            t0 = time.perf_counter()
            try:
                return original_clear(*args, **kw)
            finally:
                if idx is not None:
                    rows[idx]["clear_cache_seconds"] += time.perf_counter() - t0

        def timed_gc_collect(*args: Any, **kw: Any) -> Any:
            idx = proxy.current
            t0 = time.perf_counter()
            try:
                return original_gc.collect(*args, **kw)
            finally:
                if idx is not None:
                    rows[idx]["gc_seconds"] += time.perf_counter() - t0
                    proxy.finish_cleanup()

        base.plain = proxy
        base.unstage = timed_unstage
        base.gc = SimpleNamespace(collect=timed_gc_collect)
        eng.clear_cache = timed_clear_cache
        started = time.perf_counter()
        try:
            result = original_decode_pass(
                client, eng, geo, manifest, entries,
                _ReplayInputs(len(entries)), saved, query, **kwargs)
            proxy.verify_complete()
        finally:
            eng.clear_cache = original_clear
            base.gc = original_gc
            base.unstage = original_unstage
            base.plain = original_plain
        profile = {
            "mode": "replay_no_resident", "rank": int(runtime.rank),
            "pass_index": int(kwargs["index"]), "measured": bool(kwargs["measured"]),
            "wrapper_wall_seconds": time.perf_counter() - started,
            "original_chain_seconds": result["chain_seconds"],
            "original_layer_seconds_total": result["layer_seconds_total"],
            "reference_sha256": _sha256(cached_states),
            "totals": {field: sum(row[field] for row in rows)
                       for field in rows[0] if field != "layer"},
            "layers": rows,
        }
        (output / f"decode_oracle_profile_pass{kwargs['index']}.json").write_text(
            json.dumps(profile, indent=2))
        return result

    return decode_pass
