"""Move the independent SC Decode plaintext oracle out of the online FHE loop.

The existing linked ``decode_pass`` and every ``DecodeMVLayer.forward`` are
called unchanged.  In ``inline`` mode, this module only times the original
plaintext oracle and layer cleanup.  In ``replay`` mode, rank 0 computes the
same complete reference trajectory once after resident weights are ready,
shares its small hidden-state trajectory, and the original per-layer oracle
statements replay those states without running plaintext matrix arithmetic.

The linked driver's terminal hidden/logit/token/top-5 checks still execute
against the independent plaintext trajectory on every rank.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np


MODES = ("inline", "replay")


def _sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def compute_reference(base: Any, entries: Any, plain_inputs: Any,
                      query: np.ndarray) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Run the linked driver's exact plaintext reference equations once."""

    original_plain = base.plain
    x_ref = np.array(query, copy=True)
    states = [x_ref.copy()]
    timings = []
    for idx, entry in enumerate(entries):
        started = time.perf_counter()
        w, cal = base.load(entry, staged=True)  # resident session has already cached these
        opts = original_plain.decode_geometry(cal, w)
        kc, vc = original_plain.kv_cache(
            plain_inputs[f"layer{idx:02d}"].astype(np.float64), w,
            **{key: value for key, value in opts.items() if key != "heads"},
        )
        x_ref, _ = original_plain.decode_layer(
            x_ref, (kc, vc), w, position=128, **opts)
        states.append(np.array(x_ref, copy=True))
        timings.append({"layer": idx, "oracle_seconds": time.perf_counter() - started})
        del w, cal, kc, vc
    packed = np.stack(states)
    if not np.isfinite(packed).all():
        raise RuntimeError("plaintext Decode oracle produced a nonfinite hidden state")
    return packed, timings


class _InputToken:
    """Cheap replacement for the unused ``plain_inputs[layer].astype`` result."""

    def __init__(self, layer: int, kind: str = "input") -> None:
        self.layer, self.kind = layer, kind

    def astype(self, dtype: Any) -> "_InputToken":
        if np.dtype(dtype) != np.dtype(np.float64):
            raise RuntimeError("SC Decode oracle input dtype changed")
        return self


class _ReplayInputs:
    def __init__(self, count: int) -> None:
        self.count = count

    def __getitem__(self, key: str) -> _InputToken:
        if not key.startswith("layer") or len(key) != 7:
            raise RuntimeError(f"unexpected plaintext input key {key!r}")
        index = int(key[5:])
        if not 0 <= index < self.count or key != f"layer{index:02d}":
            raise RuntimeError(f"unexpected plaintext input key {key!r}")
        return _InputToken(index)


class _PlainProxy:
    """Instrument only the three reference calls in the original Decode loop."""

    def __init__(self, original: Any, base: Any, entries: Any,
                 rows: list[dict[str, Any]], states: np.ndarray | None,
                 geometries: list[dict[str, Any]] | None) -> None:
        self.original, self.base = original, base
        self.entries, self.rows = list(entries), rows
        self.states, self.geometries = states, geometries
        self.next_index = 0
        self.current: int | None = None
        self.started = 0.0
        self.phase = "idle"

    def __getattr__(self, name: str) -> Any:
        return getattr(self.original, name)

    def decode_geometry(self, cal: Any, w: Any) -> dict[str, Any]:
        if self.phase != "idle" or self.next_index >= len(self.entries):
            raise RuntimeError("SC Decode oracle geometry call order changed")
        idx = self.next_index
        if int(self.entries[idx]["layer"]) != idx:
            raise RuntimeError("SC Decode oracle layer order changed")
        self.current = idx
        self.started = time.perf_counter()
        self.phase = "geometry"
        t0 = time.perf_counter()
        result = (self.original.decode_geometry(cal, w) if self.states is None
                  else self.geometries[idx])
        self.rows[idx]["geometry_seconds"] += time.perf_counter() - t0
        return result

    def kv_cache(self, inputs: Any, w: Any, **kwargs: Any) -> Any:
        if self.phase != "geometry" or self.current is None:
            raise RuntimeError("SC Decode oracle KV call order changed")
        idx = self.current
        self.phase = "kv_cache"
        t0 = time.perf_counter()
        if self.states is None:
            result = self.original.kv_cache(inputs, w, **kwargs)
        else:
            if not isinstance(inputs, _InputToken) or (inputs.layer, inputs.kind) != (idx, "input"):
                raise RuntimeError("replay received a different plaintext input")
            result = (_InputToken(idx, "k"), _InputToken(idx, "v"))
        self.rows[idx]["kv_cache_seconds"] += time.perf_counter() - t0
        return result

    def decode_layer(self, x_ref: np.ndarray, kv: Any, w: Any,
                     *, position: int, **kwargs: Any) -> tuple[np.ndarray, Any]:
        if self.phase != "kv_cache" or self.current is None or position != 128:
            raise RuntimeError("SC Decode oracle layer call order or position changed")
        idx = self.current
        t0 = time.perf_counter()
        if self.states is None:
            result = self.original.decode_layer(x_ref, kv, w, position=position, **kwargs)
        else:
            if (not np.array_equal(x_ref, self.states[idx])
                    or not isinstance(kv, tuple) or len(kv) != 2
                    or any(not isinstance(item, _InputToken) or item.layer != idx
                           for item in kv)
                    or (kv[0].kind, kv[1].kind) != ("k", "v")):
                raise RuntimeError("replay reference state or KV call contract changed")
            result = (self.states[idx + 1], None)
        self.rows[idx]["decode_layer_seconds"] += time.perf_counter() - t0
        self.rows[idx]["oracle_seconds"] += time.perf_counter() - self.started
        self.phase = "cleanup"
        return result

    def finish_cleanup(self) -> None:
        if self.phase != "cleanup" or self.current != self.next_index:
            raise RuntimeError("SC Decode oracle cleanup call order changed")
        self.next_index += 1
        self.current = None
        self.phase = "idle"

    def verify_complete(self) -> None:
        if self.phase != "idle" or self.next_index != len(self.entries):
            raise RuntimeError(
                f"SC Decode oracle instrumented {self.next_index}/{len(self.entries)} layers")


def _empty_rows(entries: Any) -> list[dict[str, Any]]:
    fields = ("oracle_seconds", "geometry_seconds", "kv_cache_seconds",
              "decode_layer_seconds", "unstage_seconds", "clear_cache_seconds",
              "gc_seconds")
    return [{"layer": idx, **{field: 0.0 for field in fields}}
            for idx, _ in enumerate(entries)]


def make_decode_pass(base: Any, runtime: Any, original_decode_pass: Callable[..., Any],
                     *, mode: str) -> Callable[..., Any]:
    """Wrap the original pass without modifying FHE forward/projection code."""

    if mode not in MODES:
        raise ValueError(f"SC_DECODE_ORACLE_MODE must be one of {MODES}")
    cached_states: np.ndarray | None = None
    cached_geometries: list[dict[str, Any]] | None = None
    precompute: dict[str, Any] | None = None

    def decode_pass(client: Any, eng: Any, geo: Any, manifest: Any,
                    entries: Any, plain_inputs: Any, saved: Any, query: np.ndarray,
                    **kwargs: Any) -> Any:
        nonlocal cached_states, cached_geometries, precompute
        entries = list(entries)
        output = Path(kwargs["out"])
        if mode == "replay" and cached_states is None:
            t_setup = time.perf_counter()
            if runtime.rank == 0:
                states, layer_times = compute_reference(base, entries, plain_inputs, query)
                payload = states
            else:
                payload, layer_times = None, []
            t_collective = time.perf_counter()
            gathered = runtime.all_gather("sc3b_decode_oracle/reference_states", payload)
            collective_seconds = time.perf_counter() - t_collective
            if len(gathered) != runtime.world_size or gathered[0] is None or any(
                    state is not None for state in gathered[1:]):
                raise RuntimeError("SC Decode oracle root-only reference broadcast failed")
            cached_states = np.asarray(gathered[0])
            expected_shape = (len(entries) + 1, *np.asarray(query).shape)
            if cached_states.shape != expected_shape or not np.isfinite(cached_states).all():
                raise RuntimeError("SC Decode oracle reference trajectory has wrong shape or NaN")
            if not np.array_equal(cached_states[0], query):
                raise RuntimeError("SC Decode rank query differs from rank0 plaintext reference")
            t_geometry = time.perf_counter()
            cached_geometries = []
            for entry in entries:
                w, cal = base.load(entry, staged=True)
                cached_geometries.append(base.plain.decode_geometry(cal, w))
            geometry_seconds = time.perf_counter() - t_geometry
            precompute = {
                "mode": mode, "rank": int(runtime.rank),
                "root_compute_seconds": (t_collective - t_setup) if runtime.rank == 0 else 0.0,
                "collective_seconds": collective_seconds,
                "geometry_precompute_seconds": geometry_seconds,
                "total_setup_seconds": time.perf_counter() - t_setup,
                "reference_sha256": _sha256(cached_states),
                "reference_shape": list(cached_states.shape),
                "root_layer_oracle_seconds": layer_times,
            }
            (output / "decode_oracle_precompute.json").write_text(json.dumps(precompute, indent=2))

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
                _ReplayInputs(len(entries)) if mode == "replay" else plain_inputs,
                saved, query, **kwargs)
            proxy.verify_complete()
        finally:
            eng.clear_cache = original_clear
            base.gc = original_gc
            base.unstage = original_unstage
            base.plain = original_plain

        profile = {
            "mode": mode, "rank": int(runtime.rank),
            "pass_index": int(kwargs["index"]), "measured": bool(kwargs["measured"]),
            "wrapper_wall_seconds": time.perf_counter() - started,
            "original_chain_seconds": result["chain_seconds"],
            "original_layer_seconds_total": result["layer_seconds_total"],
            "reference_sha256": (_sha256(cached_states) if cached_states is not None else None),
            "totals": {field: sum(row[field] for row in rows)
                       for field in rows[0] if field != "layer"},
            "layers": rows,
        }
        (output / f"decode_oracle_profile_pass{kwargs['index']}.json").write_text(
            json.dumps(profile, indent=2))
        return result

    return decode_pass
