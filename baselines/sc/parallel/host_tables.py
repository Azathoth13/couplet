"""Cache exact SC DecodeMV middle tables in CPU RAM and stream one layer to GPU.

Only weight preparation changes. Original DecodeMVLayer.forward, input
broadcasts, projections, HalfBoot and linked ciphertext KV remain unchanged.
The original driver constructs each layer in its existing GPU worker thread;
this pool substitutes a CPU-table upload for repeated GPU table construction.
Uploads stay in the Decode pass and are reported separately from forward.
"""
from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

from .resident8b import (
    LAYERS, ResidentSession,
)


class HostTableSession(ResidentSession):
    """Retain exact CPU middle tables; GPU tables live for one layer only."""

    def __init__(self, *args: Any, minimum_free_mib: int = 4096, **kwargs: Any):
        super().__init__(*args, minimum_free_mib=minimum_free_mib, **kwargs)
        self.host_tables: list[dict[str, tuple[Any, ...]]] = []
        self.report.update(mode="sc8b_cpu_middle_tables_streamed_decode",
                           weight_load_mode="unstaged", host_table_bytes=0,
                           total_d2h_seconds=0.0)

    def _write(self) -> None:
        (self.out / "decode_host_table_setup.json").write_text(
            json.dumps(self.report, indent=2))

    def prepare(self) -> dict[str, Any]:
        import easyfhe as torch

        if len(self.entries) != LAYERS:
            raise ValueError("host-table experiment requires all 32 SC 8B layers")
        self.sync()
        torch.cuda.empty_cache()
        self.report["gpu_before"] = self.memory_reader()
        peak = int(self.report["gpu_before"]["used_mib"])
        started = time.perf_counter()
        try:
            for index, entry in enumerate(self.entries):
                t_layer = time.perf_counter()
                weights, calibration = self.original_load(entry, staged=False)
                projector = self.original_projector(self.eng, self.geo)
                layer = self.original_layer(
                    self.eng, weights, calibration, self.geo, projector, 128, upload=True)
                self.layers.append(layer)
                self.weights.append((weights, calibration))
                self.index_by_weight_id[id(weights)] = index
                self.sync()
                after_build = self.memory_reader()
                peak = max(peak, int(after_build["used_mib"]))
                t_d2h = time.perf_counter()
                # W1 tuple is (middle, mx); W>1 additionally retains pc groups.
                tables = {name: (value[0].cpu().numpy().copy(), *value[1:])
                          for name, value in layer.P.dev.items()}
                self.sync()
                d2h_seconds = time.perf_counter() - t_d2h
                table_bytes = sum(int(value[0].nbytes) for value in tables.values())
                self.host_tables.append(tables)
                # Keep the layer, calibration and raw weights for the existing
                # plaintext oracle, but retain no layer's GPU middle tables.
                layer.P.release()
                torch.cuda.empty_cache()
                self.sync()
                after_release = self.memory_reader()
                self.report["completed_layers"] = index + 1
                self.report["host_table_bytes"] += table_bytes
                self.report["total_d2h_seconds"] += d2h_seconds
                self.report.setdefault("layers", []).append(dict(
                    layer=index, setup_seconds=time.perf_counter() - t_layer,
                    d2h_seconds=d2h_seconds, bytes=table_bytes,
                    gpu_after_build=after_build, gpu_after_release=after_release))
                # Stop peers together if the actual single-layer construction
                # or host pool exhausted its budget; no extra probe is run.
                statuses = self.runtime.all_gather(
                    f"sc8b_host_tables/layer{index}/memory",
                    dict(free_mib=int(after_build["free_mib"]),
                         host_available_mib=self.host_reader()))
                if min(int(row["free_mib"]) for row in statuses) < self.minimum_free_mib:
                    raise MemoryError("single-layer GPU table build exceeded the free-memory budget")
                if min(int(row["host_available_mib"]) for row in statuses) < self.minimum_host_available_mib:
                    raise MemoryError("host middle-table pool exceeded available RAM")
                self.report["sampled_peak_mib"] = peak
                self._write()
            self.report.update(passed=True, setup_seconds=time.perf_counter() - started,
                               sampled_peak_mib=peak, gpu_after=self.memory_reader())
            self._write()
            return self.report
        except Exception as exc:
            self.report.update(passed=False, setup_seconds=time.perf_counter() - started,
                               sampled_peak_mib=peak,
                               error=f"{type(exc).__name__}: {exc}")
            self.release()
            raise

    def run_pass(self, original_decode_pass: Callable[..., Any], *args: Any,
                 **kwargs: Any) -> Any:
        import easyfhe as torch

        if len(self.host_tables) != LAYERS or self.released:
            raise RuntimeError("CPU tables must cover all 32 layers before Decode")
        previous_load, previous_layer = self.base.load, self.base.DecodeMVLayer
        uploads: list[dict[str, Any]] = []

        def cached_load(entry: Any, staged: bool = False) -> tuple[Any, Any]:
            return self.weights[int(entry["layer"])]

        def cached_layer(eng: Any, weights: Any, calibration: Any, geo: Any,
                         projector: Any, position: int, *, upload: bool = True) -> Any:
            index = self.index_by_weight_id[id(weights)]
            layer = self.layers[index]
            if eng is not self.eng or geo is not self.geo or position != 128 or not upload:
                raise ValueError("host-table Decode constructor arguments changed")
            started = time.perf_counter()
            layer.P.dev = {name: (torch.as_tensor(value[0], device="cuda"), *value[1:])
                           for name, value in self.host_tables[index].items()}
            self.sync()
            uploads.append(dict(layer=index, h2d_seconds=time.perf_counter() - started,
                                bytes=sum(int(value[0].nbytes)
                                          for value in self.host_tables[index].values())))
            return layer

        self.base.load, self.base.DecodeMVLayer = cached_load, cached_layer
        started = time.perf_counter()
        succeeded = False
        try:
            result = original_decode_pass(*args, **kwargs)
            if [row["layer"] for row in uploads] != list(range(LAYERS)):
                raise RuntimeError("Decode did not upload every layer exactly once")
            succeeded = True
            return result
        finally:
            wall = time.perf_counter() - started
            self.base.load, self.base.DecodeMVLayer = previous_load, previous_layer
            # Also release a table if an original forward/terminal check failed.
            for layer in self.layers:
                layer.P.release()
            index = int(kwargs.get("index", 0))
            report = dict(mode="cpu_middle_tables_per_layer_h2d", rank=int(self.runtime.rank),
                          world_size=int(self.runtime.world_size), pass_index=index,
                          measured=bool(kwargs.get("measured", False)), passed=succeeded,
                          layers=uploads,
                          total_h2d_seconds=sum(row["h2d_seconds"] for row in uploads),
                          total_h2d_bytes=sum(row["bytes"] for row in uploads),
                          wrapper_wall_seconds=wall,
                          includes_first_reference_setup=index == 0,
                          wrapper_wall_scope="includes original encryption, oracle wrapper and terminal checks; H2D remains online",
                          original_chain_scope="original driver's Decode chain wall; first upload is separately recorded",
                          asynchronous_copy=False)
            (self.out / f"decode_host_table_upload_pass{index}.json").write_text(
                json.dumps(report, indent=2))

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        for layer in self.layers:
            layer.P.release()
        self.layers.clear()
        self.weights.clear()
        self.index_by_weight_id.clear()
        self.host_tables.clear()
        gc.collect()
        self.report["gpu_after_release"] = self.memory_reader()
        self._write()


def make_decode_pass(base: Any, runtime: Any, original_decode_pass: Callable[..., Any],
                     *, total_passes: int) -> Callable[..., Any]:
    """Drop-in linked-driver hook, including its independent oracle wrapper."""
    state: dict[str, Any] = {"session": None, "calls": 0}

    def decode_pass(client: Any, eng: Any, geo: Any, manifest: Any, entries: Any,
                    plain_inputs: Any, saved: Any, query: Any, **kwargs: Any) -> Any:
        if state["session"] is None:
            session = HostTableSession(
                base, runtime, eng, geo, list(entries), Path(kwargs["out"]),
                minimum_free_mib=int(os.environ.get("SC_HOST_TABLE_MIN_FREE_MIB", "4096")),
                minimum_host_available_mib=int(os.environ.get(
                    "SC_HOST_TABLE_MIN_HOST_AVAILABLE_MIB", "65536")))
            state["session"] = session
            session.prepare()
        session = state["session"]
        failed = False
        try:
            result = session.run_pass(original_decode_pass, client, eng, geo, manifest,
                                      entries, plain_inputs, saved, query, **kwargs)
            state["calls"] += 1
            return result
        except Exception:
            failed = True
            raise
        finally:
            if failed or state["calls"] >= total_passes:
                session.release()

    return decode_pass
