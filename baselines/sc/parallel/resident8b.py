"""Probe, then optionally retain, SC 8B DecodeMV tables on each GPU.

The original linked driver and FHE layer.forward remain the arithmetic source
of truth.  The pool substitutes only its weight load/layer constructor while a
Decode pass runs.  It is opt-in and supports W8 Llama-3-8B P0 only.
"""

from __future__ import annotations

import gc
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable


LAYERS = 32
CHECKPOINTS = (4, 8, 16, 24, 32)


class ProbeComplete(Exception):
    def __init__(self, report: dict[str, Any]):
        super().__init__("resident-weight memory probe completed")
        self.report = report


def gpu_memory(runtime: Any) -> dict[str, int]:
    """Read the physical GPU, not CUDA-visible ordinal 0 in a masked child."""

    gpu_uuid = str(runtime.audit.gpu_uuid)
    line = subprocess.check_output(
        ["nvidia-smi", "-i", gpu_uuid, "--query-gpu=memory.used,memory.total",
         "--format=csv,noheader,nounits"], text=True,
    ).strip().splitlines()[0]
    used, total = (int(item.strip()) for item in line.split(","))
    return {"used_mib": used, "total_mib": total, "free_mib": total - used}


def host_available_mib() -> int:
    """Read Linux MemAvailable without importing a monitoring dependency."""

    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    raise RuntimeError("/proc/meminfo has no MemAvailable field")


class ResidentSession:
    """One rank's immutable layer/table pool and its lifecycle."""

    def __init__(self, base: Any, runtime: Any, eng: Any, geo: Any, entries: list[Any],
                 out: Path, *, memory_reader: Callable[[], dict[str, int]] | None = None,
                 host_reader: Callable[[], int] | None = None,
                 sync: Callable[[], None] | None = None,
                 minimum_free_mib: int = 20000,
                 minimum_host_available_mib: int = 65536):
        self.base, self.runtime, self.eng, self.geo = base, runtime, eng, geo
        self.entries = list(entries)
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.memory_reader = memory_reader or (lambda: gpu_memory(runtime))
        self.host_reader = host_reader or host_available_mib
        self.sync = sync or self._cuda_sync
        self.minimum_free_mib = int(minimum_free_mib)
        self.minimum_host_available_mib = int(minimum_host_available_mib)
        self.unstaged_load = os.environ.get("SC_RESIDENT_UNSTAGED_LOAD", "0") == "1"
        self.original_load = base.load
        self.original_layer = base.DecodeMVLayer
        self.original_projector = base.DeviceProjector
        self.layers: list[Any] = []
        self.weights: list[tuple[Any, Any]] = []
        self.index_by_weight_id: dict[int, int] = {}
        self.releasers: list[tuple[Any, Callable[[], None]]] = []
        self.report: dict[str, Any] = {
            "mode": "sc8b_w8_decode_mv_resident_weights",
            "rank": int(runtime.rank), "world_size": int(runtime.world_size),
            "weight_load_mode": "unstaged" if self.unstaged_load else "staged",
            "checkpoints": [], "completed_layers": 0,
            "minimum_free_mib": self.minimum_free_mib,
            "minimum_host_available_mib": self.minimum_host_available_mib,
            "arithmetic": "unchanged DecodeMVLayer.forward and DeviceProjector.project",
        }
        self.released = False

    @staticmethod
    def _cuda_sync() -> None:
        import easyfhe as torch
        torch.cuda.synchronize()

    def _write(self) -> None:
        (self.out / "decode_weight_preload.json").write_text(json.dumps(self.report, indent=2))

    def _probe_forward(self, client: Any, query: Any, saved: Any) -> dict[str, Any]:
        """Stress one real FHE Decode layer at each residency checkpoint."""

        import numpy as np
        ct = client.encrypt(self.geo.encode_x(query), slots=self.base.SLOTS)
        cache = {name: tuple(self.base.gpu_cipher(cipher) for cipher in saved[0][name])
                 for name in ("k", "v")}
        self.layers[0].times.clear()
        self.sync()
        started = time.perf_counter()
        output = self.layers[0].forward(ct, cache)
        self.sync()
        elapsed = time.perf_counter() - started
        hidden = self.geo.decode_x(
            client.decrypt(self.eng.norm(output)).cpu().numpy()[:self.base.SLOTS].real)
        finite = bool(np.isfinite(hidden).all())
        if not finite:
            raise RuntimeError("resident-weight probe produced a nonfinite L0 hidden state")
        result = {"l0_forward_seconds": round(elapsed, 4),
                  "l0_hidden_max_abs": float(np.max(np.abs(hidden))),
                  "l0_hidden_finite": finite,
                  "gpu_after_forward": self.memory_reader()}
        del output, ct, cache, hidden
        self.eng.clear_cache()
        gc.collect()
        self.sync()
        return result

    def prepare(self, *, max_layers: int = LAYERS, probe: bool = False,
                client: Any = None, query: Any = None, saved: Any = None) -> dict[str, Any]:
        import easyfhe as torch

        if len(self.entries) != LAYERS or not 1 <= max_layers <= LAYERS:
            raise ValueError("resident-weight experiment requires 32 SC 8B layers")
        if not probe and max_layers != LAYERS:
            raise ValueError("full Decode residency must cover all 32 layers")
        if probe and (client is None or query is None or saved is None):
            raise ValueError("memory probe requires real query and encrypted KV cache")
        self.sync()
        torch.cuda.empty_cache()
        self.report["gpu_before"] = self.memory_reader()
        peak = int(self.report["gpu_before"]["used_mib"])
        started = time.perf_counter()
        try:
            for index, entry in enumerate(self.entries[:max_layers]):
                t_layer = time.perf_counter()
                weights, calibration = self.original_load(
                    entry, staged=not self.unstaged_load)
                try:
                    projector = self.original_projector(self.eng, self.geo)
                    layer = self.original_layer(
                        self.eng, weights, calibration, self.geo, projector, 128, upload=True)
                finally:
                    # NumPy arrays are independent of the temporary staged files.
                    if not self.unstaged_load:
                        self.base.unstage(entry)
                release = layer.P.release
                layer.P.release = lambda: None  # original decode_pass calls release every layer
                self.releasers.append((layer.P, release))
                self.layers.append(layer)
                self.weights.append((weights, calibration))
                self.index_by_weight_id[id(weights)] = index
                self.sync()
                # Tables remain referenced by the pool. Return only unused
                # construction/Prefill allocator blocks before reading VRAM.
                torch.cuda.empty_cache()
                mem = self.memory_reader()
                peak = max(peak, int(mem["used_mib"]))
                self.report["completed_layers"] = index + 1
                self.report.setdefault("layer_setup_seconds", []).append(
                    round(time.perf_counter() - t_layer, 4))
                # All ranks must stop at the same point.  Otherwise one rank
                # can reject its memory budget while peers enter L0 forward
                # and wait forever in a ciphertext collective.
                status = self.runtime.all_gather(
                    f"sc8b_resident/layer{index}/memory_guard",
                    {"rank": int(self.runtime.rank), "free_mib": int(mem["free_mib"]),
                     "host_available_mib": self.host_reader()},
                )
                if len(status) != int(self.runtime.world_size):
                    raise RuntimeError("incomplete resident-weight memory guard")
                tightest = min(int(item["free_mib"]) for item in status)
                host_available = min(int(item["host_available_mib"]) for item in status)
                self.report["minimum_observed_host_available_mib"] = min(
                    host_available,
                    self.report.get("minimum_observed_host_available_mib", host_available),
                )
                if tightest < self.minimum_free_mib:
                    raise MemoryError(
                        f"least-free GPU has only {tightest} MiB free "
                        f"after {index + 1} resident layers")
                if host_available < self.minimum_host_available_mib:
                    raise MemoryError(
                        f"host has only {host_available} MiB available after "
                        f"{index + 1} resident layers")
                if index + 1 in CHECKPOINTS:
                    checkpoint = {"resident_layers": index + 1, "gpu_after_upload": mem}
                    if probe:
                        checkpoint.update(self._probe_forward(client, query, saved))
                        peak = max(peak, checkpoint["gpu_after_forward"]["used_mib"])
                    self.report["checkpoints"].append(checkpoint)
                    self.report["sampled_peak_mib"] = peak
                    self._write()
            self.report["setup_seconds"] = round(time.perf_counter() - started, 4)
            self.report["gpu_after"] = self.memory_reader()
            self.report["sampled_peak_mib"] = max(peak, self.report["gpu_after"]["used_mib"])
            self.report["passed"] = True
            self._write()
            return self.report
        except Exception as exc:
            self.report["setup_seconds"] = round(time.perf_counter() - started, 4)
            self.report["sampled_peak_mib"] = peak
            self.report["error"] = f"{type(exc).__name__}: {exc}"
            self.report["passed"] = False
            self._write()
            self.release()
            raise

    def run_pass(self, original_decode_pass: Callable[..., Any], *args: Any,
                 **kwargs: Any) -> Any:
        """Call the original pass with only load/construction redirected to this pool."""

        if len(self.layers) != LAYERS:
            raise RuntimeError("cannot run full Decode before all 32 layers are resident")
        base = self.base
        previous_load, previous_layer = base.load, base.DecodeMVLayer

        def cached_load(entry: Any, staged: bool = False) -> tuple[Any, Any]:
            index = int(entry["layer"])
            if not 0 <= index < LAYERS:
                raise ValueError(f"unexpected Decode layer {index}")
            return self.weights[index]

        def cached_layer(eng: Any, weights: Any, calibration: Any, geo: Any,
                         projector: Any, position: int, *, upload: bool = True) -> Any:
            index = self.index_by_weight_id.get(id(weights))
            if (index is None or eng is not self.eng or geo is not self.geo or position != 128
                    or not upload or calibration is not self.weights[index][1]):
                raise ValueError("resident Decode layer constructor arguments changed")
            return self.layers[index]

        base.load = cached_load
        base.DecodeMVLayer = cached_layer
        try:
            return original_decode_pass(*args, **kwargs)
        finally:
            base.load = previous_load
            base.DecodeMVLayer = previous_layer

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        for projector, original_release in reversed(self.releasers):
            projector.release = original_release
            original_release()
        self.releasers.clear()
        self.layers.clear()
        self.weights.clear()
        self.index_by_weight_id.clear()
        gc.collect()
        self.report["gpu_after_release"] = self.memory_reader()
        self._write()


def make_decode_pass(base: Any, runtime: Any, original_decode_pass: Callable[..., Any],
                     *, total_passes: int, probe_only: bool = False) -> Callable[..., Any]:
    """Patch point used by the isolated multigpu runner."""

    state: dict[str, Any] = {"session": None, "calls": 0}

    def decode_pass(client: Any, eng: Any, geo: Any, manifest: Any, entries: Any,
                    plain_inputs: Any, saved: Any, query: Any, **kwargs: Any) -> Any:
        if state["session"] is None:
            out = Path(kwargs["out"])
            session = ResidentSession(
                base, runtime, eng, geo, list(entries), out,
                minimum_free_mib=int(os.environ.get("SC_RESIDENT_MIN_FREE_MIB", "20000")),
                minimum_host_available_mib=int(os.environ.get(
                    "SC_RESIDENT_MIN_HOST_AVAILABLE_MIB", "65536")),
            )
            state["session"] = session
            report = session.prepare(
                max_layers=int(os.environ.get("SC_RESIDENT_PROBE_LIMIT", str(LAYERS))),
                probe=probe_only, client=client, query=query, saved=saved,
            )
            if probe_only:
                session.release()
                raise ProbeComplete(report)
        session = state["session"]
        failed = False
        try:
            result = session.run_pass(
                original_decode_pass, client, eng, geo, manifest, entries,
                plain_inputs, saved, query, **kwargs)
            state["calls"] += 1
            return result
        except Exception:
            failed = True
            raise
        finally:
            if failed or state["calls"] >= total_passes:
                session.release()

    return decode_pass
