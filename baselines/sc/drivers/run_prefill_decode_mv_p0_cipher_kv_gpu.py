"""Optional GPU-resident KV control for the SC Prefix=0 ciphertext-KV runner.

Prefill still parks its encrypted K/V on the host to bound the Prefill peak.
After Prefill, this control uploads all 32 layers once and keeps them resident
for the complete-token warmup and both same-state Decode measurements.  It
shares the source, context, correctness checks, and CLI with the host-cache
runner.  The one-time upload is written to ``gpu_cache_preload.json``; it is
not charged to each Decode token.

Run with the same environment and arguments as
``run_prefill_decode_mv_p0_cipher_kv.py``, using a distinct output directory.
The experiment fails explicitly if the GPU cannot hold the cache; it never
silently changes to the host-cache path.
"""
from __future__ import annotations

import gc
import json
import subprocess
import time
from pathlib import Path

import easyfhe as torch

import run_prefill_decode_mv_p0_cipher_kv as base


_capture_host = base.capture_prefill_caches
_upload = base.gpu_cipher


def gpu_memory():
    output = subprocess.check_output(
        ["nvidia-smi", "-i", "0", "--query-gpu=memory.used,memory.total",
         "--format=csv,noheader,nounits"], text=True,
    ).strip().splitlines()[0]
    used, total = (int(item.strip()) for item in output.split(","))
    return {"used_mib": used, "total_mib": total, "free_mib": total - used}


def capture_and_pin(geo, *, first, last, out, diagnose=False):
    report, client, eng, saved, metadata = _capture_host(
        geo, first=first, last=last, out=out, diagnose=diagnose,
    )
    out = Path(out)
    status_path = out / "gpu_cache_preload.json"
    host_bytes = sum(info["k_host_bytes"] + info["v_host_bytes"] for info in metadata)
    before = gpu_memory()
    # Reserve room for the single layer's Decode weights and FHE workspace.
    minimum_free_mib = (host_bytes + (1 << 20) - 1) // (1 << 20) + 2048
    status = {
        "mode": "gpu_resident_after_encrypted_prefill",
        "kv_from_fhe_prefill": True,
        "cache_bytes": host_bytes,
        "gpu_before": before,
        "minimum_free_mib": minimum_free_mib,
        "completed_layers": 0,
        "per_layer_upload_seconds": [],
    }
    if before["free_mib"] < minimum_free_mib:
        status["error"] = "insufficient GPU headroom before cache upload"
        status_path.write_text(json.dumps(status, indent=1))
        raise RuntimeError(status["error"])

    started = time.time()
    try:
        for index, host_cache in enumerate(saved):
            t0 = time.time()
            device_cache = {
                name: tuple(_upload(cipher) for cipher in host_cache[name])
                for name in ("k", "v")
            }
            torch.cuda.synchronize()
            saved[index] = device_cache
            status["per_layer_upload_seconds"].append(time.time() - t0)
            status["completed_layers"] += 1
            del host_cache, device_cache
        gc.collect()
        torch.cuda.empty_cache()
        status["upload_seconds"] = time.time() - started
        status["gpu_after"] = gpu_memory()
        status["passed"] = True
    except Exception as exc:
        status["upload_seconds"] = time.time() - started
        status["gpu_after"] = gpu_memory()
        status["passed"] = False
        status["error"] = repr(exc)
        status_path.write_text(json.dumps(status, indent=1))
        raise
    status_path.write_text(json.dumps(status, indent=1))
    return report, client, eng, saved, metadata


def reuse_if_resident(cipher):
    if all(part.is_cuda for part in cipher.cv):
        return cipher
    return _upload(cipher)


base.capture_prefill_caches = capture_and_pin
base.gpu_cipher = reuse_if_resident


if __name__ == "__main__":
    base.main()
