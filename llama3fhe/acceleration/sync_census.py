"""Census of host<->device synchronisations (``ODIN_SYNC_CENSUS=1``).

PyTorch's ``cuda.set_sync_debug_mode(1)`` warns on every synchronising CUDA call it knows about (``.item()``,
``.cpu()``, pageable host->device copies, ``cudaStreamSynchronize`` ...).  Each one drains the launch queue: the GPU
idles while the host runs Python up to the next kernel.  This module turns the warnings into a table of call sites
(innermost frame outside torch/easyfhe internals plus two callers), counted per :func:`take` window (a station or an
iteration).  Diagnostics only: the warnings machinery costs ~50 us per sync, so do not read walls from a census run.
"""
from __future__ import annotations

import collections
import os
import traceback
import warnings

_installed = False
_counts: collections.Counter = collections.Counter()
_total = 0
_SKIP = ("/warnings.py", "/easyfhe/cuda/", "/easyfhe/_tensor", "sync_census.py")


def enabled() -> bool:
    return os.environ.get("ODIN_SYNC_CENSUS", "0") == "1"


def _signature() -> str:
    frames = traceback.extract_stack(limit=14)
    picked = []
    for fr in reversed(frames):
        fn = fr.filename
        if any(s in fn for s in _SKIP):
            continue
        short = fn.rsplit("/site-packages/", 1)[-1] if "/site-packages/" in fn else "/".join(fn.rsplit("/", 3)[-3:])
        picked.append(f"{short}:{fr.lineno}:{fr.name}")
        if len(picked) == 3:
            break
    return " <- ".join(picked)


def install() -> dict:
    global _installed
    if _installed or not enabled():
        return {"installed": _installed}
    import easyfhe

    easyfhe.cuda.set_sync_debug_mode(1)
    warnings.simplefilter("always")
    prev = warnings.showwarning

    def hook(message, category, filename, lineno, file=None, line=None):
        global _total
        text = str(message)
        if "synchronizing CUDA operation" in text:
            _total += 1
            _counts[_signature()] += 1
            return
        if "Synchronization debug mode" in text:
            return
        prev(message, category, filename, lineno, file, line)

    warnings.showwarning = hook
    _installed = True
    return {"installed": True}


def take() -> dict:
    """Counts since the previous call (``{"total": n, "sites": [(count, signature), ...]}``) and reset."""
    global _total
    sites = sorted(((c, s) for s, c in _counts.items()), reverse=True)
    out = {"total": _total, "sites": sites}
    _counts.clear()
    _total = 0
    return out


def report(label: str, top: int = 25) -> dict:
    data = take()
    if not enabled():
        return data
    print(f"[sync-census] {label}: {data['total']} synchronising calls", flush=True)
    for count, sig in data["sites"][:top]:
        print(f"[sync-census]   {count:6d}  {sig}", flush=True)
    return data
